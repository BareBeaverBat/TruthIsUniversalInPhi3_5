import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
import torch.optim as optim
import tqdm

from direction_learning import DirVectors
from logging_setup import create_logger
from phi_3_5_constants import hidden_state_size, device
from utils import is_binary

weight_decay_key = 'weight_decay'

learn_rate_key = 'lr'

logger = create_logger(__name__)


class PolarityAwareTruthProbe(nn.Module):
    def __init__(self, mean_activation: torch.Tensor, truth_dir: torch.Tensor, polarity_dir: torch.Tensor):
        super().__init__()
        assert 2 == mean_activation.ndim == truth_dir.ndim == polarity_dir.ndim
        assert 1 == mean_activation.shape[1] == truth_dir.shape[1] == polarity_dir.shape[1]
        self.activation_size = mean_activation.shape[0]
        assert self.activation_size == truth_dir.shape[0] == polarity_dir.shape[0]
        self.register_buffer('mean_activation', mean_activation)
        self.register_buffer('truth_dir', truth_dir)
        self.truth_dir_norm = np.linalg.norm(truth_dir)
        self.register_buffer('polarity_dir', polarity_dir)
        self.polarity_dir_norm = np.linalg.norm(polarity_dir)
        
        self.output_w = nn.Linear(2*self.activation_size, 1)
        self.activ = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor | NDArray):
        assert 2 == x.ndim
        assert self.activation_size == x.shape[1]
        centered_data = x - self.mean_activation.T
        truth_proj = ((centered_data @ self.truth_dir)/self.truth_dir_norm) * self.truth_dir.T
        polarity_proj = ((centered_data @ self.polarity_dir)/self.polarity_dir_norm) * self.polarity_dir.T
        
        transformed_x = torch.concat((truth_proj, polarity_proj), dim=1)
        return self.activ(self.output_w(transformed_x))


@dataclass
class ProbesForDataset:
    lyr18_probe: PolarityAwareTruthProbe
    lyr25_probe: PolarityAwareTruthProbe
    lyrs18_and_25_probe: PolarityAwareTruthProbe    

def get_optimizer_val(optimizer: torch.optim.Optimizer, param_key: str):
    return optimizer.param_groups[0][param_key]

def scale_lr_by(optimizer: torch.optim.Optimizer, factor: float):
    for param_group in optimizer.param_groups:
        param_group[learn_rate_key] = factor * param_group[learn_rate_key]

def shift_weight_decay_by(optimizer: torch.optim.Optimizer, offset: float):
    for param_group in optimizer.param_groups:
        param_group[weight_decay_key] = offset + param_group[weight_decay_key]

def train_probe(
        train_activations: torch.Tensor, train_truth_labels: torch.Tensor, val_activations: torch.Tensor,
        val_truth_labels: torch.Tensor, mean_train_activ: torch.Tensor, truth_dir: torch.Tensor,
        polarity_dir: torch.Tensor) -> PolarityAwareTruthProbe:
    """
    
    Credit to https://machinelearningmastery.com/building-a-binary-classification-truth_probe-in-pytorch/
    :param train_activations:
    :param train_truth_labels:
    :param val_activations:
    :param val_truth_labels:
    :param mean_train_activ:
    :param truth_dir:
    :param polarity_dir:
    :return:
    """
    assert (2 == train_activations.ndim == val_activations.ndim == train_truth_labels.ndim == val_truth_labels.ndim
            == mean_train_activ.ndim == truth_dir.ndim == polarity_dir.ndim)
    assert (1 == train_truth_labels.shape[1] == val_truth_labels.shape[1] == mean_train_activ.shape[1]
            == truth_dir.shape[1] == polarity_dir.shape[1])
    num_train = train_activations.shape[0]
    assert num_train == train_truth_labels.shape[0]
    assert is_binary(train_truth_labels), "Not all train-set truth labels are 1 or 0"
    assert is_binary(val_truth_labels), "Not all validation-set truth labels are 1 or 0"
    activ_vect_size = train_activations.shape[1]
    assert (activ_vect_size == val_activations.shape[1] == mean_train_activ.shape[0] == truth_dir.shape[0]
            == polarity_dir.shape[0])
    if activ_vect_size % hidden_state_size != 0:
        logger.warning(f"NOTE- not using phi 3.5 mini because activation vector size {activ_vect_size} is wrong")
    num_val = val_activations.shape[0]
    assert num_val == val_truth_labels.shape[0]
    
    truth_probe = PolarityAwareTruthProbe(mean_train_activ, truth_dir, polarity_dir)
    truth_probe.to(device)
    
    gpu_train_activs = train_activations.to(device)
    gpu_train_labels = train_truth_labels.to(device)
    gpu_val_activs = val_activations.to(device)
    gpu_val_labels = val_truth_labels.to(device)
    
    batch_size_factor =1/16 if num_train < 50 else 1/2 if num_train < 100 else 1 if num_train < 500 \
        else 2 if num_train < 1_000 else 4
    base_learning_rate = 0.0001
    base_learning_rate = base_learning_rate / batch_size_factor if batch_size_factor > 1 else base_learning_rate
    
    weight_decay = 0.03
    
    #weight_decay = weight_decay * math.sqrt(batch_size_factor) if batch_size_factor > 1 else weight_decay # not sure if this is needed or useful, just leaving it here in case it seems clearly worth trying later
    # TODO investigate why pos animal_class layer25 never starts grokking in latest run. maybe try several runs for that scenario and see if it just happens intermittently?
    #  Also need to confirm in logs from earlier experiments whether affirmative-animals layer 25 ever grokked.
    #   It might be that, as seen when deciding on layers, layer 25 is just much worse for distinguishing between true/false than layer 18, so there's only so far you can reduce validation error with an optimal solution (e.g. maybe you just can't get it below 0.0003959?)
        # this still doesn't explain why the layers-combined scenario never converges to at least a solution with the layer25 inputs ignored (i.e. the layer18-only solution, which does grok often?, at least in affirmative animals case)
    # TODO figure out a reasonable early stopping policy that doesn't rule out grokking?
    
    #TODO test whether affirmative animal class (lyr18) even needs weight decay to reach that super-nice solution
    
    #TODO investigate why affirmative animal class (lyr18) didn't grok in run6
    #  something to do with learning rate dropping to 1e-23! see whether this stops happening now that learning rate reductions are more moderate
    
    # keep training time from getting out of hand for the big datasets
    #  for context, many are 500 or less, a couple are 2k, one is ~4.5k, and one is ~22k
    # epoch_shrinkage_factor = 1
    # if num_train > 500:
    #     dataset_size_ratio = num_train//500
    #     epoch_shrinkage_factor = min(128, 2**math.floor(math.log2(dataset_size_ratio)))
    
    n_epochs = 1_048_576#//epoch_shrinkage_factor   # number of epochs to run
    
    batch_size = int(64*batch_size_factor)
    batch_start = torch.arange(0, num_train, batch_size).to(device)
    
    loss_fn = nn.BCELoss()
    optimizer = optim.AdamW(truth_probe.parameters(), lr=base_learning_rate, weight_decay=weight_decay)
    
    best_loss = np.inf
    best_epoch = -1
    best_weights = None
    best_bias = None
    
    num_prev_losses_tracked=10
    prev_few_losses = np.array([np.inf]*num_prev_losses_tracked)
    position_in_prev_losses= 0#rotary buffer
    val_loss_msgs_for_epoch_group: list[str] = []
    
    prev_epoch_group_loss = np.inf
    had_prev_epoch_group_been_improvement = False
    num_stalls_in_epoch_group = 0
    
    num_epochs_in_group = 1024#//epoch_shrinkage_factor
    
    #for larger epoch numbers, this will be displayed without softwrap in notepad++ on my laptop ~only if few < or ! prefixes on epoch losses, making epochs with those prefixes stand out more
    num_epoch_losses_per_log_line=7
    
    def print_epoch_group_losses(latest_epoch: int, loss_change_over_group: float):
        log_msg_for_epoch_group = f"Val losses for {len(val_loss_msgs_for_epoch_group)} epochs up to epoch {latest_epoch} (delta of {loss_change_over_group}):\n"
        for line_idx in range(0, len(val_loss_msgs_for_epoch_group), num_epoch_losses_per_log_line):
            max_epoch_idx_in_line = line_idx + num_epoch_losses_per_log_line
            log_msg_for_epoch_group += '; '.join(val_loss_msgs_for_epoch_group[line_idx:max_epoch_idx_in_line]) + '\n'
        logger.debug(log_msg_for_epoch_group)
        val_loss_msgs_for_epoch_group.clear()
    
    probe_train_start_ts = time.time()
    logger.debug(f"starting to train probe on dataset of size {num_train} with validation set of size {num_val}")
    epoch=0
    for epoch in range(n_epochs):
        truth_probe.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                batch_activs, batch_labels = gpu_train_activs[start:start+batch_size], gpu_train_labels[start:start+batch_size]
                preds = truth_probe(batch_activs)
                loss = loss_fn(preds, batch_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                acc = (preds.round() == batch_labels).float().mean().item()
                bar.set_postfix(loss=loss.item(), acc=acc)
        
        # evaluate loss on validation set
        truth_probe.eval()
        with torch.no_grad():
            preds = truth_probe(gpu_val_activs)
            val_loss = loss_fn(preds, gpu_val_labels).item()
        
        is_better = all(val_loss < prev_few_losses)
        prev_few_losses[position_in_prev_losses] = val_loss
        position_in_prev_losses = (position_in_prev_losses + 1) % num_prev_losses_tracked
        
        is_new_best = val_loss < best_loss
        
        val_loss_msgs_for_epoch_group.append(f"{'! ' if is_new_best else ('<' if is_better else '')
        }{epoch}:{val_loss:.7e}")
        if not is_better:
            num_stalls_in_epoch_group += 1
        
        if epoch and epoch % num_epochs_in_group == 0:
            curr_avg_loss = np.mean(prev_few_losses)
            loss_delta_over_group = curr_avg_loss - prev_epoch_group_loss
            print_epoch_group_losses(epoch, loss_delta_over_group)
            curr_lr = get_optimizer_val(optimizer, learn_rate_key)
            if epoch > 3*num_epochs_in_group:
                if num_stalls_in_epoch_group > 0.7*num_epochs_in_group:
                    if optimizer.param_groups[0][weight_decay_key] < 0.2:
                        logger.warning(f"increasing weight decay from {get_optimizer_val(optimizer, weight_decay_key):e} at epoch {epoch} because (over last {num_epochs_in_group} timesteps) validation loss hasn't even been close to improving smoothly- more than 40% of the last {num_epochs_in_group} epochs have been stagnant")
                        shift_weight_decay_by(optimizer, 0.01)
                    elif loss_delta_over_group >= 0:
                        logger.warning(f"terminating run early at epoch {epoch} because loss (avg'd over {num_prev_losses_tracked} timesteps) has increased by {loss_delta_over_group:e} since {num_epochs_in_group} epochs ago and there have been so many mostly stagnant periods in earlier epoch groups that the weight decay has already been increased to its maximum")
                        break
                    else:
                        logger.info(f"at epoch {epoch}, last {num_epochs_in_group} epochs had a lot of stalls and weight decay has already been boosted to its maximum, but loss has improved by {-loss_delta_over_group:e} since {num_epochs_in_group} epochs ago, so continuing")
                elif loss_delta_over_group < 0 and had_prev_epoch_group_been_improvement and curr_lr < base_learning_rate:
                    # if it finally gets on a good trajectory, but only after many cuts-in-learning-rate/increases-in-weight-decay
                    #  otherwise, it can spend literally hundreds of thousands of epochs making improvement in every 1024-epoch group
                    #  relative to the prior group and yet still have a loss above 0.1 after all of that time
                    #  (because the updates were all way too small)
                    logger.info(f"scaling learning rate up from {curr_lr:e} at epoch {epoch} because loss (avg'd over {num_prev_losses_tracked} timesteps) has improved by {-loss_delta_over_group:e} since {num_epochs_in_group} epochs ago and because most of the last {num_epochs_in_group} epochs were locally improving the validation loss")
                    scale_lr_by(optimizer, 1.2)
                        
                if loss_delta_over_group >= 0:
                    logger.warning(f"shrinking learning rate from {curr_lr:e} at epoch {epoch} because loss (avg'd over {num_prev_losses_tracked} timesteps) has increased by {loss_delta_over_group:e} since {num_epochs_in_group} epochs ago")
                    scale_lr_by(optimizer, 0.707)

            
            if loss_delta_over_group < 0 and num_stalls_in_epoch_group < 50:
                logger.info(f"scaling learning rate up from {curr_lr:e} at epoch {epoch} because loss (avg'd over {num_prev_losses_tracked} timesteps) has improved by {-loss_delta_over_group:e} since {num_epochs_in_group} epochs ago and last {num_epochs_in_group} epochs have included a minimal number of stagnant or backsliding epochs")
                scale_lr_by(optimizer, 1.2)
            
            prev_epoch_group_loss = curr_avg_loss
            had_prev_epoch_group_been_improvement = loss_delta_over_group < 0
            num_stalls_in_epoch_group = 0
        
        if is_new_best:
            best_loss = val_loss
            best_epoch = epoch
            best_weights = truth_probe.output_w.weight.clone().detach().cpu()
            best_bias = truth_probe.output_w.bias.clone().detach().cpu()
            
            if epoch > 100 and val_loss < 1e-14:
                print_epoch_group_losses(epoch, 0)
                logger.info(f"stopping early at epoch {epoch}!")
                break
        
        # logger.debug(f"{'!!! ' if is_new_best else ('< ' if is_better else '')}Epoch {epoch}: Val Loss: {val_loss:.16f}, Val Acc: {val_acc:.2f}")
    
    # restore truth_probe to use weights that resulted in best validation loss
    truth_probe.output_w.weight.data.copy_(best_weights.to(device))
    truth_probe.output_w.bias.data.copy_(best_bias.to(device))
    
    # Compute final validation metrics
    truth_probe.eval()
    with torch.no_grad():
        preds = truth_probe(gpu_val_activs)
        final_val_loss = loss_fn(preds, gpu_val_labels).item()
        final_val_acc = (preds.round() == gpu_val_labels).float().mean().item()
    
    train_time_in_secs = time.time() - probe_train_start_ts
    logger.info(f"Using best Epoch {best_epoch}: Final Val Loss: {final_val_loss:.6e}, Final Val Acc: {final_val_acc:.12%};\n"
                f"Training took {train_time_in_secs // 60} min, {train_time_in_secs % 60:.3f} sec with final learning rate {get_optimizer_val(optimizer, learn_rate_key):e} and final weight decay {get_optimizer_val(optimizer, weight_decay_key):e}, ending at epoch {epoch}")
    return truth_probe

def train_probes_for_dset(output_folder: Path, output_nm_prefix: str, train_activs: torch.Tensor, 
                          train_truth_labels: torch.Tensor, val_activs: torch.Tensor, val_truth_labels: torch.Tensor,
                          dset_dirs: DirVectors) -> ProbesForDataset:
    assert 3 == train_activs.ndim == val_activs.ndim
    assert 2 == train_activs.shape[0] == val_activs.shape[0]
    assert 2 == train_truth_labels.ndim == val_truth_labels.ndim
    assert 1 == train_truth_labels.shape[1] == val_truth_labels.shape[1]
    num_train_records = train_activs.shape[1]
    assert num_train_records == train_truth_labels.shape[0]
    num_val_records = val_activs.shape[1]
    assert num_val_records == val_truth_labels.shape[0]    
    assert is_binary(train_truth_labels)
    assert is_binary(val_truth_labels)
    activs_size = train_activs.shape[2]
    assert activs_size == val_activs.shape[2] == dset_dirs.lyr18_mean_activ.shape[0]
    if activs_size != hidden_state_size:
        logger.warning(f"dataset of activations isn't from phi 3.5 mini because activation size {activs_size} is wrong")
    
    output_folder.mkdir(exist_ok=True)
    
    lyr18_train_activs = train_activs[0, :, :]
    lyr25_train_activs = train_activs[1, :, :]
    lyr18_val_activs = val_activs[0, :, :]
    lyr25_val_activs = val_activs[1, :, :]
    
    lyr18_probe_save_location = output_folder / f"{output_nm_prefix}_lyr18_probe.pth"    
    if lyr18_probe_save_location.exists():
        logger.info(f"skipping the training of the layer18 probe for {num_train_records} records of data {output_nm_prefix} in the location {output_folder} because the file {lyr18_probe_save_location} already exists")
        lyr18_probe = PolarityAwareTruthProbe(torch.ones(activs_size, 1), torch.ones(activs_size, 1), torch.ones(activs_size, 1))
        lyr18_probe_state_dict = torch.load(lyr18_probe_save_location, weights_only=True)
        lyr18_probe.load_state_dict(lyr18_probe_state_dict)
    else:
        logger.info(f"training the layer18 probe for {num_train_records} records of data {output_nm_prefix} in the location {output_folder}")
        lyr18_probe = train_probe(lyr18_train_activs, train_truth_labels, lyr18_val_activs, val_truth_labels, 
                                  dset_dirs.lyr18_mean_activ, dset_dirs.lyr18_truth_dir, dset_dirs.lyr18_polarity_dir)
        torch.save(lyr18_probe.state_dict(), lyr18_probe_save_location)
    
    lyr25_probe_save_location = output_folder / f"{output_nm_prefix}_lyr25_probe.pth"
    if lyr25_probe_save_location.exists():
        logger.info(f"skipping the training of the layer25 probe for {num_train_records} records of data {output_nm_prefix} in the location {output_folder} because the file {lyr25_probe_save_location} already exists")
        lyr25_probe = PolarityAwareTruthProbe(torch.ones(activs_size, 1), torch.ones(activs_size, 1), torch.ones(activs_size, 1))
        lyr25_probe_state_dict = torch.load(lyr25_probe_save_location, weights_only=True)
        lyr25_probe.load_state_dict(lyr25_probe_state_dict)
    else:
        logger.info(f"training the layer25 probe for {num_train_records} records of data {output_nm_prefix} in the location {output_folder}")
        lyr25_probe = train_probe(lyr25_train_activs, train_truth_labels, lyr25_val_activs, val_truth_labels,
                                  dset_dirs.lyr25_mean_activ, dset_dirs.lyr25_truth_dir, dset_dirs.lyr25_polarity_dir)
        torch.save(lyr25_probe.state_dict(), lyr25_probe_save_location)
    
    lyrs18_and_25_probe_save_location = output_folder / f"{output_nm_prefix}_lyrs18_and_25_probe.pth"
    if lyrs18_and_25_probe_save_location.exists():
        logger.info(f"skipping the training of the layers18 and 25 probe for {num_train_records} records of data {output_nm_prefix} in the location {output_folder} because the file {lyrs18_and_25_probe_save_location} already exists")
        lyrs18_and_25_probe = PolarityAwareTruthProbe(torch.ones(2*activs_size, 1), torch.ones(2*activs_size, 1), torch.ones(2*activs_size, 1))
        lyrs18_and_25_probe_state_dict = torch.load(lyrs18_and_25_probe_save_location, weights_only=True)
        lyrs18_and_25_probe.load_state_dict(lyrs18_and_25_probe_state_dict)
    else:
        logger.info(f"training the layers18 and 25 probe for {num_train_records} records of data {output_nm_prefix} in the location {output_folder}")
        lyrs18_and_25_train_activs = torch.concat((lyr18_train_activs, lyr25_train_activs), dim=1)
        lyrs18_and_25_val_activs = torch.concat((lyr18_val_activs, lyr25_val_activs), dim=1)
        lyrs18_and_25_probe = train_probe(lyrs18_and_25_train_activs, train_truth_labels, lyrs18_and_25_val_activs, val_truth_labels,
                                  dset_dirs.lyrs18_and_25_mean_activ, dset_dirs.lyrs18_and_25_truth_dir, dset_dirs.lyrs18_and_25_polarity_dir)
        torch.save(lyrs18_and_25_probe.state_dict(), lyrs18_and_25_probe_save_location)
    
    return ProbesForDataset(lyr18_probe, lyr25_probe, lyrs18_and_25_probe)