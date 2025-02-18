import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float
import torch.optim as optim
import tqdm

from .data_management import DataComponents
from .direction_learning import learn_directions_for_dset
from .logging_setup import create_logger
from .phi_3_5_constants import hidden_state_size, device, probes_folder, baseline_probes_folder, num_splits, \
    calc_seeds_for_splits, vect_norm_tol
from .utils import is_binary, greatest_power_of_two_below, float_eq, bear_jax_typed_with_independent_calls

weight_decay_key = 'weight_decay'
learn_rate_key = 'lr'

logger = create_logger(__name__)


class PolarityAwareTruthProbe(nn.Module):

    @bear_jax_typed_with_independent_calls
    def __init__(self, truth_dir: Float[torch.Tensor, "act_sz 1"], polarity_dir: Float[torch.Tensor, "act_sz 1"]):
        super().__init__()
        truth_dir_norm = torch.linalg.vector_norm(truth_dir).item()
        assert float_eq(truth_dir_norm, 1, vect_norm_tol), f"truth direction norm is {truth_dir_norm} not unit"
        polarity_dir_norm = torch.linalg.vector_norm(polarity_dir).item()
        assert float_eq(polarity_dir_norm, 1, vect_norm_tol) or float_eq(polarity_dir_norm, 0, vect_norm_tol), \
            f"polarity direction not unit norm or 0 norm, instead norm is {polarity_dir_norm}"
        self.activation_size = truth_dir.shape[0]
        tp_transform: Float[torch.Tensor, "act_sz 2"] = torch.cat((truth_dir, polarity_dir), dim=1)
        self.register_buffer('tp_transform', tp_transform)

        self.output_w = nn.Linear(2, 1, bias=True)
        self.activ = nn.Sigmoid()

    @bear_jax_typed_with_independent_calls
    def forward(self, x: Float[torch.Tensor, "batch act_sz"] | Float[torch.Tensor, "batch 2"],
                is_already_projected=False):
        n_feats = x.shape[1]
        assert ((not is_already_projected and n_feats == self.activation_size) or
                (is_already_projected and n_feats == 2)), f"{is_already_projected}, {n_feats}, {self.activation_size}"
        projected_x: Float[torch.Tensor, "batch 2"] = x if is_already_projected else x @ self.tp_transform
        return self.activ(self.output_w(projected_x))

    @property
    def truth_dir(self) -> Float[torch.Tensor, "act_sz 1"]:
        return self.tp_transform[:, 0:1]

    @property
    def polarity_dir(self) -> Float[torch.Tensor, "act_sz 1"]:
        return self.tp_transform[:, 1:2]


class LinearProbe(nn.Module):
    def __init__(self, activation_size: int):
        super().__init__()
        self.activation_size = activation_size

        self.output_w = nn.Linear(self.activation_size, 1, bias=True)
        self.activ = nn.Sigmoid()

    @bear_jax_typed_with_independent_calls
    def forward(self, x: Float[torch.Tensor, "batch {self.activation_size}"]):
        return self.activ(self.output_w(x))


@dataclass
class ProbesForScenario:
    ttpd_probe: PolarityAwareTruthProbe
    baseline_linear_probe: LinearProbe


def get_optimizer_val(optimizer: torch.optim.Optimizer, param_key: str):
    return optimizer.param_groups[0][param_key]


def scale_lr_by(optimizer: torch.optim.Optimizer, factor: float):
    for param_group in optimizer.param_groups:
        param_group[learn_rate_key] = factor * param_group[learn_rate_key]


def shift_weight_decay_by(optimizer: torch.optim.Optimizer, offset: float):
    for param_group in optimizer.param_groups:
        param_group[weight_decay_key] = offset + param_group[weight_decay_key]


@bear_jax_typed_with_independent_calls
def train_probe(
        train_activs: Float[torch.Tensor, "n_t_recs act_sz"] | Float[torch.Tensor, "n_t_recs 2"],
        train_truth_labels: Float[torch.Tensor, "n_t_recs 1"],
        val_activs: Float[torch.Tensor, "n_v_recs act_sz"] | Float[torch.Tensor, "n_v_recs 2"],
        val_truth_labels: Float[torch.Tensor, "n_v_recs 1"],
        probe: PolarityAwareTruthProbe | LinearProbe, seed_to_use: int, probe_fwd_kwargs: dict[str, Any] = None):
    """
    
    Credit to https://machinelearningmastery.com/building-a-binary-classification-truth_probe-in-pytorch/
    :param train_activs:
    :param train_truth_labels:
    :param val_activs:
    :param val_truth_labels:
    :param probe:
            The probe's weights/buffers will be moved back to the CPU before this function returns
    :param probe_fwd_kwargs: extra keyword argument's for the probe's forward method
    """
    probe_fwd_kwargs = probe_fwd_kwargs or {}

    assert is_binary(train_truth_labels), "Not all train-set truth labels are 1 or 0"
    assert is_binary(val_truth_labels), "Not all validation-set truth labels are 1 or 0"
    n_data_features = train_activs.shape[1]
    assert n_data_features == val_activs.shape[1]
    num_train = train_activs.shape[0]
    num_val = val_activs.shape[0]

    probe.to(device)
    
    orig_gpu_train_activs = train_activs.to(device)
    orig_gpu_train_labels = train_truth_labels.to(device)

    gpu_val_activs = val_activs.to(device)
    gpu_val_labels = val_truth_labels.to(device)

    torch_rng = torch.Generator(device=device)
    torch_rng.manual_seed(seed_to_use)
    
    base_batch_size = 64
    # Break the dataset into just 2-3 batches if there are only 2 variables, otherwise scale the batch size as large as
    #  possible without making it too large for the GPU or the dataset size
    batch_size_factor = ((greatest_power_of_two_below(num_train)//2) // base_batch_size) if n_data_features == 2 \
        else (1/16 if num_train < 50 else 1/2 if num_train < 100 else 1 if num_train < 500 else 4 if num_train < 1_000
              else 8)

    base_learning_rate = 0.0001
    base_learning_rate = base_learning_rate / batch_size_factor if batch_size_factor > 1 else base_learning_rate
    
    weight_decay = 0.03

    max_num_epochs = 1_048_576

    batch_size = int(base_batch_size * batch_size_factor)
    batch_start_idxs = torch.arange(0, num_train, batch_size).to(device)
    
    loss_fn = nn.BCELoss()
    optimizer = optim.AdamW(probe.parameters(), lr=base_learning_rate, weight_decay=weight_decay)
    
    best_loss = np.inf
    best_epoch = -1
    best_weights = None
    best_bias = None
    lr_at_best_loss = -1
    weight_decay_at_best_loss = -1
    largest_num_consecutive_stall_heavy_epoch_groups_before_best_loss = -1

    val_loss = -1
    num_prev_losses_tracked = 10
    # rotary buffer
    prev_few_losses = np.array([np.inf]*num_prev_losses_tracked)
    position_in_prev_losses = 0
    val_loss_msgs_for_epoch_group: list[str] = []
    
    prev_epoch_group_loss = np.inf
    had_prev_epoch_group_been_improvement = False
    num_stalls_in_epoch_group = 0
    
    num_epochs_in_group = 1024
    
    #for larger epoch numbers, this will be displayed without softwrap in notepad++ on my laptop ~only if few < or ! prefixes on epoch losses, making epochs with those prefixes stand out more
    num_epoch_losses_per_log_line = 7
    
    def print_epoch_group_losses(latest_epoch: int, loss_change_over_group: float):
        log_msg_for_epoch_group = f"Val losses for {len(val_loss_msgs_for_epoch_group)} epochs up to epoch {latest_epoch} (delta of {loss_change_over_group}):\n"
        for line_idx in range(0, len(val_loss_msgs_for_epoch_group), num_epoch_losses_per_log_line):
            max_epoch_idx_in_line = line_idx + num_epoch_losses_per_log_line
            log_msg_for_epoch_group += '; '.join(val_loss_msgs_for_epoch_group[line_idx:max_epoch_idx_in_line]) + '\n'
        logger.debug(log_msg_for_epoch_group)
        val_loss_msgs_for_epoch_group.clear()

    curr_num_consecutive_stall_heavy_epoch_groups = 0
    largest_num_consecutive_stall_heavy_epoch_groups = 0

    probe_train_start_ts = time.time()
    logger.debug(f"starting to train probe on dataset of size {num_train} with validation set of size {num_val}")
    epoch = 0
    for epoch in range(max_num_epochs):
        probe.train()
        with tqdm.tqdm(batch_start_idxs, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            train_recs_permut = torch.randperm(num_train, generator=torch_rng, device=device)
            gpu_train_activs = orig_gpu_train_activs[train_recs_permut]
            gpu_train_labels = orig_gpu_train_labels[train_recs_permut]

            for start in bar:
                batch_activs = gpu_train_activs[start:start+batch_size]
                batch_labels = gpu_train_labels[start:start+batch_size]
                preds = probe(batch_activs, **probe_fwd_kwargs)
                loss = loss_fn(preds, batch_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                acc = (preds.round() == batch_labels).float().mean().item()
                bar.set_postfix(loss=loss.item(), acc=acc)
        
        # evaluate loss on validation set
        probe.eval()
        with torch.no_grad():
            preds = probe(gpu_val_activs, **probe_fwd_kwargs)
            val_loss = loss_fn(preds, gpu_val_labels).item()
        
        is_better = all(val_loss < prev_few_losses)
        prev_few_losses[position_in_prev_losses] = val_loss
        position_in_prev_losses = (position_in_prev_losses + 1) % num_prev_losses_tracked
        
        is_new_best = val_loss < best_loss
        
        val_loss_msgs_for_epoch_group.append(f"{'! ' if is_new_best else ('<' if is_better else '')}"
                                             f"{epoch}:{val_loss:.7e}")
        if not is_better:
            num_stalls_in_epoch_group += 1

        curr_lr = get_optimizer_val(optimizer, learn_rate_key)
        curr_weight_decay = get_optimizer_val(optimizer, weight_decay_key)
        if epoch and epoch % num_epochs_in_group == 0:
            curr_avg_loss = np.mean(prev_few_losses)
            loss_delta_over_group = curr_avg_loss - prev_epoch_group_loss
            print_epoch_group_losses(epoch, loss_delta_over_group)

            if epoch > 3*num_epochs_in_group:
                if num_stalls_in_epoch_group > 0.7*num_epochs_in_group:
                    curr_num_consecutive_stall_heavy_epoch_groups += 1
                    if curr_num_consecutive_stall_heavy_epoch_groups > largest_num_consecutive_stall_heavy_epoch_groups:
                        largest_num_consecutive_stall_heavy_epoch_groups = curr_num_consecutive_stall_heavy_epoch_groups
                    if optimizer.param_groups[0][weight_decay_key] < 0.2:
                        shift_weight_decay_by(optimizer, 0.02)
                        logger.warning(f"increasing weight decay to {get_optimizer_val(optimizer, weight_decay_key):.4f} at epoch {epoch} because (over last {num_epochs_in_group} timesteps) validation loss hasn't even been close to improving smoothly- more than 70% of the last {num_epochs_in_group} epochs have been stagnant")
                    elif loss_delta_over_group >= 0:
                        logger.warning(f"terminating run early at epoch {epoch} because loss (avg'd over {num_prev_losses_tracked} timesteps) has increased by {loss_delta_over_group:e} since {num_epochs_in_group} epochs ago and there have been so many mostly stagnant periods in earlier epoch groups that the weight decay has already been increased to its maximum")
                        break
                    else:
                        logger.info(f"at epoch {epoch}, last {num_epochs_in_group} epochs had a lot of stalls and weight decay has already been boosted to its maximum, but loss has improved by {-loss_delta_over_group:e} since {num_epochs_in_group} epochs ago, so continuing")
                        if ((curr_avg_loss - best_loss) > 0 and
                                loss_delta_over_group / (curr_avg_loss - best_loss) < 0.01):
                            scale_lr_by(optimizer, 1.3)
                            logger.info(f"at epoch {epoch}, the loss improvement over the previous {num_epochs_in_group} epochs was less than 1% of the difference between the best loss so far and the loss at the end of the previous {num_epochs_in_group} epochs, so increasing learning rate to {get_optimizer_val(optimizer, learn_rate_key):e}")
                else:
                    curr_num_consecutive_stall_heavy_epoch_groups = 0
                    if (loss_delta_over_group < 0 and had_prev_epoch_group_been_improvement
                            and curr_lr < base_learning_rate):
                        # if it finally gets on a good trajectory, but only after many cuts-in-learning-rate
                        #  /increases-in-weight-decay
                        #  otherwise, it can spend literally hundreds of thousands of epochs making improvement in every
                        #  1024-epoch group relative to the prior group and yet still have a loss above 0.1 after all of
                        #  that time (because the updates were all way too small)
                        scale_lr_by(optimizer, 1.2)
                        logger.info(f"scaling learning rate up from {curr_lr:e} to {get_optimizer_val(optimizer, learn_rate_key)} at epoch {epoch} because loss (avg'd over {num_prev_losses_tracked} timesteps) has improved by {-loss_delta_over_group:e} since {num_epochs_in_group} epochs ago and because most of the last {num_epochs_in_group} epochs were locally improving the validation loss")
                    
                if loss_delta_over_group >= 0 and not had_prev_epoch_group_been_improvement:
                    scale_lr_by(optimizer, 0.707)
                    logger.warning(f"shrinking learning rate from {curr_lr:e} to {get_optimizer_val(optimizer, learn_rate_key)} at epoch {epoch} because loss (avg'd over {num_prev_losses_tracked} timesteps) has increased by {loss_delta_over_group:e} since {num_epochs_in_group} epochs ago and previous epoch group hadn't improved the loss either")
            
            if loss_delta_over_group < 0 and num_stalls_in_epoch_group < 50:
                scale_lr_by(optimizer, 1.2)
                logger.info(f"scaling learning rate up from {curr_lr:e} to {get_optimizer_val(optimizer, learn_rate_key)} at epoch {epoch} because loss (avg'd over {num_prev_losses_tracked} timesteps) has improved by {-loss_delta_over_group:e} since {num_epochs_in_group} epochs ago and last {num_epochs_in_group} epochs have included a minimal number of stagnant or backsliding epochs")
            
            prev_epoch_group_loss = curr_avg_loss
            had_prev_epoch_group_been_improvement = loss_delta_over_group < 0
            num_stalls_in_epoch_group = 0
        
        if is_new_best:
            best_loss = val_loss
            best_epoch = epoch
            best_weights = probe.output_w.weight.clone().detach().cpu()
            best_bias = probe.output_w.bias.clone().detach().cpu() if probe.output_w.bias is not None else None
            lr_at_best_loss = curr_lr
            weight_decay_at_best_loss = curr_weight_decay
            largest_num_consecutive_stall_heavy_epoch_groups_before_best_loss = (
                largest_num_consecutive_stall_heavy_epoch_groups)
            
            if epoch > 100 and val_loss < 1e-14:
                print_epoch_group_losses(epoch, 0)
                logger.info(f"stopping early at epoch {epoch}!")
                break

    # restore truth_probe to use weights that resulted in the best validation loss
    probe.output_w.weight.data.copy_(best_weights.to(device))
    if best_bias is not None:
        probe.output_w.bias.data.copy_(best_bias.to(device))
    
    # Compute final validation metrics
    probe.eval()
    with torch.no_grad():
        preds = probe(gpu_val_activs, **probe_fwd_kwargs)
        final_val_loss = loss_fn(preds, gpu_val_labels).item()
        final_val_acc = (preds.round() == gpu_val_labels).float().mean().item()
    
    train_time_in_secs = time.time() - probe_train_start_ts
    n_recs = num_train + num_val
    num_secs_per_rec = train_time_in_secs / n_recs
    logger.info(f"Using best Epoch {best_epoch} out of {epoch}: Val Loss with best weights: {final_val_loss:.6e}, Val Acc with best weights: {final_val_acc:.12%}; Val loss with terminal epoch's weights: {val_loss:.6e}\n"
                f"Training took {train_time_in_secs // 60} min, {train_time_in_secs % 60:.3f} sec on a dataset with {n_recs:.3f} records, for a rate of {num_secs_per_rec} seconds per data record; final learning rate {get_optimizer_val(optimizer, learn_rate_key):e} and final weight decay {get_optimizer_val(optimizer, weight_decay_key):.4f}"
                f"\nAfter the epoch {best_epoch} with the best loss {best_loss:.6e}, learning rate= {lr_at_best_loss:e} and weight decay={weight_decay_at_best_loss:.4f}; Before the best loss was achieved, the longest set of consecutive epoch groups with mostly stagnant or backsliding validation losses was of length {largest_num_consecutive_stall_heavy_epoch_groups_before_best_loss}")
    probe.cpu()


@bear_jax_typed_with_independent_calls
def train_probes_for_dset(
        output_subfolder: str, output_nm_prefix: str, split_variant_idx: int, train_data: DataComponents,
        val_data: DataComponents) -> ProbesForScenario:
    num_train_records = train_data.activations.shape[0]
    assert is_binary(train_data.truth_labels)
    assert is_binary(val_data.truth_labels)
    activs_size = train_data.activations.shape[1]
    if activs_size != hidden_state_size:
        logger.warning(f"dataset of activations isn't from phi 3.5 mini because activation size {activs_size} is wrong")

    output_folder = probes_folder / output_subfolder if output_subfolder else probes_folder
    baseline_output_folder = baseline_probes_folder / output_subfolder if output_subfolder else baseline_probes_folder
    retrieval_result = try_load_dset_probes(output_folder, baseline_output_folder, output_nm_prefix, split_variant_idx,
                                            activs_size)
    ttpd_probe = retrieval_result.ttpd_probe
    baseline_linear_probe = retrieval_result.baseline_linear_probe

    assert split_variant_idx < num_splits, f"{split_variant_idx} >= {num_splits}"
    seed_for_curr_split = calc_seeds_for_splits()[split_variant_idx]

    if not ttpd_probe:
        logger.info(f"training the layer18 probe for {num_train_records} records of data {output_nm_prefix} "
                    f"in the location {output_folder}")
        dset_dirs = learn_directions_for_dset(train_data.activations, train_data.truth_labels,
                                              train_data.polarity_labels)

        ttpd_probe = PolarityAwareTruthProbe(dset_dirs.truth_dir, dset_dirs.polarity_dir)
        projected_train_activs: Float[torch.Tensor, "n_t_recs 2"] = train_data.activations @ ttpd_probe.tp_transform
        projected_val_activs: Float[torch.Tensor, "n_v_recs 2"] = val_data.activations @ ttpd_probe.tp_transform
        train_probe(projected_train_activs, train_data.truth_labels, projected_val_activs, val_data.truth_labels,
                    ttpd_probe, seed_for_curr_split, {"is_already_projected": True})
        retrieval_result.ttpd_probe_save_location.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ttpd_probe.state_dict(), retrieval_result.ttpd_probe_save_location)

    if not baseline_linear_probe:
        logger.info(f"training the baseline linear probe for {num_train_records} records of data {output_nm_prefix} in "
                    f"the location {baseline_output_folder}")
        baseline_linear_probe = LinearProbe(activs_size)
        train_probe(train_data.activations, train_data.truth_labels, val_data.activations, val_data.truth_labels,
                    baseline_linear_probe, seed_for_curr_split)
        retrieval_result.baseline_linear_probe_save_location.parent.mkdir(parents=True, exist_ok=True)
        torch.save(baseline_linear_probe.state_dict(), retrieval_result.baseline_linear_probe_save_location)

    return ProbesForScenario(ttpd_probe, baseline_linear_probe)


def load_probes_for_dset(subfolder_for_dset_probes: str, output_nm_prefix: str, split_variant_idx: int,
                         activations_size=hidden_state_size
                         ) -> ProbesForScenario:
    output_folder = probes_folder / subfolder_for_dset_probes if subfolder_for_dset_probes else probes_folder
    baseline_output_folder = baseline_probes_folder / subfolder_for_dset_probes if subfolder_for_dset_probes \
        else baseline_probes_folder

    result = try_load_dset_probes(output_folder, baseline_output_folder, output_nm_prefix, split_variant_idx,
                                  activations_size)
    if result.ttpd_probe and result.baseline_linear_probe:
        return ProbesForScenario(result.ttpd_probe, result.baseline_linear_probe)
    else:
        raise FileNotFoundError(
            f"Couldn't load all probes for dataset; missing probes' locations:"
            f"\n{'' if result.ttpd_probe else result.ttpd_probe_save_location}"
            f"\n{'' if result.baseline_linear_probe else result.baseline_linear_probe_save_location}"
        )


@dataclass
class ProbesForDatasetRetrievalResult:
    ttpd_probe: PolarityAwareTruthProbe | None
    baseline_linear_probe: LinearProbe | None
    ttpd_probe_save_location: Path
    baseline_linear_probe_save_location: Path


def try_load_dset_probes(dset_probes_folder: Path, dset_baseline_probes_folder: Path, output_nm_prefix: str,
                         split_variant_idx: int, activations_size=hidden_state_size
                         ) -> ProbesForDatasetRetrievalResult:
    ttpd_probe_save_location = (
            dset_probes_folder / f"{output_nm_prefix}_lyr18_probe" / f"split-variant-{split_variant_idx}.pth")
    baseline_linear_probe_save_location = (
            dset_baseline_probes_folder / f"{output_nm_prefix}_lyr18_baseline_linear_probe"
            / f"split-variant-{split_variant_idx}.pth")

    retrieval_result = ProbesForDatasetRetrievalResult(None, None, ttpd_probe_save_location,
                                                       baseline_linear_probe_save_location)

    if ttpd_probe_save_location.exists():
        unit_norm_activ_vect = torch.ones(activations_size, 1)
        unit_norm_activ_vect = unit_norm_activ_vect / torch.linalg.vector_norm(unit_norm_activ_vect)
        ttpd_probe = PolarityAwareTruthProbe(unit_norm_activ_vect, unit_norm_activ_vect.clone())
        ttpd_probe_state_dict = torch.load(ttpd_probe_save_location, weights_only=True)
        ttpd_probe.load_state_dict(ttpd_probe_state_dict)
        retrieval_result.ttpd_probe = ttpd_probe
    if baseline_linear_probe_save_location.exists():
        baseline_linear_probe = LinearProbe(activations_size)
        baseline_linear_probe_state_dict = torch.load(baseline_linear_probe_save_location, weights_only=True)
        baseline_linear_probe.load_state_dict(baseline_linear_probe_state_dict)
        retrieval_result.baseline_linear_probe = baseline_linear_probe

    return retrieval_result

