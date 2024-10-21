import time

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
import torch.optim as optim
import tqdm

from logging_setup import create_logger
from phi_3_5_constants import hidden_state_size, device

logger = create_logger(__name__)


def is_binary(labels: torch.Tensor | NDArray) -> bool:
    result = ((labels == 1) | (labels == 0)).all()
    return result.item() if isinstance(result, torch.Tensor) else result


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
    
    loss_fn = nn.BCELoss()
    optimizer = optim.AdamW(truth_probe.parameters(), lr=0.0001, weight_decay=0.01)
    
    n_epochs = 65_536   # number of epochs to run
    batch_size = 64  # size of each batch
    batch_start = torch.arange(0, num_train, batch_size).to(device)
    
    best_loss = np.inf
    best_epoch = -1
    best_weights = None
    best_bias = None
    
    prev_loss = np.inf
    prev_millenium_loss = np.inf
    
    probe_train_start_ts = time.time()
    logger.debug(f"starting to train probe on dataset of size {num_train} with validation set of size {num_val}")
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
            val_acc = (preds.round() == gpu_val_labels).float().mean().item()
            
        if epoch % 1000 == 0:
            curr_avg_loss = (val_loss + prev_loss)/2
            if epoch > 5000 and curr_avg_loss >= prev_millenium_loss:
                logger.warning(f"stopping early at epoch {epoch} because loss (avg'd over 2 timesteps) hasn't improved since 1000 epochs ago")
                break
            
            prev_millenium_loss = curr_avg_loss
        
        is_better = val_loss < prev_loss
        prev_loss = val_loss
        
        is_new_best = val_loss < best_loss
        if is_new_best:
            best_loss = val_loss
            best_epoch = epoch
            best_weights = truth_probe.output_w.weight.clone().detach().cpu()
            best_bias = truth_probe.output_w.bias.clone().detach().cpu()
            
            if epoch > 100 and val_loss < 1e-10:
                logger.info(f"stopping early at epoch {epoch}!")
                break
        
        logger.debug(f"{'!!! ' if is_new_best else ('< ' if is_better else '')
        }Epoch {epoch}: Val Loss: {val_loss:.10f}, Val Acc: {val_acc:.2f}")
    
    # restore truth_probe to use weights that resulted in best validation loss
    truth_probe.output_w.weight.data.copy_(best_weights.to(device))
    truth_probe.output_w.bias.data.copy_(best_bias.to(device))
    
    # Compute final validation metrics
    truth_probe.eval()
    with torch.no_grad():
        preds = truth_probe(gpu_val_activs)
        final_val_loss = loss_fn(preds, gpu_val_labels).item()
        final_val_acc = (preds.round() == gpu_val_labels).float().mean().item()
    
    logger.info(f"Using best Epoch {best_epoch}: Final Val Loss: {final_val_loss:.8f}, Final Val Acc: {final_val_acc:.8f};\n"
                f"Training took {time.time() - probe_train_start_ts:.3f} sec")
    return truth_probe
