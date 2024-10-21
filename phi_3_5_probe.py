import time

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
import torch.optim as optim
import tqdm
from scipy.optimize import least_squares

from logging_setup import create_logger, StdoutToLoggerRedirection
from phi_3_5_constants import hidden_state_size, device

logger = create_logger(__name__)


def is_bipolar(labels: torch.Tensor | NDArray) -> bool:
    abs_vals = labels.abs() if isinstance(labels, torch.Tensor) else abs(labels)
    result = (abs_vals == 1).all()
    return result.item() if isinstance(result, torch.Tensor) else result

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
    
        
    

def solve_for_truth_polarity_vectors(
        centered_activations_data: torch.Tensor, truth_labels: torch.Tensor, polarity_labels: torch.Tensor,
        np_rng: np.random.Generator) -> (torch.Tensor, torch.Tensor):
    """
    
    :param centered_activations_data: this should already have had an appropriate "average activations" vector subtracted from it
    :param truth_labels:
    :param polarity_labels:
    :param np_rng:
    :return: tuple of truth vector and polarity vector
    """
    assert 2 == centered_activations_data.ndim == truth_labels.ndim == polarity_labels.ndim
    assert centered_activations_data.shape[0] == truth_labels.shape[0] == polarity_labels.shape[0]
    vector_size = centered_activations_data.shape[1]
    if vector_size % hidden_state_size != 0:
        logger.warning(f"NOTE- not using phi 3.5 mini because vector size {vector_size} is wrong")
    assert 1 == truth_labels.shape[1] == polarity_labels.shape[1]
    assert is_bipolar(truth_labels), "Not all truth labels are 1 or -1"
    assert is_bipolar(polarity_labels), "Not all polarity labels are 1 or -1"
    
    #this has to have shape (n,) rather than (n,1) because of scipy
    init_truth_and_polarity_vects = np_rng.normal(size=(2*vector_size,))
    
    def loss_fun(truth_and_polarity_vect_values: NDArray)-> float:
        truth_dir = truth_and_polarity_vect_values[0:vector_size, np.newaxis]
        polarity_dir = truth_and_polarity_vect_values[vector_size:2*vector_size, np.newaxis]
        guess_at_centered_data = truth_labels @ truth_dir.T + (truth_labels * polarity_labels) @ polarity_dir.T
        loss = np.sum(np.square(np.linalg.norm(centered_activations_data - guess_at_centered_data, axis=1)))
        return loss
    
    start_of_ols_ts = time.time()
    logger.debug("starting OLS for truth/polarity directions")
    with StdoutToLoggerRedirection(logger):
        ols_result = least_squares(loss_fun, init_truth_and_polarity_vects, verbose=2)
    num_secs_running_ols = time.time() - start_of_ols_ts
    logger.debug(f"OLS for truth/polarity directions finished after {num_secs_running_ols // 60} min, {num_secs_running_ols % 60:.3f} sec")
    
    if not ols_result['success']:
        logger.error(f"problem while solving for truth and polarity directions: {ols_result['message']}")
        raise RuntimeError(f"Scipy OLS didn't converge; {ols_result['status']}: {ols_result['message']}")
    final_truth_and_polarity_vects: NDArray = ols_result['x']
    final_truth_and_polarity_vects = final_truth_and_polarity_vects.astype(np.float32)
    return (torch.from_numpy(final_truth_and_polarity_vects[0:vector_size, np.newaxis]),
            torch.from_numpy(final_truth_and_polarity_vects[vector_size:2*vector_size, np.newaxis]))


def learn_directions_and_train_probe(
        train_activations: torch.Tensor, train_truth_labels: torch.Tensor, train_polarity_labels: torch.Tensor,
        val_activations: torch.Tensor, val_truth_labels: torch.Tensor, np_rng: np.random.Generator
) -> PolarityAwareTruthProbe:
    """
    
    
    
    Credit to https://machinelearningmastery.com/building-a-binary-classification-truth_probe-in-pytorch/
    :param train_activations:
    :param train_truth_labels:
    :param train_polarity_labels:
    :param val_activations:
    :param val_truth_labels:
    :param np_rng:
    :return:
    """
    assert (2 == train_activations.ndim == val_activations.ndim == train_truth_labels.ndim == train_polarity_labels.ndim
            == val_truth_labels.ndim), f"all ndims should be 2; actually: train_activations={train_activations.ndim}, val_activations={val_activations.ndim}, train_truth_labels={train_truth_labels.ndim}, train_polarity_labels={train_polarity_labels.ndim}, val_truth_labels={val_truth_labels.ndim}"
    assert 1 == train_truth_labels.shape[1] == train_polarity_labels.shape[1] == val_truth_labels.shape[1]
    num_train = train_activations.shape[0]
    assert num_train == train_truth_labels.shape[0] == train_polarity_labels.shape[0]
    assert is_binary(train_truth_labels), "Not all train-set truth labels are 1 or 0"
    assert is_bipolar(train_polarity_labels), "Not all polarity labels are 1 or -1"
    assert is_binary(val_truth_labels), "Not all validation-set truth labels are 1 or 0"
    activ_vect_size = train_activations.shape[1]
    assert activ_vect_size == val_activations.shape[1]
    if activ_vect_size % hidden_state_size != 0:
        logger.warning(f"NOTE- not using phi 3.5 mini because activation vector size {activ_vect_size} is wrong")
    num_val = val_activations.shape[0]
    assert num_val == val_truth_labels.shape[0]
    
    mean_train_activ = train_activations.mean(dim=0, keepdim=True)#row vector
    centered_train_activs = train_activations - mean_train_activ
    train_bipolar_truth_labels = train_truth_labels.clone()
    train_bipolar_truth_labels[train_bipolar_truth_labels == 0] = -1
    truth_dir, polarity_dir = solve_for_truth_polarity_vectors(centered_train_activs, train_bipolar_truth_labels,
                                                               train_polarity_labels, np_rng)
    
    truth_probe = PolarityAwareTruthProbe(mean_train_activ.T, truth_dir, polarity_dir)
    truth_probe.to(device)
    
    gpu_train_activs = train_activations.to(device)
    gpu_train_labels = train_truth_labels.to(device)
    gpu_val_activs = val_activations.to(device)
    gpu_val_labels = val_truth_labels.to(device)
    
    loss_fn = nn.BCELoss()
    optimizer = optim.AdamW(truth_probe.parameters(), lr=0.000125, weight_decay=0.01)
    
    n_epochs = 65_536   # number of epochs to run
    batch_size = 64  # size of each batch
    batch_start = torch.arange(0, num_train, batch_size).to(device)
    
    best_loss = np.inf
    best_epoch = -1
    best_weights = None
    best_bias = None
    
    prev_loss = np.inf
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
        
        logger.info(f"{'!!! ' if is_new_best else ('< ' if is_better else '')
        }Epoch {epoch}: Val Loss: {val_loss:.10f}, Val Acc: {val_acc:.8f}")
    
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
