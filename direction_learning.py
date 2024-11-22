import time
from dataclasses import dataclass, asdict
from pathlib import Path
import winsound


import numpy as np
import torch
from numpy.typing import NDArray
from scipy.optimize import least_squares

from logging_setup import StdoutToLoggerRedirection, create_logger
from phi_3_5_constants import hidden_state_size
from utils import is_binary, is_bipolar

logger = create_logger(__name__)


@dataclass
class DirVectors:
    lyr18_mean_activ: torch.Tensor
    lyr18_truth_dir: torch.Tensor
    lyr18_polarity_dir: torch.Tensor
    lyr25_mean_activ: torch.Tensor
    lyr25_truth_dir: torch.Tensor
    lyr25_polarity_dir: torch.Tensor
    lyrs18_and_25_mean_activ: torch.Tensor
    lyrs18_and_25_truth_dir: torch.Tensor
    lyrs18_and_25_polarity_dir: torch.Tensor
    
    def __post_init__(self):
        if not (self.lyr18_mean_activ.ndim == self.lyr18_truth_dir.ndim == self.lyr18_polarity_dir.ndim
                == self.lyr25_mean_activ.ndim == self.lyr25_truth_dir.ndim == self.lyr25_polarity_dir.ndim
                == self.lyrs18_and_25_mean_activ.ndim == self.lyrs18_and_25_truth_dir.ndim
                == self.lyrs18_and_25_polarity_dir.ndim):
            raise ValueError(f"all entries should be column-vector-type matrices with 2 entries in the pytorch tensor's shape, but instead: self.lyr18_mean_activ.ndim={self.lyr18_mean_activ.ndim}; self.lyr18_truth_dir.ndim={self.lyr18_truth_dir.ndim} ; self.lyr18_polarity_dir.ndim={self.lyr18_polarity_dir.ndim}; self.lyr25_mean_activ.ndim={self.lyr25_mean_activ.ndim}; self.lyr25_truth_dir.ndim={self.lyr25_truth_dir.ndim}; self.lyr25_polarity_dir.ndim={self.lyr25_polarity_dir.ndim}; self.lyrs18_and_25_mean_activ.ndim={self.lyrs18_and_25_mean_activ.ndim}; self.lyrs18_and_25_truth_dir.ndim={self.lyrs18_and_25_truth_dir.ndim}; self.lyrs18_and_25_polarity_dir.ndim={self.lyrs18_and_25_polarity_dir.ndim}")
        if not (1 == self.lyr18_mean_activ.shape[1] == self.lyr18_truth_dir.shape[1] == self.lyr18_polarity_dir.shape[1]
                == self.lyr25_mean_activ.shape[1] == self.lyr25_truth_dir.shape[1] == self.lyr25_polarity_dir.shape[1]
                == self.lyrs18_and_25_mean_activ.shape[1] == self.lyrs18_and_25_truth_dir.shape[1]
                == self.lyrs18_and_25_polarity_dir.shape[1]):
            raise ValueError(f"all entries should be column-vector-type matrices, but instead their second dimension's size is: self.lyr18_mean_activ.shape[1]={self.lyr18_mean_activ.shape[1]}; self.lyr18_truth_dir.shape[1]={self.lyr18_truth_dir.shape[1]} ; self.lyr18_polarity_dir.shape[1]={self.lyr18_polarity_dir.shape[1]}; self.lyr25_mean_activ.shape[1]={self.lyr25_mean_activ.shape[1]}; self.lyr25_truth_dir.shape[1]={self.lyr25_truth_dir.shape[1]}; self.lyr25_polarity_dir.shape[1]={self.lyr25_polarity_dir.shape[1]}; self.lyrs18_and_25_mean_activ.shape[1]={self.lyrs18_and_25_mean_activ.shape[1]}; self.lyrs18_and_25_truth_dir.shape[1]={self.lyrs18_and_25_truth_dir.shape[1]}; self.lyrs18_and_25_polarity_dir.shape[1]={self.lyrs18_and_25_polarity_dir.shape[1]}")
        if not (self.lyr18_mean_activ.shape[0] == self.lyr18_truth_dir.shape[0] == self.lyr18_polarity_dir.shape[0]
                == self.lyr25_mean_activ.shape[0] == self.lyr25_truth_dir.shape[0] == self.lyr25_polarity_dir.shape[0]):
            raise ValueError(f"all vectors for single-layer scenarios should have same length, but instead their first dimension's size is: self.lyr18_mean_activ.shape[0]={self.lyr18_mean_activ.shape[0]}; self.lyr18_truth_dir.shape[0]={self.lyr18_truth_dir.shape[0]} ; self.lyr18_polarity_dir.shape[0]={self.lyr18_polarity_dir.shape[0]}; self.lyr25_mean_activ.shape[0]={self.lyr25_mean_activ.shape[0]}; self.lyr25_truth_dir.shape[0]={self.lyr25_truth_dir.shape[0]}; self.lyr25_polarity_dir.shape[0]={self.lyr25_polarity_dir.shape[0]}")
        if not (self.lyrs18_and_25_mean_activ.shape[0] == self.lyrs18_and_25_truth_dir.shape[0]
                == self.lyrs18_and_25_polarity_dir.shape[0]):
            raise ValueError(f"all vectors for double-layer scenarios should have same length, but instead their first dimension's size is: self.lyrs18_and_25_mean_activ.shape[0]={self.lyrs18_and_25_mean_activ.shape[0]}; self.lyrs18_and_25_truth_dir.shape[0]={self.lyrs18_and_25_truth_dir.shape[0]}; self.lyrs18_and_25_polarity_dir.shape[0]={self.lyrs18_and_25_polarity_dir.shape[0]}")


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
    
    def loss_fun(truth_and_polarity_vect_values: NDArray) -> float:
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
    winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
    
    if not ols_result['success']:
        logger.error(f"problem while solving for truth and polarity directions: {ols_result['message']}")
        raise RuntimeError(f"Scipy OLS didn't converge; {ols_result['status']}: {ols_result['message']}")
    final_truth_and_polarity_vects: NDArray = ols_result['x']
    final_truth_and_polarity_vects = final_truth_and_polarity_vects.astype(np.float32)
    return (torch.from_numpy(final_truth_and_polarity_vects[0:vector_size, np.newaxis]),
            torch.from_numpy(final_truth_and_polarity_vects[vector_size:2*vector_size, np.newaxis]))


def learn_directions_for_dset(
        output_folder: Path, output_nm_prefix: str, train_activs: torch.Tensor, train_truth_labels: torch.Tensor,
        train_polarity_labels, np_rng: np.random.Generator
) -> DirVectors:
    assert 3 == train_activs.ndim
    assert 2 == train_activs.shape[0]
    assert 2 == train_truth_labels.ndim == train_polarity_labels.ndim
    assert 1 == train_truth_labels.shape[1] == train_polarity_labels.shape[1]
    num_train_records = train_activs.shape[1]
    assert num_train_records == train_truth_labels.shape[0] == train_polarity_labels.shape[0]
    assert is_bipolar(train_polarity_labels)
    assert is_binary(train_truth_labels)
    activs_size = train_activs.shape[2]
    if activs_size != hidden_state_size:
        logger.warning(f"dataset of activations isn't from phi 3.5 mini because activation size {activs_size} is wrong")
    
    output_folder.mkdir(exist_ok=True)

    save_location = output_folder / f"{output_nm_prefix}.pt"
    if save_location.exists():
        logger.info(f"skipping direction-learning for {num_train_records} records of data {output_nm_prefix} in the location {output_folder} because the file {save_location} already exists")
        tensors_dict = torch.load(save_location, weights_only=True)
        assert isinstance(tensors_dict, dict)
        return DirVectors(**tensors_dict)
    
    train_bipolar_truth_labels = train_truth_labels.clone()
    train_bipolar_truth_labels[train_bipolar_truth_labels == 0] = -1

    logger.info(f"doing direction-learning for layer 18 for {num_train_records} records of data {output_nm_prefix} in the location {output_folder}")
    lyr18_train_activs = train_activs[0, :, :]
    lyr18_mean_train_activ = lyr18_train_activs.mean(dim=0, keepdim=True).T
    assert lyr18_mean_train_activ.shape == (activs_size, 1)
    lyr18_centered_train_activs = lyr18_train_activs - lyr18_mean_train_activ.T
    assert lyr18_centered_train_activs.shape == (num_train_records, activs_size)

    lyr18_dirs_temp_save_location = output_folder / f"{output_nm_prefix}_lyr18_tmp.pt"
    if lyr18_dirs_temp_save_location.exists():
        logger.info("skipping layer 18 direction-learning because the file already exists")
        lyr18_saved_dirs = torch.load(lyr18_dirs_temp_save_location, weights_only=True)
        assert isinstance(lyr18_saved_dirs, dict)
        lyr18_truth_dir, lyr18_polarity_dir = lyr18_saved_dirs['lyr18_truth_dir'], lyr18_saved_dirs['lyr18_polarity_dir']
    else:
        lyr18_truth_dir, lyr18_polarity_dir = solve_for_truth_polarity_vectors(
            lyr18_centered_train_activs, train_bipolar_truth_labels, train_polarity_labels, np_rng)
        lyr18_dirs_storage = {'lyr18_truth_dir': lyr18_truth_dir, 'lyr18_polarity_dir': lyr18_polarity_dir}
        torch.save(lyr18_dirs_storage, lyr18_dirs_temp_save_location)
    
    logger.info(f"doing direction-learning for layer 25 for {num_train_records} records of data {output_nm_prefix} in the location {output_folder}")
    lyr25_train_activs = train_activs[1, :, :]
    lyr25_mean_train_activ = lyr25_train_activs.mean(dim=0, keepdim=True).T
    assert lyr25_mean_train_activ.shape == (activs_size, 1)
    lyr25_centered_train_activs = lyr25_train_activs - lyr25_mean_train_activ.T
    assert lyr25_centered_train_activs.shape == (num_train_records, activs_size)

    lyr25_dirs_temp_save_location = output_folder / f"{output_nm_prefix}_lyr25_tmp.pt"
    if lyr25_dirs_temp_save_location.exists():
        logger.info("skipping layer 25 direction-learning because the file already exists")
        lyr25_saved_dirs = torch.load(lyr25_dirs_temp_save_location, weights_only=True)
        assert isinstance(lyr25_saved_dirs, dict)
        lyr25_truth_dir, lyr25_polarity_dir = lyr25_saved_dirs['lyr25_truth_dir'], lyr25_saved_dirs['lyr25_polarity_dir']
    else:
        lyr25_truth_dir, lyr25_polarity_dir = solve_for_truth_polarity_vectors(
            lyr25_centered_train_activs, train_bipolar_truth_labels, train_polarity_labels, np_rng)
        lyr25_dirs_storage = {'lyr25_truth_dir': lyr25_truth_dir, 'lyr25_polarity_dir': lyr25_polarity_dir}
        torch.save(lyr25_dirs_storage, lyr25_dirs_temp_save_location)
    
    logger.info(f"doing direction-learning for layers 18 & 25 for {num_train_records} records of data {output_nm_prefix} in the location {output_folder}")
    lyrs18_and_25_mean_train_activ = torch.concat((lyr18_mean_train_activ, lyr25_mean_train_activ), dim=0)
    assert lyrs18_and_25_mean_train_activ.shape == (2*activs_size, 1)
    lyrs18_and_25_centered_train_activs = torch.concat(
        (lyr18_centered_train_activs, lyr25_centered_train_activs), dim=1)
    assert lyrs18_and_25_centered_train_activs.shape == (num_train_records, 2*activs_size)
    
    lyrs18_and_25_truth_dir, lyrs18_and_25_polarity_dir = solve_for_truth_polarity_vectors(
        lyrs18_and_25_centered_train_activs, train_bipolar_truth_labels, train_polarity_labels, np_rng)
    
    vectors = DirVectors(lyr18_mean_activ=lyr18_mean_train_activ, lyr18_truth_dir=lyr18_truth_dir,
                         lyr18_polarity_dir=lyr18_polarity_dir, lyr25_mean_activ=lyr25_mean_train_activ,
                         lyr25_truth_dir=lyr25_truth_dir, lyr25_polarity_dir=lyr25_polarity_dir,
                         lyrs18_and_25_mean_activ=lyrs18_and_25_mean_train_activ,
                         lyrs18_and_25_truth_dir=lyrs18_and_25_truth_dir,
                         lyrs18_and_25_polarity_dir=lyrs18_and_25_polarity_dir)

    torch.save(asdict(vectors), save_location)

    if lyr18_dirs_temp_save_location.exists():
        lyr18_dirs_temp_save_location.unlink()
    if lyr25_dirs_temp_save_location.exists():
        lyr25_dirs_temp_save_location.unlink()

    return vectors


@dataclass
class ReconLosses:
    lyr18_train_mean_activ_loss_on_train: float
    lyr18_train_mean_activ_and_t_p_dirs_loss_on_train: float
    lyr18_train_mean_activ_loss_on_validation: float
    lyr18_train_mean_activ_and_t_p_dirs_loss_on_validation: float
    lyr18_validation_mean_activ_loss_on_validation: float
    lyr18_validation_mean_activ_and_t_p_dirs_loss_on_validation: float
    lyr25_train_mean_activ_loss_on_train: float
    lyr25_train_mean_activ_and_t_p_dirs_loss_on_train: float
    lyr25_train_mean_activ_loss_on_validation: float
    lyr25_train_mean_activ_and_t_p_dirs_loss_on_validation: float
    lyr25_validation_mean_activ_loss_on_validation: float
    lyr25_validation_mean_activ_and_t_p_dirs_loss_on_validation: float
    lyrs18_and_25_train_mean_activ_loss_on_train: float
    lyrs18_and_25_train_mean_activ_and_t_p_dirs_loss_on_train: float
    lyrs18_and_25_train_mean_activ_loss_on_validation: float
    lyrs18_and_25_train_mean_activ_and_t_p_dirs_loss_on_validation: float
    lyrs18_and_25_validation_mean_activ_loss_on_validation: float
    lyrs18_and_25_validation_mean_activ_and_t_p_dirs_loss_on_validation: float


def record_count_normalized_recon_loss(
        activations_data: torch.Tensor, truth_labels: torch.Tensor, polarity_labels: torch.Tensor,
        mean_activation_estim: torch.Tensor, truth_dir_estim: torch.Tensor, polarity_dir_estim: torch.Tensor) -> (
        float, float):
    assert (2 == activations_data.ndim == truth_labels.ndim == polarity_labels.ndim == mean_activation_estim.ndim
            == truth_dir_estim.ndim == polarity_dir_estim.ndim)
    assert activations_data.shape[0] == truth_labels.shape[0] == polarity_labels.shape[0]
    vector_size = activations_data.shape[1]
    if vector_size % hidden_state_size != 0:
        logger.warning(f"NOTE- not using phi 3.5 mini because vector size {vector_size} is wrong")
    assert vector_size == mean_activation_estim.shape[0] == truth_dir_estim.shape[0] == polarity_dir_estim.shape[0]
    assert 1 == truth_labels.shape[1] == polarity_labels.shape[1] == mean_activation_estim.shape[1] == \
           truth_dir_estim.shape[1] == polarity_dir_estim.shape[1]
    assert is_bipolar(truth_labels), "Not all truth labels are 1 or -1"
    assert is_bipolar(polarity_labels), "Not all polarity labels are 1 or -1"

    data_reconstr = (mean_activation_estim.T + truth_labels @ truth_dir_estim.T
                     + (truth_labels * polarity_labels) @ polarity_dir_estim.T)

    loss_per_record = np.mean(np.square(np.linalg.norm(activations_data - data_reconstr, axis=1)))
    loss_per_record_with_just_mean_activ = np.mean(
        np.square(np.linalg.norm(activations_data - mean_activation_estim.T, axis=1)))
    # loss_per_record_normalized_for_dset = loss_per_record / loss_per_record_with_just_mean_activ

    return loss_per_record_with_just_mean_activ, loss_per_record


def compute_recon_losses(
        dir_vects_from_train: DirVectors, train_activs: torch.Tensor, train_truth_labels: torch.Tensor,
        train_polarity_labels: torch.Tensor, validation_activs: torch.Tensor, validation_truth_labels: torch.Tensor,
        validation_polarity_labels: torch.Tensor) -> ReconLosses:
    assert (3 == train_activs.ndim == validation_activs.ndim)
    assert (2 == train_activs.shape[0] == validation_activs.shape[0])
    assert (2 == train_truth_labels.ndim == validation_truth_labels.ndim == train_polarity_labels.ndim
            == validation_polarity_labels.ndim)
    assert (1 == train_truth_labels.shape[1] == validation_truth_labels.shape[1] == train_polarity_labels.shape[1]
            == validation_polarity_labels.shape[1])
    num_train = train_activs.shape[1]
    assert num_train == train_truth_labels.shape[0] == train_polarity_labels.shape[0]
    assert is_binary(train_truth_labels), "Not all train-set truth labels are 1 or 0"
    assert is_bipolar(train_polarity_labels), "Not all train-set polarity labels are 1 or -1"
    assert is_binary(validation_truth_labels), "Not all validation-set truth labels are 1 or 0"
    assert is_bipolar(validation_polarity_labels), "Not all validation-set polarity labels are 1 or -1"
    activ_vect_size = train_activs.shape[2]
    assert (activ_vect_size == validation_activs.shape[2] == dir_vects_from_train.lyr18_mean_activ.shape[0])
    if activ_vect_size % hidden_state_size != 0:
        logger.warning(f"NOTE- not using phi 3.5 mini because activation vector size {activ_vect_size} is wrong")
    num_val = validation_activs.shape[1]
    assert num_val == validation_truth_labels.shape[0] == validation_polarity_labels.shape[0]

    train_bipolar_truth_labels = train_truth_labels.clone()
    train_bipolar_truth_labels[train_bipolar_truth_labels == 0] = -1

    validation_bipolar_truth_labels = validation_truth_labels.clone()
    validation_bipolar_truth_labels[validation_bipolar_truth_labels == 0] = -1

    lyr18_train_activs = train_activs[0, :, :]
    lyr25_train_activs = train_activs[1, :, :]
    lyrs18_and_25_train_activs = torch.concat((lyr18_train_activs, lyr25_train_activs), dim=1)

    lyr18_validation_activs = validation_activs[0, :, :]
    lyr18_mean_validation_activ = lyr18_validation_activs.mean(dim=0, keepdim=True).T
    lyr25_validation_activs = validation_activs[1, :, :]
    lyr25_mean_validation_activ = lyr25_validation_activs.mean(dim=0, keepdim=True).T
    lyrs18_and_25_validation_activs = torch.concat((lyr18_validation_activs, lyr25_validation_activs), dim=1)
    lyrs18_and_25_mean_validation_activ = torch.concat(
        (lyr18_mean_validation_activ, lyr25_mean_validation_activ), dim=0)

    lyr18_train_mean_activ_loss_on_train, lyr18_train_mean_activ_and_t_p_dirs_loss_on_train = \
        record_count_normalized_recon_loss(
            lyr18_train_activs, train_bipolar_truth_labels, train_polarity_labels,
            dir_vects_from_train.lyr18_mean_activ, dir_vects_from_train.lyr18_truth_dir,
            dir_vects_from_train.lyr18_polarity_dir)
    lyr18_train_mean_activ_loss_on_validation, lyr18_train_mean_activ_and_t_p_dirs_loss_on_validation = \
        record_count_normalized_recon_loss(
            lyr18_validation_activs, validation_bipolar_truth_labels, validation_polarity_labels,
            dir_vects_from_train.lyr18_mean_activ, dir_vects_from_train.lyr18_truth_dir,
            dir_vects_from_train.lyr18_polarity_dir)
    lyr18_validation_mean_activ_loss_on_validation, lyr18_validation_mean_activ_and_t_p_dirs_loss_on_validation = \
        record_count_normalized_recon_loss(
            lyr18_validation_activs, validation_bipolar_truth_labels, validation_polarity_labels,
            lyr18_mean_validation_activ, dir_vects_from_train.lyr18_truth_dir,
            dir_vects_from_train.lyr18_polarity_dir)
    
    lyr25_train_mean_activ_loss_on_train, lyr25_train_mean_activ_and_t_p_dirs_loss_on_train = \
        record_count_normalized_recon_loss(
            lyr25_train_activs, train_bipolar_truth_labels, train_polarity_labels,
            dir_vects_from_train.lyr25_mean_activ, dir_vects_from_train.lyr25_truth_dir,
            dir_vects_from_train.lyr25_polarity_dir)
    lyr25_train_mean_activ_loss_on_validation, lyr25_train_mean_activ_and_t_p_dirs_loss_on_validation = \
        record_count_normalized_recon_loss(
            lyr25_validation_activs, validation_bipolar_truth_labels, validation_polarity_labels,
            dir_vects_from_train.lyr25_mean_activ, dir_vects_from_train.lyr25_truth_dir,
            dir_vects_from_train.lyr25_polarity_dir)
    lyr25_validation_mean_activ_loss_on_validation, lyr25_validation_mean_activ_and_t_p_dirs_loss_on_validation = \
        record_count_normalized_recon_loss(
            lyr25_validation_activs, validation_bipolar_truth_labels, validation_polarity_labels,
            lyr25_mean_validation_activ, dir_vects_from_train.lyr25_truth_dir,
            dir_vects_from_train.lyr25_polarity_dir)

    lyrs18_and_25_train_mean_activ_loss_on_train, lyrs18_and_25_train_mean_activ_and_t_p_dirs_loss_on_train = \
        record_count_normalized_recon_loss(
            lyrs18_and_25_train_activs, train_bipolar_truth_labels, train_polarity_labels,
            dir_vects_from_train.lyrs18_and_25_mean_activ, dir_vects_from_train.lyrs18_and_25_truth_dir,
            dir_vects_from_train.lyrs18_and_25_polarity_dir)
    (lyrs18_and_25_train_mean_activ_loss_on_validation,
     lyrs18_and_25_train_mean_activ_and_t_p_dirs_loss_on_validation) = \
        record_count_normalized_recon_loss(
            lyrs18_and_25_validation_activs, validation_bipolar_truth_labels, validation_polarity_labels,
            dir_vects_from_train.lyrs18_and_25_mean_activ, dir_vects_from_train.lyrs18_and_25_truth_dir,
            dir_vects_from_train.lyrs18_and_25_polarity_dir)
    (lyrs18_and_25_validation_mean_activ_loss_on_validation,
     lyrs18_and_25_validation_mean_activ_and_t_p_dirs_loss_on_validation) = \
        record_count_normalized_recon_loss(
            lyrs18_and_25_validation_activs, validation_bipolar_truth_labels, validation_polarity_labels,
            lyrs18_and_25_mean_validation_activ, dir_vects_from_train.lyrs18_and_25_truth_dir,
            dir_vects_from_train.lyrs18_and_25_polarity_dir)

    return ReconLosses(
        lyr18_train_mean_activ_loss_on_train=lyr18_train_mean_activ_loss_on_train,
        lyr18_train_mean_activ_and_t_p_dirs_loss_on_train=lyr18_train_mean_activ_and_t_p_dirs_loss_on_train,
        lyr18_train_mean_activ_loss_on_validation=lyr18_train_mean_activ_loss_on_validation,
        lyr18_train_mean_activ_and_t_p_dirs_loss_on_validation=lyr18_train_mean_activ_and_t_p_dirs_loss_on_validation,
        lyr18_validation_mean_activ_loss_on_validation=lyr18_validation_mean_activ_loss_on_validation,
        lyr18_validation_mean_activ_and_t_p_dirs_loss_on_validation=
        lyr18_validation_mean_activ_and_t_p_dirs_loss_on_validation,
        lyr25_train_mean_activ_loss_on_train=lyr25_train_mean_activ_loss_on_train,
        lyr25_train_mean_activ_and_t_p_dirs_loss_on_train=lyr25_train_mean_activ_and_t_p_dirs_loss_on_train,
        lyr25_train_mean_activ_loss_on_validation=lyr25_train_mean_activ_loss_on_validation,
        lyr25_train_mean_activ_and_t_p_dirs_loss_on_validation=lyr25_train_mean_activ_and_t_p_dirs_loss_on_validation,
        lyr25_validation_mean_activ_loss_on_validation=lyr25_validation_mean_activ_loss_on_validation,
        lyr25_validation_mean_activ_and_t_p_dirs_loss_on_validation=
        lyr25_validation_mean_activ_and_t_p_dirs_loss_on_validation,
        lyrs18_and_25_train_mean_activ_loss_on_train=lyrs18_and_25_train_mean_activ_loss_on_train,
        lyrs18_and_25_train_mean_activ_and_t_p_dirs_loss_on_train=
        lyrs18_and_25_train_mean_activ_and_t_p_dirs_loss_on_train,
        lyrs18_and_25_train_mean_activ_loss_on_validation=lyrs18_and_25_train_mean_activ_loss_on_validation,
        lyrs18_and_25_train_mean_activ_and_t_p_dirs_loss_on_validation=
        lyrs18_and_25_train_mean_activ_and_t_p_dirs_loss_on_validation,
        lyrs18_and_25_validation_mean_activ_loss_on_validation=lyrs18_and_25_validation_mean_activ_loss_on_validation,
        lyrs18_and_25_validation_mean_activ_and_t_p_dirs_loss_on_validation=
        lyrs18_and_25_validation_mean_activ_and_t_p_dirs_loss_on_validation)
