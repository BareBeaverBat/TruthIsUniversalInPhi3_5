import dataclasses
from dataclasses import dataclass

import numpy as np
import torch as t
from jaxtyping import Float
from sklearn.linear_model import LogisticRegression
from typeguard import typechecked, check_type

from logging_setup import create_logger
from phi_3_5_constants import hidden_state_size
from utils import is_binary, is_bipolar

logger = create_logger(__name__)


def zero_t() -> Float[t.Tensor, ""]:
    return t.tensor(0.0)


def neg1_t() -> Float[t.Tensor, ""]:
    return t.tensor(-1.0)


@dataclass
class DirVectors:
    mean_activ: Float[t.Tensor, "vect_sz 1"]
    truth_dir: Float[t.Tensor, "vect_sz 1"]
    polarity_dir: Float[t.Tensor, "vect_sz 1"]
    
    def __post_init__(self):
        check_type(self.mean_activ, Float[t.Tensor, "vect_sz 1"])
        check_type(self.truth_dir, Float[t.Tensor, "vect_sz 1"])
        truth_dir_norm = t.linalg.vector_norm(self.truth_dir).item()
        if abs(truth_dir_norm - 1) > 1e-8:
            raise ValueError(f"truth direction should have unit norm, instead: {truth_dir_norm}")
        check_type(self.polarity_dir, Float[t.Tensor, "vect_sz 1"])
        polarity_dir_norm = t.linalg.vector_norm(self.polarity_dir).item()
        if abs(polarity_dir_norm-1) > 1e-8:
            raise ValueError(f"polarity direction should have unit norm, instead: {truth_dir_norm}")


@typechecked
def learn_directions_for_dset(
        train_activs: Float[t.Tensor, "n_t_records vect_sz"],
        train_truth_labels: Float[t.Tensor, "n_t_records 1"],
        train_polarity_labels: Float[t.Tensor, "n_t_records 1"]) -> DirVectors:
    assert is_bipolar(train_polarity_labels)
    assert is_binary(train_truth_labels)
    if train_activs.shape[1] != hidden_state_size:
        logger.warning(f"activations aren't from phi 3.5 mini because activation size {train_activs.shape[1]} is wrong")

    train_bipolar_truth_labels = t.where(train_truth_labels == zero_t(), neg1_t(), train_truth_labels)

    logger.info(f"doing direction-learning for {train_activs.shape[0]} records of data")

    mean_train_activ: Float[t.Tensor, "vect_sz 1"] = train_activs.mean(dim=0, keepdim=True).T
    centered_activations_data = train_activs - mean_train_activ.T
    Y: Float[t.Tensor, "n_records 2"] = t.cat(
        (train_bipolar_truth_labels, train_bipolar_truth_labels * train_polarity_labels), dim=1)
    jointly_learned_truth_polarity_dirs: Float[t.Tensor, "2 vect_sz"] = (
            t.linalg.inv(Y.T @ Y) @ Y.T @ centered_activations_data)
    truth_dir: Float[t.Tensor, "vect_sz 1"] = jointly_learned_truth_polarity_dirs[0, :, None]
    truth_dir = truth_dir / t.linalg.vector_norm(truth_dir)
    # following Bürger et al. in discarding jointly learned polarity direction
    
    binary_polarity_labels: Float[t.Tensor, "n_records 1"] = t.where(
        train_polarity_labels == neg1_t(), zero_t(), train_polarity_labels)
    # following Bürger et al. in the choice of not doing regularization when training a polarity direction
    polarity_lin_classif = LogisticRegression(penalty=None, fit_intercept=True)
    polarity_lin_classif.fit(train_activs.numpy(), binary_polarity_labels.numpy())
    polarity_dir: Float[t.Tensor, "vect_sz 1"] = t.from_numpy(polarity_lin_classif.coef_).T
    polarity_dir = polarity_dir / t.linalg.vector_norm(polarity_dir)

    return DirVectors(mean_train_activ, truth_dir, polarity_dir)


@dataclass
class ReconLosses:
    train_mean_activ_loss_on_train: float
    train_mean_activ_and_t_p_dirs_loss_on_train: float
    train_mean_activ_loss_on_validation: float
    train_mean_activ_and_t_p_dirs_loss_on_validation: float
    validation_mean_activ_loss_on_validation: float
    validation_mean_activ_and_t_p_dirs_loss_on_validation: float


@typechecked
def record_count_normalized_recon_loss(
        activations_data: Float[t.Tensor, "n_records vect_sz"], truth_labels: Float[t.Tensor, "n_records 1"],
        polarity_labels: Float[t.Tensor, "n_records 1"], estimated_vects: DirVectors) -> (
        float, float):
    assert is_bipolar(truth_labels), "Not all truth labels are 1 or -1"
    assert is_bipolar(polarity_labels), "Not all polarity labels are 1 or -1"

    data_reconstr = (estimated_vects.mean_activ.T + truth_labels @ estimated_vects.truth_dir.T
                     + (truth_labels * polarity_labels) @ estimated_vects.polarity_dir.T)

    loss_per_record = np.mean(np.square(np.linalg.norm(activations_data - data_reconstr, axis=1)))
    loss_per_record_with_just_mean_activ = np.mean(
        np.square(np.linalg.norm(activations_data - estimated_vects.mean_activ.T, axis=1)))
    # loss_per_record_normalized_for_dset = loss_per_record / loss_per_record_with_just_mean_activ

    return loss_per_record_with_just_mean_activ, loss_per_record


@typechecked
def compute_recon_losses(
        dir_vects_from_train: DirVectors, train_activs: Float[t.Tensor, "n_t_records vect_sz"],
        train_truth_labels: Float[t.Tensor, "n_t_records 1"],
        train_polarity_labels: Float[t.Tensor, "n_t_records 1"],
        validation_activs: Float[t.Tensor, "n_v_records vect_sz"],
        validation_truth_labels: Float[t.Tensor, "n_v_records 1"],
        validation_polarity_labels: Float[t.Tensor, "n_v_records 1"]) -> ReconLosses:
    assert is_binary(train_truth_labels), "Not all train-set truth labels are 1 or 0"
    assert is_bipolar(train_polarity_labels), "Not all train-set polarity labels are 1 or -1"
    assert is_binary(validation_truth_labels), "Not all validation-set truth labels are 1 or 0"
    assert is_bipolar(validation_polarity_labels), "Not all validation-set polarity labels are 1 or -1"
    if train_activs.shape[1] % hidden_state_size != 0:
        logger.warning(f"NOTE- not using phi 3.5 mini because activation size {train_activs.shape[1]} is wrong")

    train_bipolar_truth_labels = t.where(train_truth_labels == zero_t(), neg1_t(), train_truth_labels)
    validation_bipolar_truth_labels = t.where(validation_truth_labels == zero_t(), neg1_t(), validation_truth_labels)

    train_mean_activ_loss_on_train, train_mean_activ_and_t_p_dirs_loss_on_train = \
        record_count_normalized_recon_loss(
            train_activs, train_bipolar_truth_labels, train_polarity_labels, dir_vects_from_train)
    train_mean_activ_loss_on_validation, train_mean_activ_and_t_p_dirs_loss_on_validation = \
        record_count_normalized_recon_loss(
            validation_activs, validation_bipolar_truth_labels, validation_polarity_labels, dir_vects_from_train)

    mean_validation_activ: Float[t.Tensor, "vect_sz 1"] = validation_activs.mean(dim=0, keepdim=True).T
    validation_mean_activ_loss_on_validation, validation_mean_activ_and_t_p_dirs_loss_on_validation = \
        record_count_normalized_recon_loss(
            validation_activs, validation_bipolar_truth_labels, validation_polarity_labels,
            dataclasses.replace(dir_vects_from_train, mean_activ=mean_validation_activ))

    return ReconLosses(
        train_mean_activ_loss_on_train=train_mean_activ_loss_on_train,
        train_mean_activ_and_t_p_dirs_loss_on_train=train_mean_activ_and_t_p_dirs_loss_on_train,
        train_mean_activ_loss_on_validation=train_mean_activ_loss_on_validation,
        train_mean_activ_and_t_p_dirs_loss_on_validation=train_mean_activ_and_t_p_dirs_loss_on_validation,
        validation_mean_activ_loss_on_validation=validation_mean_activ_loss_on_validation,
        validation_mean_activ_and_t_p_dirs_loss_on_validation=validation_mean_activ_and_t_p_dirs_loss_on_validation)
