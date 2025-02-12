import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from jaxtyping import Float
from pandas.core.dtypes.common import is_integer_dtype
from beartype import beartype
from beartype.door import die_if_unbearable

from phi_3_5_constants import dsets_folder, finalized_activations_dir


@dataclass
class ProbeTrainScenario:
    result_folder_name: str
    scenario_name: str
    src_dset_idxs: list[int]

    def scenario_key(self) -> tuple[int, ...]:
        return tuple(sorted(self.src_dset_idxs))


# this will contain a list of the above data structure
train_scenarios_spec_path = Path(".") / "train_scenarios_spec.json"


@dataclass
class TrainValidationSplitsIdxs:
    train_idxs: list[int]
    validation_idxs: list[int]


train_validation_splits_idxs_path = dsets_folder / "train_validation_splits_idxs.json"


def save_train_validation_splits(train_valid_splits_spec: dict[int, list[TrainValidationSplitsIdxs]]):
    with train_validation_splits_idxs_path.open("w") as f:
        json.dump({
            str(dset_idx): list(map(asdict, dset_split_specs))
            for dset_idx, dset_split_specs in train_valid_splits_spec.items()
        }, f)  # todo add indent=2 when rerunning this with 20 splits, but revert that if resulting file is too huge


def load_train_validation_splits() -> dict[int, list[TrainValidationSplitsIdxs]]:
    with open(train_validation_splits_idxs_path, "r") as f:
        raw_split_specs = json.load(f)
    return {int(dset_idx_str): [TrainValidationSplitsIdxs(**split_spec) for split_spec in split_specs]
            for dset_idx_str, split_specs in raw_split_specs.items()}


@dataclass
class DataComponents:
    activations: Float[torch.Tensor, "_dset_sz act_sz"]
    truth_labels: Float[torch.Tensor, "_dset_sz 1"]
    polarity_labels: Float[torch.Tensor, "_dset_sz 1"]

    def __post_init__(self):
        die_if_unbearable(self.activations, Float[torch.Tensor, "dset_sz act_sz"])
        die_if_unbearable(self.truth_labels, Float[torch.Tensor, "dset_sz 1"])
        die_if_unbearable(self.polarity_labels, Float[torch.Tensor, "dset_sz 1"])


class ActivationsDataSelector:

    def __init__(self, dsets_index_df: pd.DataFrame,
                 train_valid_splits_spec: dict[int, list[TrainValidationSplitsIdxs]]):
        assert dsets_index_df.shape[0] > 0, "must be provided info about nonzero number of datasets"
        dsets_index = dsets_index_df.index
        assert is_integer_dtype(dsets_index), f"invalid datasets df index dtype {dsets_index.dtype}"
        assert dsets_index.is_unique, "dset index df index is not unique"
        required_cols = {"Categ_Folder", "Dataset_File", "is_negated", "is_conj", "is_disj", "is_other", "in_german"}
        actual_cols = set(dsets_index_df.columns)
        assert required_cols.issubset(actual_cols), f"dset index df is missing columns: {required_cols - actual_cols}"
        self.dsets_index_df = dsets_index_df

        assert set(train_valid_splits_spec.keys()) == set(dsets_index), f"{train_valid_splits_spec.keys()} vs {dsets_index}"
        self.train_valid_splits_spec = train_valid_splits_spec
        self.num_split_variants = len(self.train_valid_splits_spec[dsets_index[0]])
        assert all([len(split_variants) == self.num_split_variants for split_variants in
                    self.train_valid_splits_spec.values()])

        # top-level key is the name of a topic that has 6 variants, second level key is one of those variant names
        self.dset_idxs_for_6way_topics: dict[str, dict[Literal['affirm', 'neg', 'conj', 'disj', 'de', 'de_neg'], int]] \
            = {}
        self.idxs_for_other_dsets: dict[str, int] = {}

        self.act_sz = -1
        # specifically the single best layer's activations (layer 18 in phi 3.5 mini's case)
        self.all_dsets_activations: dict[int, Float[torch.Tensor, "_dset_szs act_sz"]] = {}
        self.all_dsets_truth_labels: dict[int, Float[torch.Tensor, "_dset_szs 1"]] = {}
        self.all_dsets_polarity_labels: dict[int, Float[torch.Tensor, "_dset_szs 1"]] = {}
        
        self._load_data()

    def _load_data(self):
        """
        Initializes the data selector by loading the activations data from the files specified by the datasets index
        dataframe and otherwise processing that index for later convenience
        Loads all data immediately for simplicity and speed, given that this is only being used for modest-sized
        datasets
        """
        for dset_idx, dset_dtls in self.dsets_index_df.iterrows():
            assert isinstance(dset_idx, int)
            categ_nm = dset_dtls["Categ_Folder"]
            dset_file_nm = dset_dtls["Dataset_File"]
            dset_nm = os.path.splitext(dset_file_nm)[0]

            is_neg = dset_dtls['is_negated']

            if dset_dtls['is_other']:
                self.idxs_for_other_dsets[dset_nm] = dset_idx
            else:
                if categ_nm not in self.dset_idxs_for_6way_topics.keys():
                    self.dset_idxs_for_6way_topics[categ_nm] = {}

                in_german = dset_dtls['in_german']
                if is_neg:
                    self.dset_idxs_for_6way_topics[categ_nm]["de_neg" if in_german else "neg"] = dset_idx
                elif in_german:
                    self.dset_idxs_for_6way_topics[categ_nm]["de"] = dset_idx
                elif dset_dtls['is_conj']:
                    self.dset_idxs_for_6way_topics[categ_nm]["conj"] = dset_idx
                elif dset_dtls['is_disj']:
                    self.dset_idxs_for_6way_topics[categ_nm]["disj"] = dset_idx
                else:
                    self.dset_idxs_for_6way_topics[categ_nm]["affirm"] = dset_idx

            dataset = pd.read_csv(dsets_folder / categ_nm / dset_file_nm)
            dset_size = dataset.shape[0]
            dset_truth_labels = torch.from_numpy(dataset['label'].to_numpy().astype(np.float32)[:, np.newaxis])
            self.all_dsets_truth_labels[dset_idx] = dset_truth_labels

            polarity_labels = torch.full(dset_truth_labels.shape, -1.0 if is_neg else 1.0)
            self.all_dsets_polarity_labels[dset_idx] = polarity_labels

            activs_path = finalized_activations_dir / categ_nm / (dset_nm + ".pt")
            relevant_activations = torch.load(activs_path, weights_only=True)
            assert relevant_activations.shape[:2] == (2, dset_size)
            curr_act_sz: int = relevant_activations.shape[2]
            if self.act_sz == -1:
                self.act_sz = curr_act_sz
            else:
                assert self.act_sz == curr_act_sz, f"on {dset_idx}th dataset: {self.act_sz} vs {curr_act_sz}"
            lyr18_activations = relevant_activations[0, :, :]
            self.all_dsets_activations[dset_idx] = lyr18_activations
        assert all([len(idxs_of_6way_topic) == 6 for idxs_of_6way_topic in self.dset_idxs_for_6way_topics.values()])

    @beartype
    def select_train_validation_for_scenario(self, scenario_dset_idxs: list[int], split_variant_idx: int
                                             ) -> tuple[DataComponents, DataComponents]:
        assert all([dset_idx in self.dsets_index_df.index for dset_idx in scenario_dset_idxs])
        assert split_variant_idx < self.num_split_variants

        scenario_train_acts_lst: list[Float[torch.Tensor, "_t_dset_szs act_sz"]] = []
        scenario_validation_acts_lst: list[Float[torch.Tensor, "_v_dset_szs act_sz"]] = []

        scenario_train_truth_labels_lst: list[Float[torch.Tensor, "_t_dset_szs 1"]] = []
        scenario_validation_truth_labels_lst: list[Float[torch.Tensor, "_v_dset_szs 1"]] = []

        scenario_train_polarity_labels_lst: list[Float[torch.Tensor, "_t_dset_szs 1"]] = []
        scenario_validation_polarity_labels_lst: list[Float[torch.Tensor, "_v_dset_szs 1"]] = []

        for dset_idx in scenario_dset_idxs:
            curr_split = self.train_valid_splits_spec[dset_idx][split_variant_idx]

            dset_acts = self.all_dsets_activations[dset_idx]
            scenario_train_acts_lst.append(dset_acts[curr_split.train_idxs, :])
            scenario_validation_acts_lst.append(dset_acts[curr_split.validation_idxs, :])
            dset_truth_labels = self.all_dsets_truth_labels[dset_idx]
            scenario_train_truth_labels_lst.append(dset_truth_labels[curr_split.train_idxs, :])
            scenario_validation_truth_labels_lst.append(dset_truth_labels[curr_split.validation_idxs, :])
            dset_polarity_labels = self.all_dsets_polarity_labels[dset_idx]
            scenario_train_polarity_labels_lst.append(dset_polarity_labels[curr_split.train_idxs, :])
            scenario_validation_polarity_labels_lst.append(dset_polarity_labels[curr_split.validation_idxs, :])

        scenario_train_acts: Float[torch.Tensor, "combined_t_dset_sz act_sz"] = torch.cat(scenario_train_acts_lst)
        scenario_validation_acts: Float[torch.Tensor, "combined_v_dset_sz act_sz"] = torch.cat(
            scenario_validation_acts_lst)
        scenario_train_truth_labels: Float[torch.Tensor, "combined_t_dset_sz 1"] = torch.cat(
            scenario_train_truth_labels_lst)
        scenario_validation_truth_labels: Float[torch.Tensor, "combined_v_dset_sz 1"] = torch.cat(
            scenario_validation_truth_labels_lst)
        scenario_train_polarity_labels: Float[torch.Tensor, "combined_t_dset_sz 1"] = torch.cat(
            scenario_train_polarity_labels_lst)
        scenario_validation_polarity_labels: Float[torch.Tensor, "combined_v_dset_sz 1"] = torch.cat(
            scenario_validation_polarity_labels_lst)

        return (DataComponents(scenario_train_acts, scenario_train_truth_labels, scenario_train_polarity_labels),
                DataComponents(scenario_validation_acts, scenario_validation_truth_labels,
                               scenario_validation_polarity_labels))

    def grab_all_data_for_dset(self, dset_idx: int) -> DataComponents:
        assert dset_idx in self.dsets_index_df.index, f"{dset_idx} bad; options: {self.dsets_index_df.index.tolist()}"
        return DataComponents(self.all_dsets_activations[dset_idx], self.all_dsets_truth_labels[dset_idx],
                              self.all_dsets_polarity_labels[dset_idx])
