import math
from pathlib import Path

import torch

hidden_state_size = 3072
num_layers = 33

seed = 1894327

num_splits = 20


def calc_seeds_for_splits(split_count=num_splits, base_seed=seed) -> list[int]:
    return [int((base_seed+739*k)*(math.pi-2)**k) for k in range(split_count)]


dsets_folder = Path("./true_false_datasets")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

directions_results_folder = Path("./learned_vectors")
finalized_activations_dir = Path("D:\\TruthIsUniversal_In_Phi_3_5_Mini\\best2_layers_activations_for_final_token_of_sequences")
dsets_index_path = dsets_folder / "datasets_index.csv"

probes_folder = Path("./trained_probes")
baseline_probes_folder = Path("./baseline_linear_probes")
analysis_results_folder = Path("./analysis_results")

directions_reconstruction_losses_path = analysis_results_folder / "reconstruction_losses.csv"

train_split_classification_metrics_path = analysis_results_folder / "train_split_metrics.json"
validation_split_classification_metrics_path = analysis_results_folder / "validation_split_metrics.json"

test_classification_metrics_path = analysis_results_folder / "test_metrics.json"

separation_by_layer_analysis_path = analysis_results_folder / "separation_by_layer.csv"

# vector norms are surprisingly noisy; checking e.g. if a vector is unit norm or zero norm requires wide tolerance
vect_norm_tol = 1e-4
