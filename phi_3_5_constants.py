from pathlib import Path

import torch

hidden_state_size = 3072
num_layers = 33

seed = 1894327

dsets_folder = Path("true_false_datasets")

token_lengths_path = dsets_folder / "dset_record_token_lengths.json"
train_split_records_path = dsets_folder / "train_split_record_indices.json"
validation_split_records_path = dsets_folder / "validation_split_record_indices.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

directions_results_folder = Path("learned_vectors")
finalized_activations_dir = Path("D:\\TruthIsUniversal_In_Phi_3_5_Mini\\best2_layers_activations_for_final_token_of_sequences")
dsets_index_path = dsets_folder / "datasets_index.csv"

misc_datasets_index_path = dsets_folder / "misc_dsets_index.json"
four_way_topics_index_path = dsets_folder / "four_way_dsets_index.json"

