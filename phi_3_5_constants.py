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