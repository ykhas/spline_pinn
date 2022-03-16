import torch
from torch.utils.data import Dataset

class WaveDataset(Dataset):
  def __init__(self, w, dataset_size=1000, full_res_multiplier=4):
    self.dataset_size = dataset_size
    self.w = w

    self.boolean_masks = torch.zeros(dataset_size, w)
    self.boundary_values = torch.zeros(dataset_size, w)
    self.hidden_states = torch.zeros(dataset_size, w-1)

  def __getitem__(self, idx):
    return self.boundary_values[idx], self.boolean_masks[idx], self.hidden_states[idx]

  def update_items(self, idx, boolean_mask, boundary_value, hidden_state):
    self.boolean_masks[idx] = boolean_mask 
    self.boundary_values[idx] = boundary_value 
    self.hidden_states[idx] = hidden_state

  def __len__(self):
    return self.dataset_size

