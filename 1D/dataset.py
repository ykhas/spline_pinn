import torch
from torch.utils.data import Dataset

class WaveDataset(Dataset):
  def __init__(self, w, dataset_size=1000, full_res_multiplier=4):
    self.dataset_size = dataset_size
    self.w = w

    self.boundary = torch.zeros(dataset_size, w)
    self.domain = torch.zeros(dataset_size, w)
    self.hidden_states = torch.zeros(dataset_size, w-1)

  def __getitem__(self, idx):
    return self.boundary[idx], self.domain[idx], self.hidden_states[idx]

  def update_items(self, idx, boundary, domain, hidden_state):
    self.boundary[idx] = boundary
    self.domain[idx] = domain
    self.hidden_states[idx] = hidden_state

  def __len__(self):
    return self.dataset_size

