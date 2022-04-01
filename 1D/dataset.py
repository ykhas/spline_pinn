from tokenize import Number
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from const import TORCH_DTYPE

class WaveDataset(Dataset):
  def __init__(self, width, num_kernels, num_hidden_state_dataset_size=1000, time_step = 0.1):
    self.num_hidden_state_dataset_size = num_hidden_state_dataset_size
    self.width = width
    self.time_step = time_step
    self.emitter_size = 3
    # self.true_domain_size = width * num_support_points - (width - 2) * (num_support_points // 2)

    self.boolean_masks = torch.zeros(num_hidden_state_dataset_size, 1, width, dtype=TORCH_DTYPE)
    self.boundary_values = torch.zeros(num_hidden_state_dataset_size, 1, width, dtype=TORCH_DTYPE)
    self.hidden_states = torch.zeros(num_hidden_state_dataset_size, num_kernels, width-1, dtype=TORCH_DTYPE)
    self.phases = torch.randn(self.num_hidden_state_dataset_size)
    self.__initialize_boundaries()

  def __initialize_boundaries(self):
    '''
    Initializes random sinusoidal boundary conditions
    '''
    self.boolean_masks[:, 0, -self.emitter_size:] = 1 # set domain boundary
    self.boolean_masks[:, 0, :self.emitter_size] = 1 # set domain boundary
    sinusoids = torch.sin(self.phases.unsqueeze(1)).expand(-1, self.emitter_size) # create random sinusoids as the emitter's state
    self.boolean_masks[:, 0, self.width//2: self.width // 2 + self.emitter_size] = 1 # emitter boundary
    self.boundary_values[:, 0, self.width//2: self.width // 2 + self.emitter_size] = 0.5 #sinusoids

  def reset_hidden_states(self, num_indices, all=False):
    if not all:
      indices = torch.randint(0, self.num_hidden_state_dataset_size, (1, num_indices))
      self.hidden_states[indices] = 0
    else:
      self.hidden_states[:,:,:] = 0

  def evolve_boundary(self):
    '''
    Propagates boundary condition forward in time
    '''
    return
    self.phases += self.time_step
    sinusoids = torch.sin(self.phases.unsqueeze(1)).expand(-1, self.emitter_size) # create random sinusoids as the emitter's state
    self.boundary_values[:, 0, self.width//2: self.width // 2 + self.emitter_size] = sinusoids

  def __getitem__(self, idx):
    return self.boolean_masks[idx], self.boundary_values[idx], self.hidden_states[idx], idx
  
  def get_unsqueezed_item(self, idx):
    return self.boolean_masks[[idx]], self.boundary_values[[idx]], self.hidden_states[[idx]]

  def update_items(self, idx, hidden_state):
    return
    self.hidden_states[idx] = hidden_state

  def __len__(self):
    return self.num_hidden_state_dataset_size

