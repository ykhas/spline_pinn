from tokenize import Number
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass

class WaveDataset(Dataset):
  def __init__(self, width, num_hidden_state_dataset_size=1000):
    self.num_hidden_state_dataset_size = num_hidden_state_dataset_size
    self.width = width

    self.boolean_masks = torch.zeros(num_hidden_state_dataset_size, 1, width)
    self.boundary_values = torch.zeros(num_hidden_state_dataset_size, 1, width)
    self.hidden_states = torch.zeros(num_hidden_state_dataset_size, width-1)
    self.phases = torch.randn(self.num_hidden_state_dataset_size)
    self.__initialize_boundaries()

  def __initialize_boundaries(self):
    '''
    Initializes random sinusoidal boundary conditions
    '''
    self.boolean_masks[:, 0, -1: 2] = 1 # set domain boundary
    emitter_size = 2
    sinusoids = torch.sin(self.phases.unsqueeze(1)).expand(-1, emitter_size) # create random sinusoids as the emitter's state
    self.boolean_masks[:, 0, self.width//2: self.width // 2 + emitter_size] = 1 # emitter boundary
    self.boundary_values[:, 0, self.width//2: self.width // 2 + emitter_size] = sinusoids

  def __getitem__(self, idx):
    return self.boolean_masks[idx], self.boundary_values[idx], self.hidden_states[idx]

  def update_items(self, idx, boolean_mask, boundary_value, hidden_state):
    self.boolean_masks[idx] = boolean_mask 
    self.boundary_values[idx] = boundary_value 
    self.hidden_states[idx] = hidden_state

  def __len__(self):
    return self.num_hidden_state_dataset_size

