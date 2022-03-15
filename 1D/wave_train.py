from numpy import integer
from dataset import WaveDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

# bs = batch size?

class WaveModel(nn.Module):
  def __init__(self, z_order, v_order,hidden_size=8,interpolation_size=5, input_size=2):
    """
    :param v_order: highest order of spline for velocity potential (should be at least 2)
    :param z_order: highest order of spline for wave field
    :param hidden_size: hidden size of neural net
    :param interpolation_size: size of first interpolation layer for z_cond and z_mask
    :param input_size: the number of types of boundary conditions that the network takes in. This is usually dirichlet and emitter boundaries.
    """
    super(WaveModel, self).__init__()
    
    # hidden state needs to account for all possible z and v orders.
    hidden_state_size = z_order + v_order + 1 # this should have an extra +1 I think....

    self.interpol = nn.Conv1d(input_size,interpolation_size,kernel_size=2) # interpolate z_cond (2) and z_mask (1) from 4 surrounding fields
    self.conv1 = nn.Conv1d(hidden_state_size+interpolation_size, hidden_size,kernel_size=3,padding=1) # input: hidden_state + interpolation of v_cond and v_mask
    self.conv2 = nn.Conv1d(hidden_size, hidden_size,kernel_size=3,padding=1) # input: hidden_state + interpolation of v_cond and v_mask
    self.conv3 = nn.Conv1d(hidden_size, hidden_state_size,kernel_size=3,padding=1) # input: hidden_state + interpolation of v_cond and v_mask
    
    # if self.hidden_state_size == 18: # if orders_z = 2
    #   self.output_scaler_wave = toCuda(torch.Tensor([5,0.5,0.05,0.5, 0.05,0.05,0.05,0.05,0.05, 5,0.5,0.05,0.5, 0.05,0.05,0.05,0.05,0.05]).unsqueeze(0).unsqueeze(2).unsqueeze(3))
    # elif self.hidden_state_size == 8: # if orders_z = 1
    #   self.output_scaler_wave = toCuda(torch.Tensor([5,0.5,0.5,0.05, 5,0.5,0.5,0.05]).unsqueeze(0).unsqueeze(2).unsqueeze(3))
    
     
  
  def forward(self,hidden_state,boundary_condition,emitter_values):
    """
    :hidden_state: old hidden state of size: bs x hidden_state_size x (w-1) 
    :boundary_condition: (dirichlet) conditions on boundaries (average value within cell): bs x 1 x w
    :emitter_values: says what values are on the emitter boundaries: bs x 1 x w
    :return: new hidden state of size: bs x hidden_state_size x (w-1)
    """
    x = torch.cat([boundary_condition,emitter_values],dim=1)
    
    x = self.interpol(x)
    
    x = torch.cat([hidden_state,x],dim=1)
    x = torch.relu(self.conv1(x))
    x = torch.relu(self.conv2(x))
    x = self.conv3(x)
    
    # residual connections
    return torch.tanh((x[:,:,:]+hidden_state[:,:,:]))




def train(dataset: WaveDataset, epochs: integer, n_batches: integer, n_samples: integer):
  model = WaveModel(2,2)
  optimizer = Adam(lr=0.0001)
  for epoch in range(epochs):
    print(f"{epoch} / {epochs}")

    data_loader = DataLoader(dataset, batch_size=100)

    for i, data in enumerate(data_loader):
      boundary, domain, hidden_state = data
      optimizer.zero_grad()
      outputs = model(hidden_state, boundary, domain)


if __name__ == "__main__":
  model = WaveModel(2,2)
  grid_size = 200
  z_boundary_cond = F.pad(torch.zeros(1,1,grid_size - 4),(2,2), value=1) 
  z_emitter_mask = F.pad(torch.ones(1,1,4), (grid_size // 2 -2, grid_size // 2 - 2), value = 0)
  hidden_state = torch.randn(1,5,199)
  new_state = model.forward(hidden_state, z_boundary_cond, z_emitter_mask)
  x = 1


