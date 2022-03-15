from numpy import integer
from dataset import WaveDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
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
    
     
  
  def forward(self,hidden_state,boundary_boolean_mask,boundary_values):
    """
    :hidden_state: old hidden state of size: bs x hidden_state_size x (w-1) 
    :boundary_boolean_mask: (dirichlet) conditions on boundaries (average value within cell): bs x 1 x w
    :boundary_values: says what values are on the emitter boundaries: bs x 1 x w
    :return: new hidden state of size: bs x hidden_state_size x (w-1)
    """
    x = torch.cat([boundary_boolean_mask,boundary_values],dim=1)
    
    x = self.interpol(x)
    
    x = torch.cat([hidden_state,x],dim=1)
    x = torch.relu(self.conv1(x))
    x = torch.relu(self.conv2(x))
    x = self.conv3(x)
    
    # residual connections
    return torch.tanh((x[:,:,:]+hidden_state[:,:,:]))




def train(dataset: WaveDataset, epochs: integer, n_batches: integer, n_samples: integer):
  # initialize model and move it to cuda device if available.
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = WaveModel(2,2).to(device)
  optimizer = Adam(lr=0.0001)
  for epoch in range(epochs):
    print(f"{epoch} / {epochs}")

    data_loader = DataLoader(dataset, batch_size=100)

    for i, data in enumerate(data_loader):
      boundary, domain, hidden_state = data
      optimizer.zero_grad()
      outputs = model(hidden_state, boundary, domain)

      loss = 12300123
      model.zero_grad()
      loss.backward()

def compute_loss(sample_boundary_boolean_mask, sample_boundary_values, 
                previous_state, b_loss_weight = 0.1, v_loss_weight = 0.1, 
                z_loss_weight = 0.1):
  # compute new z, grad_z, dz_dt, v, a 
  z, grad_z, dz_dt, v, a

  # compute boundary loss - the value should be 
  # need to update these for 2 dimensions
  loss_boundary = torch.mean(sample_boundary_boolean_mask[:,:,1:-1] * ( (z - sample_boundary_values[:,:,1:-1])**2 ))

  # not sure if this is needed or what it accomplishes. Need to ask.
  # loss_boundary_reg = torch.mean(sample_boundary_mask[:,:,1:-1] * a**2

  # loss to connect dz_dt and v
  loss_v = torch.mean((v - dz_dt)**2)

  loss_z = torch.mean((a - stiffness * laplace_z + damping * v)**2)

  return b_loss_weight * loss_boundary + v_loss_weight * loss_v + z_loss_weight * loss_z


class Loss_Calculator():
  def __init__(self, stiffness, damping):
    self.stiffness = stiffness
    self.damping = damping

  def compute_loss(self, sample_boundary_boolean_mask, sample_boundary_values, 
                  previous_state, b_loss_weight = 0.1, v_loss_weight = 0.1, 
                  z_loss_weight = 0.1):
    # compute new z, grad_z, dz_dt, v, a 
    z, grad_z, dz_dt, v, a

    # compute boundary loss - the value should be 
    # need to update these for 2 dimensions
    loss_boundary = torch.mean(sample_boundary_boolean_mask[:,:,1:-1] * ( (z - sample_boundary_values[:,:,1:-1])**2 ))

    # not sure if this is needed or what it accomplishes. Need to ask.
    # loss_boundary_reg = torch.mean(sample_boundary_mask[:,:,1:-1] * a**2

    # loss to connect dz_dt and v
    loss_v = torch.mean((v - dz_dt)**2)

    # loss for z
    loss_z = torch.mean((a - self.stiffness * laplace_z + self.damping * v)**2)

    return b_loss_weight * loss_boundary + v_loss_weight * loss_v + z_loss_weight * loss_z

if __name__ == "__main__":
  model = WaveModel(2,2)
  grid_size = 200
  z_boundary_cond = F.pad(torch.zeros(1,1,grid_size - 4),(2,2), value=1) 
  z_emitter_mask = F.pad(torch.ones(1,1,4), (grid_size // 2 -2, grid_size // 2 - 2), value = 0)
  hidden_state = torch.randn(1,5,199)
  new_state = model.forward(hidden_state, z_boundary_cond, z_emitter_mask)
  x = 1


