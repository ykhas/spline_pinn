from copy import deepcopy
from numpy import integer
from dataset import WaveDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import torch
from interpolations import KernelValuesHolder, interpolate_wave_in_time, total_num_kernels, interpolate_kernels
from numbers import Number
import math

class Loss_Calculator():
    def __init__(self, stiffness, damping, b_loss_weight=200, v_loss_weight=1,
                 z_loss_weight=1):
        self.stiffness = stiffness
        self.damping = damping
        self.b_loss_weight = b_loss_weight
        self.v_loss_weight = v_loss_weight
        self.z_loss_weight = z_loss_weight


    def diffuse(self, domain):
        kernel_width = 3
        kernel = torch.exp(-torch.arange(-2,2.001,4/(2*kernel_width)).float()**2).cuda()
        return F.conv1d(domain, kernel.reshape(1,1,-1), padding=kernel_width)

    def compute_loss(self, sample_boundary_boolean_mask, sample_boundary_values,
                     old_coefficients_z, new_coefficients_z,
                     old_coefficients_v, new_coefficients_v,
                     zero_deriv_kernels, laplace_kernels, time_step: Number):

        z, laplace_z, dz_dt, v, a = interpolate_wave_in_time(old_coefficients_z,
                                                             new_coefficients_z,
                                                             old_coefficients_v,
                                                             new_coefficients_v,
                                                             zero_deriv_kernels,
                                                             laplace_kernels,
                                                             time_step=time_step)

        # compute boundary loss - the value should be
        # need to update these for 2 dimensions

        # interpolated_mask = interpolate_kernels(sample_boundary_boolean_mask[:, :, 1:-1], kernel_values_and_derivs)[0]
        # interpolated_boundary_vals = interpolate_kernels(sample_boundary_values[:, :, 1:-1], kernel_values_and_derivs)[0]
        sample_boundary_boolean_mask = sample_boundary_boolean_mask + sample_boundary_boolean_mask * self.diffuse(1 - sample_boundary_boolean_mask)
        upsampled_boundary_mask = F.interpolate(sample_boundary_boolean_mask[:,:,:], z.shape[2])
        upsampled_boundary_values = F.interpolate(sample_boundary_values[:,:,:], z.shape[2])
        loss_boundary = torch.mean(
            upsampled_boundary_mask[:, :, :] * ((z - upsampled_boundary_values[:, :, :])**2))

        # not sure if this is needed or what it accomplishes. Need to ask.
        # loss_boundary_reg = torch.mean(upsampled_boundary_mask[:,:,:] * a**2)

        # loss to connect dz_dt and v
        loss_v = torch.mean((v - dz_dt)**2)

        # loss for z
        loss_z = torch.mean(
            (a - self.stiffness * laplace_z + self.damping * v)**2)

        return self.b_loss_weight * loss_boundary + self.v_loss_weight * loss_v + self.z_loss_weight * loss_z# + loss_boundary_reg


class WaveModel(nn.Module):
    def __init__(self, z_order, v_order, hidden_size=8, interpolation_size=5, input_size=2):
        """
        :param v_order: highest order of spline for velocity potential (should be at least 2)
        :param z_order: highest order of spline for wave field
        :param hidden_size: hidden size of neural net
        :param interpolation_size: size of first interpolation layer for z_cond and z_mask
        :param input_size: the number of types of boundary conditions that the network takes in. This is usually dirichlet and emitter boundaries.
        """
        super(WaveModel, self).__init__()

        # hidden state needs to account for all possible kernels which might be used to construct
        # splines with the requested orders
        self.num_kernels = total_num_kernels(
            z_order + 1) + total_num_kernels(v_order + 1)
        # first section of coefficients used to predict evolution of z wave, second section used to predict v wave
        self.index_of_v_coefficients = total_num_kernels(z_order + 1)

        # interpolate z_cond (2) and z_mask (1) from 4 surrounding fields
        self.interpol = nn.Conv1d(
            input_size, interpolation_size, kernel_size=2)
        # input: hidden_state + interpolation of v_cond and v_mask
        self.conv1 = nn.Conv1d(
            self.num_kernels+interpolation_size, hidden_size, kernel_size=3, padding=1)
        # input: hidden_state + interpolation of v_cond and v_mask
        self.conv2 = nn.Conv1d(hidden_size, hidden_size,
                               kernel_size=3, padding=1)
        # input: hidden_state + interpolation of v_cond and v_mask
        self.conv3 = nn.Conv1d(
            hidden_size, self.num_kernels, kernel_size=3, padding=1)

        # if self.hidden_state_size == 18: # if orders_z = 2
        #   self.output_scaler_wave = toCuda(torch.Tensor([5,0.5,0.05,0.5, 0.05,0.05,0.05,0.05,0.05, 5,0.5,0.05,0.5, 0.05,0.05,0.05,0.05,0.05]).unsqueeze(0).unsqueeze(2).unsqueeze(3))
        # elif self.hidden_state_size == 8: # if orders_z = 1
        #   self.output_scaler_wave = toCuda(torch.Tensor([5,0.5,0.5,0.05, 5,0.5,0.5,0.05]).unsqueeze(0).unsqueeze(2).unsqueeze(3))

    def forward(self, hidden_state, boundary_boolean_mask, boundary_values):
        """
        :hidden_state: old hidden state of size: bs x hidden_state_size x (w-1) 
        :boundary_boolean_mask: (dirichlet) conditions on boundaries (average value within cell): bs x 1 x w
        :boundary_values: says what values are on the emitter boundaries: bs x 1 x w
        :return: new hidden state of size: bs x hidden_state_size x (w-1)
        """
        x = torch.cat([boundary_boolean_mask, boundary_values], dim=1)

        x = self.interpol(x)

        x = torch.cat([hidden_state, x], dim=1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)

        # residual connections
        return torch.tanh((x[:, :, :]+hidden_state[:, :, :]))


def train(num_grid_points: integer, epochs: integer, n_batches: integer, n_samples: integer, loss_calc: Loss_Calculator,
            save_model = False):
    # initialize model and move it to cuda device if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu_device = torch.device("cpu")
    highest_z_order, highest_v_order = 2, 2
    num_support_points = 11
    model = WaveModel(highest_z_order, highest_v_order).to(device)
    dataset = WaveDataset(num_grid_points, model.num_kernels, num_hidden_state_dataset_size=100)
    optimizer = Adam(model.parameters(), lr=0.00007)
    kernel_values_holder = KernelValuesHolder(num_support_points, highest_z_order + 1, device) # assuming z order and v order are equal... might need to change that assumption

    record_loss_state = None
    record_loss = 3000
    for epoch in range(epochs):
        print(f"{epoch} / {epochs}")

        batch_size = 10
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for i, data in enumerate(data_loader):
            boolean_mask, boundary_value, old_hidden_state, index = data
            boolean_mask = boolean_mask.to(device)
            boundary_value = boundary_value.to(device)
            old_hidden_state = old_hidden_state.to(device)

            optimizer.zero_grad()

            output = model(old_hidden_state, boolean_mask, boundary_value)

            cloned_kernels = kernel_values_holder.kernel_values_and_derivs.detach().clone()
            loss = loss_calc.compute_loss(boolean_mask, boundary_value,
                                          old_hidden_state[:, :model.index_of_v_coefficients, :],
                                          output[:, :model.index_of_v_coefficients, :],
                                          old_hidden_state[:, model.index_of_v_coefficients:, :],
                                          output[:, model.index_of_v_coefficients:, :],
                                          cloned_kernels[:model.index_of_v_coefficients, :, :],
                                          cloned_kernels[model.index_of_v_coefficients:, :, :],
                                          time_step=dataset.time_step)
            
            print(f"loss is:{loss}")
            loss.backward()
            optimizer.step()
            dataset.update_items(index, output.detach().to(cpu_device))
            dataset.evolve_boundary()    
            # if loss < record_loss:
            #     record_loss_state = deepcopy(model.state_dict())
            #     record_loss = loss
        
        if epoch % 5 == 0:
            dataset.reset_hidden_states(batch_size)
    
    if save_model:
        torch.save(model.state_dict(), '/home/yaniv/Documents/Research/Spline_PINN/1D/models/model')
    


if __name__ == "__main__":
    loss_calc = Loss_Calculator(0.1, 0.5)
    train(num_grid_points = 200, epochs = 100, n_batches = 1, n_samples = 1, loss_calc=loss_calc, save_model=True)
