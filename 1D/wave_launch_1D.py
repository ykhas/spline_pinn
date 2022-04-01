from sympy import interpolate
import torch
from dataset import WaveDataset
from interpolations import KernelValuesHolder, interpolate_wave_in_time, interpolate_kernels
from matplotlib import pyplot as plt
from wave_train_1D import WaveModel
from const import TORCH_DTYPE


if __name__=="__main__":
    num_grid_points = 100
    highest_z_order = 2
    num_support_points = 5
    data_index = 0
    model = WaveModel(highest_z_order, highest_z_order)
    model.load_state_dict(torch.load("/home/yaniv/Documents/Research/Spline_PINN/1D/models/model"))
    model.eval()

    dataset = WaveDataset(num_grid_points, model.num_kernels)

    kernel_values_holder = KernelValuesHolder(num_support_points, highest_z_order + 1) # assuming z order and v order are equal... might need to change that assumption

    boundary, values, old_hidden_state = dataset.get_unsqueezed_item(data_index)

    new_state = model(old_hidden_state, boundary, values)

    z = interpolate_wave_in_time(old_hidden_state[:,:model.index_of_v_coefficients,:],
                                          new_state[:,:model.index_of_v_coefficients,:],
                                          old_hidden_state[:, model.index_of_v_coefficients:, :],
                                          new_state[:, model.index_of_v_coefficients:, :],
                                          kernel_values_holder.kernel_values_and_derivs[model.index_of_v_coefficients:, :, :],
                                          kernel_values_holder.kernel_values_and_derivs[:model.index_of_v_coefficients, :, :],
                                          dataset.time_step)[0]
    
    for i in range(30):
        boundary, values, old_hidden_state = dataset.get_unsqueezed_item(data_index)

        new_state = model(old_hidden_state, boundary, values)
        new_z = interpolate_kernels(new_state[:,:model.index_of_v_coefficients,:], 
            kernel_values_holder.kernel_values_and_derivs[:model.index_of_v_coefficients, :, :])
        plt.plot(boundary[0,0,:].detach().numpy(), 'x')
        plt.plot(values[0,0,:].detach().numpy(), '.')
        plt.plot(new_z[0,0,:].detach().numpy(), '-')
        plt.show()
        dataset.evolve_boundary()
    





