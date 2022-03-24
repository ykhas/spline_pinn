from dataset import WaveDataset
from wave_train_1D import WaveModel, Loss_Calculator, train
from torch.utils.data import DataLoader
from interpolations import KernelValuesHolder
import torch.nn.functional as F
import torch
import copy
import torch
import unittest

domain_width = 200

class TestDataSet(unittest.TestCase):
    def test_dataset_change(self):
        random_num_kernels = 10
        dataset = WaveDataset(domain_width, random_num_kernels)
        dataloader = DataLoader(dataset, 100)

        data_iterable = iter(dataloader)

        # first_items_copy = copy.deepcopy(dataset.__getitem__(0))

        # fake_item = torch.ones(domain_width)
        fake_state = torch.ones(domain_width - 1)
        # dataset.update_items(0, fake_item, fake_item, fake_state)
        dataset.update_items(0, fake_state)

        first_items_real = next(data_iterable)

        # for real_item, copy_item in zip(first_items_copy, first_items_real):
        #     self.assertFalse(torch.equal(real_item, copy_item))
        self.assertFalse(torch.equal(first_items_real[2], fake_state))

    def test_dataset_phase(self):
        torch.manual_seed(0)
        arbitrary_num_kernels = 10
        expected_boundary_values = torch.tensor([[[0.000000000,  0.000000000,  0.000000000,  0.000000000,  0.000000000,
                                                   0.999556005,  0.999556005,  0.000000000,  0.000000000,  0.000000000]],

                                                 [[0.000000000,  0.000000000,  0.000000000,  0.000000000,  0.000000000,
                                                   -0.289236248, -0.289236248,  0.000000000,  0.000000000,  0.000000000]],

                                                 [[0.000000000,  0.000000000,  0.000000000,  0.000000000,  0.000000000,
                                                   -0.820796072, -0.820796072,  0.000000000,  0.000000000,  0.000000000]],

                                                 [[0.000000000,  0.000000000,  0.000000000,  0.000000000,  0.000000000,
                                                   0.538310647,  0.538310647,  0.000000000,  0.000000000,  0.000000000]],

                                                 [[0.000000000,  0.000000000,  0.000000000,  0.000000000,  0.000000000,
                                                   -0.884080291, -0.884080291,  0.000000000,  0.000000000,  0.000000000]]])

        dataset = WaveDataset(10, arbitrary_num_kernels, num_hidden_state_dataset_size=5)
        self.assertTrue(torch.equal(
            dataset.boundary_values, expected_boundary_values))
        
        # now we test that when we propagate the time, the tensor is not the same
        dataset.evolve_boundary()
        self.assertFalse(torch.equal(
            dataset.boundary_values, expected_boundary_values))

class TestUtils:
    @staticmethod
    def generate_sample_state():
        model = WaveModel(2, 2)
        # 1s around border, zero in the middle.
        z_boundary_cond = F.pad(torch.zeros(
            1, 1, domain_width - 4), (2, 2), value=1)
        # emitter is in the middle, zeros elsewhere
        z_emitter_mask = F.pad(torch.ones(
            1, 1, 4), (domain_width // 2 - 2, domain_width // 2 - 2), value=0)
        
        hidden_state = torch.randn(1, model.num_kernels, domain_width - 1)
        return model, z_boundary_cond, z_emitter_mask, hidden_state


class TestForwardModel(unittest.TestCase):
    def test_forward_computation(self):
        model, z_boundary_cond, z_emitter_mask, hidden_state = TestUtils.generate_sample_state()
        model.forward(hidden_state, z_boundary_cond, z_emitter_mask)

    def test_forward_computation2(self):
        model = WaveModel(2,2)
        dataset = WaveDataset(10, model.num_kernels, num_hidden_state_dataset_size=5)
        data_loader = DataLoader(dataset, batch_size=50)
        datum = next(iter(data_loader))
        boolean_mask, boundary_value, old_hidden_state = datum
        model(old_hidden_state, boolean_mask, boundary_value)

class TestKernelValuesHolder(unittest.TestCase):
    def test_kernel_values_even_support(self):
        '''
        Should be unable to create kernel holder for kernels with even numbers
        of support points
        '''

        # check that we can make it with odd amount of support points
        KernelValuesHolder(11,1)
        KernelValuesHolder(11,2)

        # now let's verify that we can't make it with even numbers of points
        failed = True
        try:
            KernelValuesHolder(10,1)
        except ValueError:
            failed = False

        self.assertFalse(failed)
    

# class TestTrainModel(unittest.TestCase):
#     def test_train_iteration(self):
#       dataset = WaveDataset(domain_width)
#       loss_calc = Loss_Calculator(0.1, 0.5)
#       train(dataset, epochs = 1, n_batches = 1, n_samples = 1, loss_calc=loss_calc)


if __name__ == "__main__":
    unittest.main()

    # to run with debugger, comment above line and uncomment the two below.
    # model_train_test = TestTrainModel()
    # model_train_test.test_train_iteration()
    # model_forward = TestForwardModel()
    # model_forward.test_forward_computation2()
    # dataset_test = TestDataSet()
    # dataset_test.test_dataset_phase()
    # kernel_test = TestKernelValuesHolder()
    # kernel_test.test_kernel_values_even_support()
