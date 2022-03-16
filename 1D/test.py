from dataset import WaveDataset
from torch.utils.data import DataLoader
import copy
import torch
import unittest

class TestDataSet(unittest.TestCase):
  def test_dataset_change(self):
    domain_width = 200
    dataset = WaveDataset(domain_width)
    dataloader = DataLoader(dataset, 100)

    data_iterable = iter(dataloader)

    first_items_copy = copy.deepcopy(dataset.__getitem__(0))

    fake_item = torch.ones(domain_width)
    fake_state = torch.ones(domain_width - 1)
    dataset.update_items(0, fake_item,fake_item,fake_state)

    first_items_real = next(data_iterable)

    for real_item, copy_item in zip(first_items_copy, first_items_real):
      self.assertFalse(torch.equal(real_item, copy_item))

  
if __name__ == "main":
  unittest.main()

