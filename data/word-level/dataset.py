from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
import torch 

dataset = TUDataset(root="./old", name="MUTAG")


print(torch.Tensor([dataset[i].y for i in range(188)]).unique())