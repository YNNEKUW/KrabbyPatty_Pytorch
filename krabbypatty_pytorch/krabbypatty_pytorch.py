import torch
import torch.nn as nn
import torch.nn.functional as F
from einpos import repeat, rearrange

class NMF(nn.Module):
    def __init__(self, input_dim, inner_dim, K=6):
        super().__init__()
        C = nn.init.uniform_(torch.zeros(inner_dim, input_dim))
        
        self.C = nn.Parameter(C)
        self.K = K
    
    def forward(self, input_tensor):
        input_length, batch_size, input_dim
        # non-negative
        input_tensor = F.relu(input_tensor)


        


class KrabbyPatty(nn.Module):
    def __init__(self, input_dim, n, inner_dim=None, K=6):
        super().__init__()
        # check whether "breads" change the dimensions
        if inner_dim is None:
            inner_dim = input_dim
        
        self.inner_dim = inner_dim
        self.K = K

        # There are no parameters in Ham
        self.lower_bread = nn.Linear(input_dim, inner_dim)
        self.upper_bread = nn.Linear(inner_dim, input_dim)
    

    def forward(self, input_tensor):
        input_length, batch_size, hidden_dim = input_tensor.shape
        input_tensor = input_tensor.flatten(2)

        input_tensor = self.lower_bread(input_tensor)
        # Ham
        input_tensor = F.relu(input_tensor)
        D = nn.init.uniform_(torch.zeros(input_length, self.inner_dim))
        C = nn.init.uniform_(torch.zeros(self.inner_dim, hidden_dim))


        input_tensor = self.upper_bread(input_tensor)

        return input_tensor.reshape(input_length, batch_size, hidden_dim)