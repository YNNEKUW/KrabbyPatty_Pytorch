import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange



class KrabbyPatty(nn.Module):
    def __init__(self, input_dim, inner_dim=None, ratio=8, K=6, eps=1e-9):
        super().__init__()
        # Check whether "breads" change the dimensions
        if inner_dim is None:
            inner_dim = input_dim
        
        self.inner_dim = inner_dim
        self.K = K
        self.r = self.inner_dim // ratio
        self.eps = eps

        # There are no parameters in Ham
        self.lower_bread = nn.Linear(input_dim, self.inner_dim)
        self.upper_bread = nn.Linear(self.inner_dim, input_dim)
    

    def forward(self, input_tensor):
        input_length, batch_size, hidden_dim = input_tensor.shape
        # X = input_tensor.flatten(2)

        X = rearrange(input_tensor, 'l b d -> b l d')
        X = self.lower_bread(X)
        # Ham
        X = F.relu(X)
        
        # Check the device type
        device = X.device
        
        D = nn.init.uniform_(torch.zeros(input_length, self.r, device=device))
        C = nn.init.uniform_(torch.zeros(self.r, self.inner_dim, device=device))
        D = repeat(D, 'l r -> b l r', b=batch_size)
        C = repeat(C, 'r d -> b r d', b=batch_size)

        # Transpose for batch
        def t_b(t): return rearrange(t, 'b i j -> b j i')
        for i in range(self.K):
            if i == self.K - 1:
                # D should be generated by the updated C ??
                # Here is an optimization problem
                # Should check the magnitude of "l" and "r"
                C = C * (t_b(D) @ X) / (t_b(D) @ D @ C + self.eps)
                D = D * (X @ t_b(C)) / (D @ C @ t_b(C) + self.eps)
            else:
                with torch.no_grad():
                    C = C * (t_b(D) @ X) / (t_b(D) @ D @ C + self.eps)
                    D = D * (X @ t_b(C)) / (D @ C @ t_b(C) + self.eps)
        
        X = D @ C
        X = self.upper_bread(X)
        input_tensor = rearrange(X, 'b l d -> l b d')


        return input_tensor