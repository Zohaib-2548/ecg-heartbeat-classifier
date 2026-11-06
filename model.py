"""Task 2 + 3.1: ECGNet (built from raw weight matrices) and softmax cross-entropy."""
import math
import torch
import torch.nn as nn

class ECGNet(nn.Module):
    def __init__(self, sizes=(187, 128, 64, 5)):
        super().__init__()
        self.W = nn.ParameterList()
        self.b = nn.ParameterList()
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            # He init: std = sqrt(2 / n_in), bias = 0. W has shape (n_out, n_in)
            self.W.append(nn.Parameter(torch.randn(n_out, n_in) * math.sqrt(2.0 / n_in)))
            self.b.append(nn.Parameter(torch.zeros(n_out)))

    def forward(self, x, return_hidden=False):
        a = x                                       # A[0] = X, shape (B, 187)
        last = len(self.W) - 1
        hidden = []
        for l in range(len(self.W)):
            z = a @ self.W[l].T + self.b[l]         # Z[l] = A[l-1] W[l]^T + b[l]
            if l < last:
                a = torch.clamp(z, min=0)           # ReLU written by hand
                hidden.append(a)
        if return_hidden:
            return z, hidden
        return z                                    # logits, shape (B, 5)

def cross_entropy(logits, y):
    """Mean of  -z_y + log sum_j exp(z_j)  over the batch."""
    lse = torch.logsumexp(logits, dim=1)
    z_y = logits.gather(1, y.unsqueeze(1)).squeeze(1)
    return (lse - z_y).mean()

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    torch.manual_seed(0)
    m = ECGNet()
    print("trainable parameters:", count_params(m))
    print("by hand: (187*128+128) + (128*64+64) + (64*5+5) =",
          187*128+128, "+", 128*64+64, "+", 64*5+5, "=", 187*128+128 + 128*64+64 + 64*5+5)
