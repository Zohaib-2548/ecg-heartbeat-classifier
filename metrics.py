"""Task 5.2: metrics written by hand."""
import torch

def accuracy(y_true, y_pred):
    return (y_true == y_pred).float().mean().item()

def confusion_matrix(y_true, y_pred, k=5):
    """C[j, k] = rows with true class j predicted as k."""
    return torch.bincount(y_true * k + y_pred, minlength=k * k).reshape(k, k)

def per_class_accuracy(C):
    return (torch.diagonal(C).float() / C.sum(dim=1).float()).tolist()

def balanced_accuracy(C):
    pc = per_class_accuracy(C)
    return sum(pc) / len(pc)
