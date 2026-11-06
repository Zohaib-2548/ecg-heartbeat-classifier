"""Task 3.2: compare hand-written output-layer gradients with autograd."""
import torch
from prepare import prepare
from model import ECGNet, cross_entropy

if __name__ == "__main__":
    data, _ = prepare()
    torch.manual_seed(0)
    model = ECGNet()
    x, y = data["X_train"][:64], data["y_train"][:64]
    B = x.shape[0]

    logits, hidden = model(x, return_hidden=True)
    loss = cross_entropy(logits, y)
    loss.backward()

    A2 = hidden[1]                                      # (B, 64)
    Yhat = torch.softmax(logits.detach(), dim=1)        # (B, 5)
    Y = torch.zeros_like(Yhat).scatter_(1, y.unsqueeze(1), 1.0)
    dW3 = (Yhat - Y).T @ A2.detach() / B                # (5, 64)
    db3 = (Yhat - Y).sum(dim=0) / B                     # (5,)

    dW_diff = (dW3 - model.W[2].grad).abs().max().item()
    db_diff = (db3 - model.b[2].grad).abs().max().item()
    print(f"loss on 64 rows: {loss.item():.4f}")
    print(f"max |dW3 hand - autograd| = {dW_diff:.3e}")
    print(f"max |db3 hand - autograd| = {db_diff:.3e}")
    print("PASS (< 1e-5)" if max(dW_diff, db_diff) < 1e-5 else "FAIL")
    with open("results/grad_check.txt", "w") as fh:
        fh.write(f"max diff W3 = {dW_diff:.3e}\nmax diff b3 = {db_diff:.3e}\n")
