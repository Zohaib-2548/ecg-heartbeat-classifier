"""Task 5: evaluate the saved model on the test set (once)."""
import json
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from prepare import prepare
from model import ECGNet
from metrics import accuracy, confusion_matrix, per_class_accuracy, balanced_accuracy

NAMES = ["N", "S", "V", "F", "Q"]

if __name__ == "__main__":
    data, _ = prepare()
    model = ECGNet()
    model.load_state_dict(torch.load("results/best_model.pt"))
    with torch.no_grad():
        pred = model(data["X_test"]).argmax(dim=1)
    y = data["y_test"]

    C = confusion_matrix(y, pred)
    acc = accuracy(y, pred)
    pc = per_class_accuracy(C)
    bal = balanced_accuracy(C)
    always_n = (y == 0).float().mean().item()

    print("Confusion matrix (rows = true, cols = predicted):\n", C)
    print("sum of all entries:", C.sum().item(), "(should be 21892)")
    for n, a in zip(NAMES, pc):
        print(f"per-class accuracy {n}: {a:.4f}")
    print(f"test accuracy: {acc:.4f}\nbalanced accuracy: {bal:.4f}\nalways-N accuracy: {always_n:.4f}")

    # Task 6.2 helper: most confused class for each true class
    for j in range(5):
        row = C[j].clone(); row[j] = 0
        k = row.argmax().item()
        print(f"true {NAMES[j]} is most often mistaken for {NAMES[k]} ({row[k].item()} rows)")

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(C.numpy(), cmap="Blues")
    ax.set_xticks(range(5)); ax.set_xticklabels(NAMES)
    ax.set_yticks(range(5)); ax.set_yticklabels(NAMES)
    ax.set_xlabel("predicted class"); ax.set_ylabel("true class")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, int(C[i, j]), ha="center", va="center",
                    color="white" if C[i, j] > C.max() / 2 else "black", fontsize=9)
    fig.colorbar(im); fig.tight_layout()
    fig.savefig("results/confusion_matrix.png", dpi=150)

    with open("results/test_results.json", "w") as fh:
        json.dump({"accuracy": acc, "balanced_accuracy": bal, "always_N_accuracy": always_n,
                   "per_class_accuracy": dict(zip(NAMES, pc)),
                   "confusion_matrix": C.tolist()}, fh, indent=2)
