"""Task 4: manual mini-batch SGD with momentum + L2 weight decay."""
import copy, json, os, time
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from prepare import prepare
from model import ECGNet, cross_entropy, count_params

BATCH, BETA, LAM, EPOCHS = 128, 0.9, 1e-4, 30
ALPHAS = [0.01, 0.1, 0.5]

@torch.no_grad()
def evaluate(model, X, y, bs=8192):
    tot_loss, correct = 0.0, 0
    for i in range(0, len(y), bs):
        logits = model(X[i:i + bs])
        tot_loss += cross_entropy(logits, y[i:i + bs]).item() * len(y[i:i + bs])
        correct += (logits.argmax(dim=1) == y[i:i + bs]).sum().item()
    return tot_loss / len(y), correct / len(y)

def train(alpha, data, epochs=EPOCHS, verbose=True):
    torch.manual_seed(0)
    model = ECGNet()
    # (parameter, is_weight) -> weight decay only on weights
    params = [(p, True) for p in model.W] + [(p, False) for p in model.b]
    vel = [torch.zeros_like(p) for p, _ in params]
    Xtr, ytr = data["X_train"], data["y_train"]
    n = len(ytr)
    gen = torch.Generator().manual_seed(0)
    hist = {k: [] for k in ["train_loss", "val_loss", "train_acc", "val_acc"]}
    best = {"val_loss": float("inf"), "epoch": -1, "state": None}
    t0 = time.time()
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n, generator=gen)              # shuffle every epoch
        for i in range(0, n, BATCH):
            bi = perm[i:i + BATCH]
            loss = cross_entropy(model(Xtr[bi]), ytr[bi])
            for p, _ in params:
                p.grad = None
            loss.backward()
            with torch.no_grad():
                for (p, is_w), v in zip(params, vel):
                    g = p.grad + LAM * p if is_w else p.grad
                    v.mul_(BETA).add_((1 - BETA) * g)         # v = beta v + (1-beta) g
                    p.sub_(alpha * v)                         # P = P - alpha v
        tl, ta = evaluate(model, Xtr, ytr)
        vl, va = evaluate(model, data["X_val"], data["y_val"])
        for k, val in zip(hist, [tl, vl, ta, va]):
            hist[k].append(val)
        if vl < best["val_loss"]:
            best.update(val_loss=vl, epoch=ep, state=copy.deepcopy(model.state_dict()))
        if verbose:
            print(f"alpha={alpha} epoch {ep:2d} | train loss {tl:.4f} acc {ta:.4f} | val loss {vl:.4f} acc {va:.4f}")
    return hist, best, time.time() - t0

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    data, _ = prepare()

    torch.manual_seed(0)
    init_model = ECGNet()
    init_loss, _ = evaluate(init_model, data["X_train"], data["y_train"])
    n_params = count_params(init_model)
    print(f"parameters: {n_params}, initial train loss: {init_loss:.4f} (log 5 = 1.6094)")

    runs = {}
    for a in ALPHAS:
        runs[a] = train(a, data)

    # Figure: val loss for the 3 learning rates
    plt.figure(figsize=(6, 4))
    for a in ALPHAS:
        plt.plot(range(1, EPOCHS + 1), runs[a][0]["val_loss"], label=f"alpha={a}")
    plt.xlabel("epoch"); plt.ylabel("validation loss"); plt.legend(); plt.yscale("log")
    plt.title("Validation loss for three learning rates"); plt.tight_layout()
    plt.savefig("results/lr_comparison.png", dpi=150); plt.close()

    best_alpha = min(ALPHAS, key=lambda a: runs[a][1]["val_loss"])
    hist, best, secs = runs[best_alpha]
    print(f"chosen alpha = {best_alpha}, best epoch = {best['epoch']}")

    for keys, name, yl in [(("train_loss", "val_loss"), "loss_curves", "loss"),
                           (("train_acc", "val_acc"), "acc_curves", "accuracy")]:
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, EPOCHS + 1), hist[keys[0]], label="train")
        plt.plot(range(1, EPOCHS + 1), hist[keys[1]], label="validation")
        plt.xlabel("epoch"); plt.ylabel(yl); plt.legend()
        plt.title(f"{yl} (alpha={best_alpha})"); plt.tight_layout()
        plt.savefig(f"results/{name}.png", dpi=150); plt.close()

    torch.save(best["state"], "results/best_model.pt")

    e = best["epoch"] - 1
    hw = "CPU"
    if torch.cuda.is_available():
        hw = torch.cuda.get_device_name(0)
    summary = {
        "params": n_params, "initial_loss": init_loss, "chosen_alpha": best_alpha,
        "best_epoch": best["epoch"],
        "train_loss": hist["train_loss"][e], "val_loss": hist["val_loss"][e],
        "train_acc": hist["train_acc"][e], "val_acc": hist["val_acc"][e],
        "train_time_sec": secs, "hardware": hw,
        "lr_table": {str(a): {"lowest_val_loss": runs[a][1]["val_loss"],
                              "val_acc_at_that_epoch": runs[a][0]["val_acc"][runs[a][1]["epoch"] - 1],
                              "epoch": runs[a][1]["epoch"]} for a in ALPHAS},
    }
    with open("results/train_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    with open("results/history.json", "w") as fh:
        json.dump({str(a): runs[a][0] for a in ALPHAS}, fh)
    print(json.dumps(summary, indent=2))
