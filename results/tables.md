# Results tables

## Table 1 – Data

| Item | Value |
|---|---|
| Train / validation / test rows | 78,799 / 8,755 / 21,892 |
| Minimum / maximum signal value | 0.0 / 1.0 |

| Rows per class | N | S | V | F | Q |
|---|---|---|---|---|---|
| Train file | 72,471 | 2,223 | 5,788 | 641 | 6,431 |
| Test file | 18,118 | 556 | 1,448 | 162 | 1,608 |

## Table 2 – Network and training

| Item | Value |
|---|---|
| Trainable parameters | 32,645 |
| Initial loss (before training) | 1.7174 |
| Gradient check, max difference W[3] / b[3] | 2.98e-08 / 5.96e-08 |
| Chosen α | 0.5 |
| Best epoch (lowest validation loss) | 25 |
| Training / validation loss at best epoch | 0.0540 / 0.0708 |
| Training / validation accuracy at best epoch | 98.45% / 97.92% |
| Training time and hardware | 40.6 s, CPU (Google Colab) |

## Table 3 – Learning-rate choice

| α | Lowest validation loss | Validation accuracy at that epoch |
|---|---|---|
| 0.01 | 0.1624 (epoch 30) | 95.75% |
| 0.1 | 0.0828 (epoch 30) | 97.60% |
| 0.5 | 0.0708 (epoch 25) | 97.92% |

## Table 4 – Test results

| | N | S | V | F | Q |
|---|---|---|---|---|---|
| Per-class accuracy | 99.23% | 71.76% | 93.02% | 68.52% | 96.83% |

| Metric | Value |
|---|---|
| Test accuracy | 97.72% |
| Balanced accuracy | 85.87% |
| Always-"N" accuracy | 82.76% |

### Test confusion matrix (rows = true class, columns = predicted class)

| true \ predicted | N | S | V | F | Q |
|---|---|---|---|---|---|
| **N** | 17978 | 87 | 37 | 5 | 11 |
| **S** | 142 | 399 | 14 | 0 | 1 |
| **V** | 77 | 4 | 1347 | 15 | 5 |
| **F** | 33 | 0 | 18 | 111 | 0 |
| **Q** | 37 | 5 | 9 | 0 | 1557 |

Total = 21,892.
