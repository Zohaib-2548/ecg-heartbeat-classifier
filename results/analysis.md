# Task 6 – Analysis

## 6.1 Always answering "N"

A model that always predicts N gets 18,118 / 21,892 = **82.76%** test accuracy without learning anything. Our network reaches **97.72%**, so the real gain over this trivial baseline is about 15 points. Balanced accuracy is only **85.87%**, 12 points below overall accuracy, because the rare classes S (71.8%) and F (68.5%) are recognised much worse than N (99.2%). Overall accuracy is dominated by the common class N, so it hides how the model treats rare beats. On imbalanced data accuracy alone is not the right number to trust; per-class accuracy and balanced accuracy must be reported next to it.

## 6.2 Weakest class

The lowest per-class accuracy is **F (fusion beat), 68.52%**: 111 of 162 test beats were recognised. Of the 51 errors, 33 were predicted as **N** and 18 as V. F is therefore most often mistaken for N. A fusion beat is produced when a normal beat and a ventricular beat occur at almost the same moment, so its shape lies between an N beat and a V beat and can look very similar to a normal one. In addition, F has only 641 training rows against 72,471 for N, so the network sees few examples of it and the loss is dominated by N.

## 6.3 Loss curves

For the chosen α = 0.5, at the best epoch (25) the training loss is 0.054 and the validation loss is 0.071, with accuracies of 98.45% and 97.92%. The gap is small, and both losses fell strongly during training and both accuracies are high, so the network is **not underfitting**. After about epoch 25 the training loss stays around 0.05 while the validation loss fluctuates (0.0786, 0.0741, 0.0766 in epochs 27–29) instead of falling further, which is a mild sign that overfitting is starting. Overall the network is neither clearly overfitting nor underfitting; this is why we saved the parameters from the epoch with the lowest validation loss. The noise from epoch to epoch comes from the large learning rate.

## 6.4 Initialisation

If all weights were zero, every neuron in a layer would compute the same value and receive the same gradient, so all neurons would stay identical forever (the symmetry is never broken). With ReLU and zero weights the hidden activations are zero, so the gradients reaching the earlier layers are zero as well and those layers do not learn. Random initial values break this symmetry.

For a ReLU layer, z = Σ w_j a_j over n inputs gives Var(z) = n · Var(w) · E[a²]. ReLU sets about half of the values to zero, so E[a²] is about half of the variance of the previous layer's z. To keep the variance of the signal the same from layer to layer (so activations and gradients neither vanish nor explode), we need Var(w) = 2/n, i.e. standard deviation √(2/n). That is the He rule.

## 6.5 Momentum and weight decay

The velocity is v = βv + (1−β)g. When two consecutive mini-batch gradients point in opposite directions, they partly cancel inside v, so the update becomes small and the zig-zag is damped. When the gradients keep the same sign, they add up and the steps become larger in that direction.

For a weight that receives no gradient, the update uses g = λP, so the weight is multiplied by roughly (1 − αλ) at every step and decays exponentially towards zero.

## 6.6 Helping rare classes

One option is to use a **class-weighted loss**: multiply the loss of each example by a weight that is larger for rare classes (for example inversely proportional to the class frequency). A mistake on an F beat then costs the network much more, so the gradient pays attention to F instead of being dominated by the 72,000 N beats. Another option on the data side is to oversample the F beats (or create slightly modified copies of them) so that the network sees them more often during training.
