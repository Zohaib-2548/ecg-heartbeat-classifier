# ECG Heartbeat Classification (MLP from scratch in PyTorch)

Fully connected network (187 -> 128 -> 64 -> 5) built from raw weight matrices.
Manual softmax cross-entropy, manual SGD with momentum and L2 weight decay.

## Setup
```bash
pip install -r requirements.txt
```
Download `mitbih_train.csv` and `mitbih_test.csv` from
https://www.kaggle.com/datasets/shayanfazeli/heartbeat and put them in `data/`.

## Run (from the repository root, in this order)
```bash
python src/load_data.py      # check files load
python src/prepare.py        # Task 1: shapes, counts, split, example-beats figure
python src/model.py          # Task 2: parameter count (32,645)
python src/grad_check.py     # Task 3: hand gradient vs autograd
python src/train.py          # Task 4: 3 learning rates, curves, saved model
python src/test_model.py     # Task 5: test accuracy, confusion matrix
streamlit run app/app.py     # Task 7: web interface
```
All figures, tables and the saved model go to `results/`.

## Front end screenshot
(add screenshot here)

## Results
(fill in Tables 1-4 from `results/`)
