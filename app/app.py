"""Task 7.1: Streamlit front end.  Run from repo root:  streamlit run app/app.py"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import streamlit as st
import torch
import matplotlib.pyplot as plt
from model import ECGNet

NAMES = ["N: normal", "S: supraventricular ectopic", "V: ventricular ectopic",
         "F: fusion", "Q: unknown / paced"]

@st.cache_resource
def load_model():
    m = ECGNet()
    m.load_state_dict(torch.load(ROOT / "results" / "best_model.pt"))
    m.eval()
    return m

@st.cache_data
def load_test():
    return pd.read_csv(ROOT / "data" / "mitbih_test.csv", header=None)

st.title("ECG heartbeat classifier")
model = load_model()
mode = st.radio("Input", ["Test-set row", "Upload CSV (one row, 187 values)"])
x, true_label = None, None
if mode == "Test-set row":
    df = load_test()
    i = st.number_input("Row number", 0, len(df) - 1, 0)
    x = df.iloc[int(i), :187].values.astype(np.float32)
    true_label = int(df.iloc[int(i), 187])
else:
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        vals = pd.read_csv(up, header=None).values.flatten()[:187]
        if len(vals) == 187:
            x = vals.astype(np.float32)
        else:
            st.error("Need exactly 187 values.")

if x is not None:
    with torch.no_grad():
        probs = torch.softmax(model(torch.tensor(x).unsqueeze(0)), dim=1)[0].numpy()
    pred = int(probs.argmax())
    fig, ax = plt.subplots(figsize=(7, 2.5))
    ax.plot([1000 * k / 125 for k in range(187)], x)
    ax.set_xlabel("time (ms)"); ax.set_ylabel("amplitude")
    st.pyplot(fig)
    st.subheader(f"Predicted class: {NAMES[pred]}")
    if true_label is not None:
        st.write(f"True class: {NAMES[true_label]}")
    fig2, ax2 = plt.subplots(figsize=(7, 3))
    ax2.bar([n.split(":")[0] for n in NAMES], probs)
    ax2.set_ylabel("probability"); ax2.set_ylim(0, 1)
    st.pyplot(fig2)
