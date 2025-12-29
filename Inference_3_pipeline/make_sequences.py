#dùng environment: tok-encode

import os, re, pickle, json
import numpy as np
import pandas as pd

TEST_FILE = r"C:\Users\PC\CS221\Inference_3_pipeline\100_samples.xlsx"
TEXT_COL  = "Tweet"

TOKENIZER_PKL = r"C:\Users\PC\CS221\output_RNN_LSTM\LSTM1D_model\tokenizer_lstm1d.pkl"
OUT_X_NPY     = r"C:\Users\PC\CS221\X_rnn.npy"

def basic_clean(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"http\S+|www\.\S+", " <URL> ", s)
    s = re.sub(r"@\w+", " <USER> ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def main():
    # load test
    if TEST_FILE.lower().endswith(".xlsx"):
        df = pd.read_excel(TEST_FILE)
    else:
        df = pd.read_csv(TEST_FILE)
    texts = df[TEXT_COL].astype(str).fillna("").tolist()

    # load tokenizer (needs keras 3)
    with open(TOKENIZER_PKL, "rb") as f:
        tok = pickle.load(f)

    cleaned = [basic_clean(t) for t in texts]
    seqs = tok.texts_to_sequences(cleaned)

    # LƯU sequences dạng ragged (list of lists) -> npy object
    np.save(OUT_X_NPY, np.array(seqs, dtype=object), allow_pickle=True)
    print("Saved sequences:", OUT_X_NPY, "num_samples=", len(seqs))

if __name__ == "__main__":
    main()
