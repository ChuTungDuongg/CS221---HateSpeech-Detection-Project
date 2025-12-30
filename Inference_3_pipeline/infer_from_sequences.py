#dùng environment: rnn-infer

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import pad_sequences
import json


X_NPY = r"C:\Users\PC\CS221\X_rnn.npy"
RNN_MODEL_KERAS = r"C:\Users\PC\CS221\output_RNN_LSTM\LSTM1D_model\lstm1d_hate_speech_patched_v2.keras"
OUT_CSV = r"C:\Users\PC\CS221\rnn_preds.csv"
CFG_JSON = r"C:\Users\PC\CS221\output_RNN_LSTM\LSTM1D_model\config_only.json"
WEIGHTS_H5 = r"C:\Users\PC\CS221\output_RNN_LSTM\LSTM1D_model\weights.h5"

with open(CFG_JSON, "r", encoding="utf-8") as f:
    cfg = json.load(f)

ID2LABEL_RNN = {0:"Hate", 1:"Offensive", 2:"Neither"}  # sửa nếu cần

def main():
    seqs = np.load(X_NPY, allow_pickle=True).tolist()

    model = tf.keras.models.model_from_config(cfg)
    model.load_weights(WEIGHTS_H5, skip_mismatch=True, by_name=True)

    max_len = model.input_shape[1]

    X = pad_sequences(seqs, maxlen=max_len, padding="post", truncating="post")
    probs = model.predict(X, verbose=0)
    pred_ids = probs.argmax(axis=1)
    preds = [ID2LABEL_RNN[int(i)] for i in pred_ids]

    pd.DataFrame({"pred_RNN": preds}).to_csv(OUT_CSV, index=False)
    print("Saved:", OUT_CSV)

if __name__ == "__main__":
    main()
