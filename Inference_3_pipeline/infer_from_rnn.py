import os
import json
import pickle
import re

import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ====== PATHS ======
IN_CSV  = r"C:\Users\PC\CS221\Inference_3_pipeline\test_100_stratified_from_test.csv"
OUT_CSV = r"C:\Users\PC\CS221\rnn_preds.csv"

SAVE_DIR = r"C:\Users\PC\CS221\output_RNN_LSTM\LSTM1D_model"
MODEL_PATH = os.path.join(SAVE_DIR, "lstm1d_hate_speech.keras")  # đổi nếu tên khác
TOK_PATH   = os.path.join(SAVE_DIR, "tokenizer_lstm1d.pkl")
CFG_PATH   = os.path.join(SAVE_DIR, "config_lstm1d.json")

# ====== CSV COLUMN NAMES (SỬA Ở ĐÂY NẾU KHÁC) ======
TEXT_COL  = "Tweet"   # hoặc "text", "sentence", ...
LABEL_COL = "Final Votes"   # nếu file có nhãn; nếu không có thì để None

# ====== NLTK (bật 1 lần nếu thiếu) ======
# nltk.download("stopwords")
# nltk.download("punkt")

stop_words = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    text = text or ""
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    filtered = [w for w in tokens if w.isalpha() and w not in stop_words]
    return " ".join(filtered)

class LSTMInfer:
    def __init__(self, save_dir: str):
        self.model = load_model(MODEL_PATH, compile=False)

        with open(TOK_PATH, "rb") as f:
            self.tokenizer = pickle.load(f)

        with open(CFG_PATH, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        # MAX_LEN trong config (ưu tiên), fallback theo model.input_shape
        self.max_len = int(self.config.get("MAX_LEN", self.model.input_shape[1]))
        # LABEL_MAPPING trong config: ví dụ {"0":"Hate","1":"Offensive","2":"Neither"}
        self.label_map = self.config.get("LABEL_MAPPING", {0: "Hate", 1: "Offensive", 2: "Neither"})

    def encode(self, texts):
        seq = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(seq, maxlen=self.max_len, padding="post", truncating="post")

    def predict_batch(self, texts):
        cleaned = [clean_text(t) for t in texts]
        X = self.encode(cleaned)
        probs = self.model.predict(X, verbose=0)
        pred_ids = probs.argmax(axis=1).astype(int)
        preds = [self.label_map[str(i)] if str(i) in self.label_map else self.label_map.get(i, str(i)) for i in pred_ids]
        conf = probs.max(axis=1)  # confidence = max softmax prob
        return pred_ids, preds, conf

def main():
    df = pd.read_csv(IN_CSV)

    if TEXT_COL not in df.columns:
        raise ValueError(f"Không thấy cột text '{TEXT_COL}' trong CSV. Columns hiện có: {list(df.columns)}")

    texts = df[TEXT_COL].astype(str).tolist()

    infer = LSTMInfer(SAVE_DIR)
    pred_ids, preds, conf = infer.predict_batch(texts)

    # Xuất file: giữ lại text + pred + pred_id + confidence
    out_df = pd.DataFrame({
        TEXT_COL: df[TEXT_COL],
        "pred_id": pred_ids,
        "pred_RNN": preds,
        "confidence": conf
    })

    # Nếu có label thì thêm để bạn dễ đối chiếu (không bắt buộc)
    if LABEL_COL and LABEL_COL in df.columns:
        out_df[LABEL_COL] = df[LABEL_COL]

    out_df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print("Saved:", OUT_CSV)

if __name__ == "__main__":
    main()
