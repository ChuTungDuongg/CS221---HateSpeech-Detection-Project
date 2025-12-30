import json, pickle, re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# nếu máy chưa có nltk data thì bật 1 lần
# nltk.download("stopwords")
# nltk.download("punkt")

stop_words = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    filtered = [w for w in tokens if w.isalpha() and w not in stop_words]
    return " ".join(filtered)

class LSTMInfer:
    def __init__(self, save_dir: str):
        self.model = load_model(f"{save_dir}/lstm1d_hate_speech.keras")

        with open(f"{save_dir}/tokenizer_lstm1d.pkl", "rb") as f:
            self.tokenizer = pickle.load(f)

        with open(f"{save_dir}/config_lstm1d.json", "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.max_len = int(self.config["MAX_LEN"])
        self.label_map = self.config["LABEL_MAPPING"]

    def encode(self, texts):
        seq = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(seq, maxlen=self.max_len, padding="post", truncating="post")

    def predict_one(self, text: str) -> str:
        x = self.encode([clean_text(text)])
        probs = self.model.predict(x, verbose=0)
        pred_id = int(probs.argmax(axis=1)[0])
        return self.label_map[str(pred_id)]

if __name__ == "__main__":
    # đổi SAVE_DIR đúng với nơi bạn lưu 3 file
    SAVE_DIR = r"C:\Users\PC\CS221\output_RNN_LSTM\LSTM1D_model"

    infer = LSTMInfer(SAVE_DIR)

    text = "you are a stupid, an idiot, a cow in a small circle"
    print(infer.predict_one(text))
