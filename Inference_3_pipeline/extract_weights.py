#dùng environment: rnn-infer

import zipfile, os, shutil, tempfile

SRC = r"C:\Users\PC\CS221\output_RNN_LSTM\LSTM1D_model\lstm1d_hate_speech_patched_v2.keras"
DST = r"C:\Users\PC\CS221\output_RNN_LSTM\LSTM1D_model\weights.h5"

def main():
    tmp = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(SRC, "r") as z:
            z.extractall(tmp)

        # Keras zip thường có weights nằm ở một trong các tên sau
        candidates = []
        for root, _, files in os.walk(tmp):
            for fn in files:
                if fn.endswith(".h5") or fn.endswith(".weights.h5"):
                    candidates.append(os.path.join(root, fn))

        if not candidates:
            raise FileNotFoundError("No .h5 weights found inside the .keras archive")

        # ưu tiên file tên variables.h5 hoặc model.weights.h5
        candidates_sorted = sorted(
            candidates,
            key=lambda p: (
                0 if os.path.basename(p) in ["variables.h5", "model.weights.h5"] else 1,
                len(p)
            )
        )
        src_w = candidates_sorted[0]
        shutil.copyfile(src_w, DST)
        print("Copied weights:", src_w, "->", DST)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

if __name__ == "__main__":
    main()
