#dùng environment: rnn-infer
import zipfile, json, shutil, os, tempfile

SRC = r"C:\Users\PC\CS221\output_RNN_LSTM\LSTM1D_model\lstm1d_hate_speech.keras"
DST = r"C:\Users\PC\CS221\output_RNN_LSTM\LSTM1D_model\lstm1d_hate_speech_patched_v2.keras"

def fix(obj):
    # Recursively patch config:
    # 1) batch_shape -> batch_input_shape
    # 2) dtype policy dict (DTypePolicy) -> dtype string (e.g. "float32")
    if isinstance(obj, dict):
        # batch_shape
        if "batch_shape" in obj and "batch_input_shape" not in obj:
            obj["batch_input_shape"] = obj.pop("batch_shape")

        # dtype policy
        if "dtype" in obj and isinstance(obj["dtype"], dict):
            d = obj["dtype"]
            if d.get("class_name") == "DTypePolicy":
                # prefer config.name if exists
                name = None
                cfg = d.get("config", {})
                if isinstance(cfg, dict):
                    name = cfg.get("name")
                obj["dtype"] = name or "float32"

        for k, v in list(obj.items()):
            fix(v)

    elif isinstance(obj, list):
        for x in obj:
            fix(x)

def main():
    if os.path.abspath(SRC) == os.path.abspath(DST):
        raise ValueError("DST must be different from SRC")

    tmpdir = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(SRC, "r") as z:
            z.extractall(tmpdir)

        cfg_path = os.path.join(tmpdir, "config.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError("config.json not found inside .keras")

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        fix(cfg)

        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f)

        if os.path.exists(DST):
            os.remove(DST)

        with zipfile.ZipFile(DST, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk(tmpdir):
                for fn in files:
                    full = os.path.join(root, fn)
                    rel = os.path.relpath(full, tmpdir)
                    z.write(full, rel)

        print("Patched saved to:", DST)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

if __name__ == "__main__":
    main()
