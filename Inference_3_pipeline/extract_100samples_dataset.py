import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 1) Read dataset
df = pd.read_csv(r"C:\Users\PC\CS221\NoteBook_ML\labeled_data.csv")

# 2) Split 10% test (stratify theo cột nhãn "class")
_, test_df = train_test_split(
    df,
    test_size=0.10,
    random_state=42,
    stratify=df["class"]
)

# 3) Chỉ giữ 2 cột cần thiết + đổi tên cột
test_df = test_df[["tweet", "class"]].rename(
    columns={"tweet": "Tweet", "class": "Final Votes"}
)

# 4) Lấy 100 mẫu theo đúng tỉ lệ lớp trong test
N = 100
vc = test_df["Final Votes"].value_counts(normalize=True)

raw = vc * N
base = np.floor(raw).astype(int)
remainder = N - base.sum()

frac = (raw - base).sort_values(ascending=False)
for cls in frac.index[:remainder]:
    base.loc[cls] += 1

parts = []
for cls, n_cls in base.items():
    parts.append(
        test_df[test_df["Final Votes"] == cls].sample(n=n_cls, random_state=42)
    )

test_100 = (
    pd.concat(parts, ignore_index=True)
      .sample(frac=1, random_state=42)
      .reset_index(drop=True)
)

# 5) Đổi nhãn số -> text
label_map = {0: "Hate", 1: "Offensive", 2: "Neither"}
test_100["Final Votes"] = test_100["Final Votes"].map(label_map)

# (optional) check có giá trị nào map bị NaN không
assert test_100["Final Votes"].isna().sum() == 0, "Có nhãn không nằm trong {0,1,2}"

# 6) Export
test_100.to_csv("test_100_stratified_from_test.csv", index=False, encoding="utf-8-sig")

print("Saved: test_100_stratified_from_test.csv")
print(test_100["Final Votes"].value_counts())
