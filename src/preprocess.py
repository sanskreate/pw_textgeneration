import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_and_split(path="data/common_gen.csv"):
    df = pd.read_csv(path)
    train, val = train_test_split(df, test_size=0.1, random_state=42)
    train.to_csv("data/train.csv", index=False)
    val.to_csv("data/val.csv", index=False)
    print("✅ Train and validation CSVs created")

if __name__ == "__main__":
    preprocess_and_split()