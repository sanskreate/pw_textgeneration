from datasets import load_dataset
import pandas as pd
import os

def load_and_save_common_gen(path="data/common_gen.csv", split='train', limit=5000):
    dataset = load_dataset("allenai/common_gen")
    data = dataset[split]

    rows = []
    for i, sample in enumerate(data):
        if limit and i >= limit:
            break
        context = " ".join(sample['concepts'])
        target = sample['target']
        rows.append({"context": context, "text": target})

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"✅ Saved {len(df)} rows to {path}")
    return df

if __name__ == "__main__":
    load_and_save_common_gen()
