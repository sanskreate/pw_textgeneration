import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.optim import AdamW
import pandas as pd
import os
from src.utils import get_logger

logger = get_logger("TrainT5")

class CommonGenDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.data = pd.read_csv(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        inputs = self.tokenizer(row['context'], return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length)
        targets = self.tokenizer(row['text'], return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length)
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        labels = targets['input_ids'].squeeze()
        labels[labels == tokenizer.pad_token_id] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def train(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    for i, batch in enumerate(dataloader):
        for k in batch:
            batch[k] = batch[k].to(device)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

        if i % 10 == 0:
            print(f"Step {i}, Batch Loss: {loss.item():.4f}")
    avg_loss = total_loss / len(dataloader)
    print(f"✅ Training step completed. Avg Loss: {avg_loss:.4f}")
'''
    model.train()
    for batch in dataloader:
        for k in batch:
            batch[k] = batch[k].to(device)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"✅ Training step completed with loss {loss.item():.4f}")
'''


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)

    train_dataset = CommonGenDataset("data/train.csv", tokenizer, max_length=64)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(3):
        print(f"📦 Epoch {epoch + 1}")
        train(model, train_loader, optimizer)

    model.save_pretrained("models/t5_baseline")
    tokenizer.save_pretrained("models/t5_baseline")



'''
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataset = CommonGenDataset("data/train.csv", tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(3):
        print(f"Epoch {epoch + 1}")
        train(model, train_loader, optimizer)

    model.save_pretrained("models/t5_baseline")
    tokenizer.save_pretrained("models/t5_baseline")
'''