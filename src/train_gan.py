import torch
import torch.nn as nn
import torch.optim as optim
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from src.utils import get_logger, tokenize_batch

logger = get_logger("Train-GAN")

class KeywordTextDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len=64):
        data = pd.read_csv(csv_path)
        self.contexts = data["context"].tolist()
        self.texts = data["text"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        input_text = self.contexts[idx]
        target_text = self.texts[idx]

        input_enc = self.tokenizer(
            input_text, padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt"
        )
        target_enc = self.tokenizer(
            target_text, padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt"
        )

        return {
            "input_ids": input_enc.input_ids.squeeze(),
            "attention_mask": input_enc.attention_mask.squeeze(),
            "labels": target_enc.input_ids.squeeze()
        }

class DMKLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(DMKLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.alpha = alpha

    def forward(self, logits, labels, keyword_mask):
        ce_loss = self.ce(logits.view(-1, logits.size(-1)), labels.view(-1))

        # Keyword-focused loss (encourages keywords in output)
        keyword_logits = logits[keyword_mask]
        keyword_labels = labels[keyword_mask]
        if keyword_logits.nelement() == 0:
            keyword_loss = 0.0
        else:
            keyword_loss = self.ce(keyword_logits, keyword_labels)

        total_loss = self.alpha * ce_loss + (1 - self.alpha) * keyword_loss
        return total_loss

def train_gan_model(csv_path="data/common_gen.csv", epochs=3, batch_size=8, lr=5e-5, max_len=64):
    logger.info("Starting GAN training with DMK Loss...")

    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    dataset = KeywordTextDataset(csv_path, tokenizer, max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    dmk_loss = DMKLoss(alpha=0.7)

    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        total_loss = 0
        logger.info(f"🔥 Epoch {epoch + 1}/{epochs}")
    
        for i, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits

            # Dummy keyword mask
            keyword_mask = (labels != -100)

            loss = dmk_loss(logits, labels, keyword_mask)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            if i % 10 == 0:
                logger.info(f"🔁 Batch {i}/{len(loader)}")

        logger.info(f"📉 Epoch {epoch + 1} complete. Avg Loss: {total_loss / len(loader):.4f}")


    model.save_pretrained("models/gan_dmk_model")
    tokenizer.save_pretrained("models/gan_dmk_model")
    logger.info("✅ Model saved at models/gan_dmk_model")


if __name__ == "__main__":
    train_gan_model()
