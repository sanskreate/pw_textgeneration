# src/utils.py
'''
import re

def keyword_coverage(text, keywords):
    present = [kw for kw in keywords if re.search(rf"\b{re.escape(kw)}\b", text)]
    return len(present) / len(keywords)

def check_length(text, min_words=750):
    return len(text.split()) >= min_words


def log_metrics(epoch, loss, coverage=None):
    with open("logs/train_log.txt", "a") as f:
        f.write(f"Epoch: {epoch}, Loss: {loss:.4f}, Coverage: {coverage}\n")

'''

from transformers import logging as hf_logging
import logging

def tokenize_batch(tokenizer, texts, max_len):
    return tokenizer(
        texts,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

def get_logger(name="TextGen"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

# Optional: suppress HF warnings
hf_logging.set_verbosity_error()
