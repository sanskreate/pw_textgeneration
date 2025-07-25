# src/generate.py

import os
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from src.dckg import DomainConstrainedKeywordGenerator
from src.utils import get_logger
from models.gan_dmk_model import TextGenerator

logger = get_logger(__name__)

def generate_text(context_sentences, t5_model_path, gan_model_path):
    logger.info("Loading T5 model and tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_path)
    t5_model.eval()

    logger.info("Generating keywords using DCKG...")
    context_input = " ".join(context_sentences)
    keywords = generate_keywords(context_input)  # Returns a list of keywords

    logger.info(f"Generated Keywords: {keywords}")

    # Prepare GAN model
    logger.info("Loading GAN generator model...")
    gan_generator = TextGenerator()
    gan_generator.load_state_dict(torch.load(gan_model_path, map_location=torch.device('cpu')))
    gan_generator.eval()

    # Prepare T5 input
    keyword_string = ", ".join(keywords)
    input_text = f"generate: {context_input} | keywords: {keyword_string}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True)

    logger.info("Generating text using T5...")
    summary_ids = t5_model.generate(input_ids, max_length=512, num_beams=4, early_stopping=True)
    t5_output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    logger.info("Enhancing output using GAN...")
    gan_input = torch.tensor(tokenizer.encode(t5_output, max_length=512, truncation=True)).unsqueeze(0)
    with torch.no_grad():
        enhanced_output = gan_generator(gan_input)

    final_output = tokenizer.decode(enhanced_output[0], skip_special_tokens=True)

    logger.info("Text generation complete.")
    return final_output


if __name__ == "__main__":
    context = [
        "The sun was setting over the ocean.",
        "Waves crashed softly on the shore."
    ]
    t5_model_dir = "models/t5_baseline/"
    gan_model_file = "models/gan_dmk_model/generator.pth"

    output = generate_text(context, t5_model_dir, gan_model_file)
    print("\n--- Generated Output ---\n")
    print(output)