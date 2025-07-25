import logging
import random
from transformers import pipeline

from src.utils import get_logger

logger = get_logger("DCKG")

class DomainConstrainedKeywordGenerator:
    def __init__(self, domain_keywords=None):
        """
        Initializes the keyword generator with optional domain-specific keywords.
        """
        self.domain_keywords = domain_keywords or ["education", "learning", "AI", "content", "students"]
        self.keyword_extractor = pipeline("ner", grouped_entities=True, model="dslim/bert-base-NER")

        logger.info("Initialized DomainConstrainedKeywordGenerator.")

    def extract_keywords(self, context):
        """
        Extracts entity-based keywords from the context using a pre-trained NER model.
        """
        logger.debug(f"Extracting keywords from context: {context}")
        ner_results = self.keyword_extractor(context)
        extracted = list(set(entity['word'].strip() for entity in ner_results if entity['entity_group'] in ['ORG', 'MISC', 'PER', 'LOC']))
        logger.info(f"Extracted keywords: {extracted}")
        return extracted

    def diversify_keywords(self, base_keywords, top_k=5):
        """
        Combines extracted keywords with domain-specific keywords for diversity.
        """
        logger.debug(f"Diversifying keywords: {base_keywords}")
        diversified = list(set(base_keywords + random.sample(self.domain_keywords, k=min(top_k, len(self.domain_keywords)))))
        logger.info(f"Diversified keywords: {diversified}")
        return diversified

    def generate_keywords(self, context):
        """
        End-to-end generation of domain-constrained keywords from context.
        """
        logger.info("Generating domain-constrained keywords...")
        base_keywords = self.extract_keywords(context)
        final_keywords = self.diversify_keywords(base_keywords)
        return final_keywords


if __name__ == "__main__":
    # Example usage
    sample_context = "Artificial Intelligence is transforming online education platforms like Coursera and Udemy by providing personalized learning experiences."
    dckg = DomainConstrainedKeywordGenerator()
    keywords = dckg.generate_keywords(sample_context)
    print("Generated Keywords:", keywords)
