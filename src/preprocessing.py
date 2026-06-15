"""
Text Preprocessing Utilities for Sentiment Analysis
====================================================
This module contains helper functions for text cleaning and preprocessing.
"""

import re
from typing import List


def clean_html(text: str) -> str:
    """Remove HTML tags from text."""
    return re.sub(r'<[^>]+>', '', text)


def clean_urls(text: str) -> str:
    """Remove URLs from text."""
    return re.sub(r'http\S+|www\S+', '', text)


def clean_special_chars(text: str) -> str:
    """Remove special characters, keeping only alphanumeric and spaces."""
    return re.sub(r'[^a-zA-Z\s]', '', text)


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text."""
    return re.sub(r'\s+', ' ', text).strip()


def preprocess_text(text: str) -> str:
    """
    Full preprocessing pipeline for baseline model.
    
    Steps:
    1. Remove HTML tags
    2. Remove URLs
    3. Remove special characters
    4. Convert to lowercase
    5. Normalize whitespace
    
    Args:
        text: Raw review text
        
    Returns:
        Cleaned text string
    """
    text = clean_html(text)
    text = clean_urls(text)
    text = clean_special_chars(text)
    text = text.lower()
    text = normalize_whitespace(text)
    return text


def preprocess_batch(texts: List[str]) -> List[str]:
    """Apply preprocessing to a batch of texts."""
    return [preprocess_text(text) for text in texts]


def get_text_stats(text: str) -> dict:
    """Get basic statistics for a text."""
    words = text.split()
    return {
        'char_count': len(text),
        'word_count': len(words),
        'avg_word_length': sum(len(w) for w in words) / max(len(words), 1),
        'sentence_count': text.count('.') + text.count('!') + text.count('?')
    }


if __name__ == "__main__":
    # Test preprocessing
    sample = "<br/>This is a <b>test</b> review! Visit http://example.com for more. It's great!!!"
    print(f"Original: {sample}")
    print(f"Cleaned: {preprocess_text(sample)}")
    print(f"Stats: {get_text_stats(sample)}")
