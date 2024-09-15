# Author: Raphael Senn
# This is really simple tokenization (could be ordered on wish :D), but it works for this simple language model.

import re

# Word pattern.
WORD_PATTERN = r"[A-Za-z0-9]+"

# Really simple word (including numbers) tokenizer.
def tokenize_words(text: str) -> list[str]:
    return re.findall(WORD_PATTERN, text.lower())

# Really simple character (including numbers) tokenizer.
def tokenize_chars(text: str) -> list[str]:
    text_words = tokenize_words(text)
    text_chars = [char for word in text_words for char in word] 
    return text_chars

# Computes vocabulary of entire text file.
def compute_vocabulary(text_file: str, word_token: bool = True) -> dict[str, int]:
    text = open(text_file, 'r')
    vocab: dict[str, int] = {}
    num_unique_words: int = 0
    for line in text:
        if word_token:
            for token in tokenize_words(line):
                vocab[token] = num_unique_words
                num_unique_words += 1
        else:
            for token in tokenize_chars(line):
                vocab[token] = num_unique_words
                num_unique_words += 1
    return vocab