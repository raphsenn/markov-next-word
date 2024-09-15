#!/usr/bin/env python3
# Author: Raphael Senn
# This is really simple tokenization (could be ordered on wish :D), but it works for this simple language model.

import re

# Word pattern.
WORD_PATTERN = r"[A-Za-z0-9]+"

# Really simple word (including numbers) tokenizer.
def tokenize_words(text: str) -> list[str]:
    return re.findall(WORD_PATTERN, text.lower())