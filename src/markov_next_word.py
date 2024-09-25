#!/usr/bin/env python3
# Author: Raphael Senn

import numpy as np
from src.utils import tokenize_words


class MarkovNextWord:
    def __init__(self) -> None:
        self.word_to_nextwords: dict[str, list[str]] = {}
        self.word_to_next_word_prob: dict[tuple[str, str]: float] = {}

    def train(self, file: str) -> None:  
        # Tokenization and fetch each word to all next words.
        with open(file, 'r') as text_train:
            for line in text_train:
                words = tokenize_words(line)
                for i in range(len(words) - 1):
                    word, next_word = words[i], words[i+1]
                    if word not in self.word_to_nextwords:
                        self.word_to_nextwords[word] = []
                    self.word_to_nextwords[word].append(next_word)
        
        # Calculate probabilitys from word to next_word. 
        for word in self.word_to_nextwords:
            for next_word in self.word_to_nextwords[word]:
                num_word_to_next_words = len(self.word_to_nextwords[word]) 
                if (word, next_word) not in self.word_to_next_word_prob:
                    self.word_to_next_word_prob[(word, next_word)] = 1.0 / num_word_to_next_words
                else: self.word_to_next_word_prob[(word, next_word)] += 1.0 / num_word_to_next_words

    def predict_next_words(self, word: str, top_k: int=5) -> list[tuple[str, float]]:
        result = [] 
        word = word.lower() 
        if word in self.word_to_nextwords:
            next_words = list(set(self.word_to_nextwords[word]))
            for next_word in next_words:
                    result.append((next_word, self.word_to_next_word_prob[(word, next_word)]))
            result = sorted(result, key=lambda x: x[1], reverse=True)
        return result[:top_k]
    
    def generate_text(self, input_text: str, seq_len: int=50, temperature: float=1.0, top_k: int=5) -> None:
        input_text_tokenized = tokenize_words(input_text)
        last_word = input_text_tokenized[-1]
        print(" ".join(input_text_tokenized), end=' ')
        
        for _ in range(seq_len):
            next_words = self.predict_next_words(last_word, top_k)
            if len(next_words) > 0:
                words, probs = zip(*next_words)

                # Temperatur scaling.
                scaled_props = np.array(probs) ** (1.0 / temperature)
                scaled_props /= np.sum(scaled_props) 
                last_word = np.random.choice(words, p=scaled_props)
                print(last_word, end=' ')
            else:
                break