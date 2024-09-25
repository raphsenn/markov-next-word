#!/usr/bin/env python3
# Author: Raphael Senn

from src.markov_next_word import MarkovNextWord

def test_train() -> None:
    mnw = MarkovNextWord()
    mnw.train("data/test.txt")
    assert mnw.word_to_nextwords == {'i': ['like', 'like', 'hate', 'love', 'love'], 'like': ['math', 'physics'], 'hate': ['war'], 'love': ['schnitzel', 'science']}

def test_next_word_prediction() -> None:
    mnw = MarkovNextWord()
    mnw.train("data/test.txt")
    assert mnw.predict_next_words('I') == [('like', 0.4), ('love', 0.4), ('hate', 0.2)]
    assert mnw.predict_next_words('i') == [('like', 0.4), ('love', 0.4), ('hate', 0.2)]
    assert mnw.predict_next_words('hate') == [('war', 1.0)]