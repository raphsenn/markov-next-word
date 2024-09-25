#!/usr/bin/env python3
# Author: Raphael Senn

from src.markov_next_word import MarkovNextWord

def test_train() -> None:
    mnw = MarkovNextWord()
    mnw.train("data/test.txt")
    assert mnw.word_to_nextwords == {'i': ['like', 'like', 'hate', 'love', 'love', 'love'], 'like': ['math', 'physics'], 'hate': ['war'], 'love': ['schnitzel', 'science', 'water']}

def test_next_word_prediction() -> None:
    mnw = MarkovNextWord()
    mnw.train("data/test.txt")
    assert mnw.predict_next_words('I') == [('love', 0.5), ('like', 0.3333333333333333), ('hate', 0.16666666666666666)]
    assert mnw.predict_next_words('i') == [('love', 0.5), ('like', 0.3333333333333333), ('hate', 0.16666666666666666)]
    assert mnw.predict_next_words('hate') == [('war', 1.0)]