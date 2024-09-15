#!/usr/bin/env python3

from src.markov_next_word import MarkovNextWord


def test_next_word_prediction():
    mnw = MarkovNextWord()
    mnw.train("tests/test.txt")
    assert mnw.predict_next_words('I') == [('like', 0.6666666666666666), ('love', 0.3333333333333333)]
    assert mnw.predict_next_words('i') == [('like', 0.6666666666666666), ('love', 0.3333333333333333)]
    assert mnw.predict_next_words('like') == [('photography', 0.5), ('science', 0.5)]
    assert mnw.predict_next_words('love') == [('mathematics', 1.0)]