# markov-next-word
Next word prediction using markov chains and python.

## What is a markov property?
In a process where the next state depends only on the current state is called markov property.
A sequence of events which follow the Markov property is referred to as the Markov Chain.

## Next word prediction using markov property

### Consider following example data

```console
I like Math.
I like Physics.
I hate War.
I love Schnitzel.
I love Science.
```

### Mapping words to next words.

```console
i -> [like, like, hate, love, love]
like -> [math, physics]
hate -> [war]
love -> [schnitzel, science]
```
#### Graph representation

![image](./res/graph.png)

### Mapping word and next_word to its probability.

```console
(i, like) -> 0.4
(i, hate) -> 0.2
(i, love) -> 0.4
(like, math) -> 0.5
(like, physics) -> 0.5
(hate, war) -> 1.0
(love, schnitzel) -> 0.5
(love, science) -> 0.5
```
#### Graph representation with probabilitys

![image](./res/graph_probs.png)

### Example using the model

```python
>>> from src/markov_next_word import MarkovNextWord
>>> mnw = MarkovNextWord()
>>> mnw.train('data/test.txt')
>>> mnw.word_to_nextwords
{'i': ['like', 'like', 'hate', 'love', 'love', 'love'], 'like': ['math', 'physics'], 'hate': ['war'], 'love': ['schnitzel', 'science']}
>>> mnw.word_to_next_word_prob
{('i', 'like'): 0.4, ('i', 'hate'): 0.2, ('i', 'love'): 0.4, ('like', 'math'): 0.5, ('like', 'physics'): 0.5, ('hate', 'war'): 1.0, ('love', 'schnitzel'): 0.5, ('love', 'science'): 0.5}
```