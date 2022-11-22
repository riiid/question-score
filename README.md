# question-score
Repository for KDA(Knowledge-dependent Answerability), EMNLP 2022 work

# How to use

```
pip install -e .
```

```python
from question_score import KDA
kda = KDA()
print(
  kda.kda_small(
    "passage",
    "question",
    ["option1", "option2", "option3", "option4"],
    1
  )
)
```
you can use KDA_small or KDA_large instead of heavy metric using all model. The performances of metrics are presented in the appendix of the paper.
