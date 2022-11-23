# question-score
Repository for KDA(Knowledge-dependent Answerability), EMNLP 2022 work

# How to use

```
pip install --upgrade pip
pip install question-score
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

# What does the score mean?

You can check the explanation of KDA on https://arxiv.org/abs/2211.11902 now.
The official link from EMNLP 2022 will soon be released.

You can use $KDA_{small}$ or $KDA_{large}$ instead of heavy metric using all model.
Below is the performance of the submetrics, which mentioned on the appendix of the paper.

| Sub Metric | Model Count ( Total Size ) | KDA (Valid) | Likert (Test) |
|:----------:|:--------------------------:|:-----------:|:-------------:|
|  KDA_small |          4 (3.5GB)         |    0.740    |     0.377     |
|  KDA_large |         10 (19.2GB)        |    0.784    |     0.421     |


# Citation
```
@inproceedings{moon2022eval,
  title={Evaluating the Knowledge Dependency of Questions,
  author={Moon, H., Yang, Y., Shin, J., Yu, H., Lee, S., Jeong, M., Park, J., Kim, M., & Choi, S.},
  booktitle={Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing},
  year={2022}
}
```
