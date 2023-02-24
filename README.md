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
@inproceedings{moon-etal-2022-evaluating,
    title = "Evaluating the Knowledge Dependency of Questions",
    author = "Moon, Hyeongdon  and
      Yang, Yoonseok  and
      Yu, Hangyeol  and
      Lee, Seunghyun  and
      Jeong, Myeongho  and
      Park, Juneyoung  and
      Shin, Jamin  and
      Kim, Minsam  and
      Choi, Seungtaek",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.718",
    pages = "10512--10526",
    abstract = "The automatic generation of Multiple Choice Questions (MCQ) has the potential to reduce the time educators spend on student assessment significantly. However, existing evaluation metrics for MCQ generation, such as BLEU, ROUGE, and METEOR, focus on the n-gram based similarity of the generated MCQ to the gold sample in the dataset and disregard their educational value.They fail to evaluate the MCQ{'}s ability to assess the student{'}s knowledge of the corresponding target fact. To tackle this issue, we propose a novel automatic evaluation metric, coined Knowledge Dependent Answerability (KDA), which measures the MCQ{'}s answerability given knowledge of the target fact. Specifically, we first show how to measure KDA based on student responses from a human survey.Then, we propose two automatic evaluation metrics, KDA{\_}disc and KDA{\_}cont, that approximate KDA by leveraging pre-trained language models to imitate students{'} problem-solving behavior.Through our human studies, we show that KDA{\_}disc and KDA{\_}soft have strong correlations with both (1) KDA and (2) usability in an actual classroom setting, labeled by experts. Furthermore, when combined with n-gram based similarity metrics, KDA{\_}disc and KDA{\_}cont are shown to have a strong predictive power for various expert-labeled MCQ quality measures.",
}
```
