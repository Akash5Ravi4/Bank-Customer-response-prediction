# Bank Customer Response Prediction

Predicting which bank clients are likely to subscribe to a term deposit after a marketing call, so the bank can call the most promising clients first and waste fewer calls.

> The target is "did the client subscribe", not "did the client leave", so this is a response/retention propensity problem rather than literal churn. The ML workflow (classification, imbalanced target, model comparison, explanation) is the same.

## The problem
Telemarketing calls cost time and money, and only about 11% of clients say yes. If the bank calls everyone, roughly 9 in 10 calls are wasted. This project ranks clients by their chance of subscribing so the bank can focus on the best ones.

## Data
[UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank), file `bank-additional-full.csv`: 41,188 phone calls from a Portuguese bank (2008 to 2010), 20 input columns, target `y` (subscribed yes/no). The data is not included in this repo; download it from the link above and update the path in the first cell of the notebook.

## Data cleaning decisions
- Dropped **12 duplicate rows** before splitting, so no row can land in both train and test.
- Dropped **`duration`** (call length). It is only known after the call, so using it would be data leakage.
- **`pdays = 999`** means "never contacted before", not 999 days. Replaced it with a missing value and added a `previously_contacted` flag.
- Kept **"unknown"** as its own category instead of guessing a value, since it may carry signal (`default` is 21% unknown).

## Method
- Stratified 80/20 train/test split (the 11.3% yes-rate is the same in both parts).
- `sklearn` `Pipeline`: median imputation and scaling for numeric columns, one-hot encoding for categories. Everything is fitted on training data only.
- Compared logistic regression, random forest and gradient boosting (`HistGradientBoostingClassifier`) with stratified 5-fold cross-validation, using `class_weight="balanced"` for the imbalance.
- The test set was used once, for the final model.

## Results
| Model | CV ROC-AUC | CV PR-AUC |
|---|---|---|
| Logistic regression | 0.791 | 0.446 |
| Random forest | 0.795 | 0.466 |
| Gradient boosting (final) | 0.799 | 0.466 |

**Test set (gradient boosting):** ROC-AUC **0.813**, PR-AUC **0.488**. A random model scores about 0.113 on PR-AUC, so the model is roughly 4 times better than guessing. The three models are within 0.01 ROC-AUC of each other, so most of the signal is in the data and a more complex model adds little.

### Business impact
Test clients ranked by predicted chance of subscribing:

| Clients called | Subscribers captured | Lift vs. random calling |
|---|---|---|
| Top 10% | 45% | 4.5x |
| Top 20% | 66% | 3.3x |
| Top 30% | 75% | 2.5x |
| Top 50% | 84% | 1.7x |

Calling only the top 20% reaches about two thirds of all subscribers with 80% fewer calls. In that group, 37% of clients actually subscribed, compared with 11% overall.

![Gains chart](gain_chart.png)

## What drives a "yes"
Two methods were used on the final model.

- **Permutation importance** (drop in test ROC-AUC when a column is shuffled): `nr.employed` 0.090, `contact` 0.024, `euribor3m` 0.021, `cons.price.idx` 0.016, `cons.conf.idx` 0.014, then `previously_contacted`, `campaign`, `month`.
- **SHAP** (average push on the predicted probability, summed by original column): `nr.employed` 0.069, `cons.price.idx` 0.029, `euribor3m` 0.027, `contact` 0.026, `cons.conf.idx` 0.024, then `age`, `campaign`, `previously_contacted`, `emp.var.rate`, `education`.

The two rankings use different units, so compare the order, not the numbers.

- **Economic indicators dominate** both rankings.
- **Contact type** (cellular vs. telephone) is the strongest client-level feature.
- `previously_contacted` and `campaign` give smaller but real effects.

![SHAP summary](shap_summary.png)

## Limitations
- The economic columns change slowly over time, so they mostly tell the model **when** the call happened rather than **who** was called. The data covers 2008 to 2010, so the model may not transfer to a different economic period.
- `nr.employed`, `euribor3m` and `emp.var.rate` are strongly correlated, so how importance is shared among them is somewhat arbitrary. Read them as a group.
- The target is subscription to a term deposit, not churn.

## Repo contents
```
.
├── Bank_Customer_Response_Prediction.ipynb   # full workflow: cleaning, EDA, models, explanation
├── images/                                   # gains chart and SHAP plots used in this README
├── requirements.txt
└── README.md
```

## Run it
```bash
pip install -r requirements.txt
jupyter notebook Bank_Customer_Response_Prediction.ipynb
```
The notebook was developed in Google Colab. If you run it locally, remove the two Google Drive lines in the setup cell and set `FOLDER` to the folder holding the dataset.

## Citation
S. Moro, P. Cortez and P. Rita. *A Data-Driven Approach to Predict the Success of Bank Telemarketing.* Decision Support Systems, 62:22-31, 2014.
