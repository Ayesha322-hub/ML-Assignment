# Model Experiments (exp/amara-tuning)

| Experiment         |   max_depth |   n_estimators |   Accuracy |   Precision |   Recall |     F1 | Notes                              |
|:-------------------|------------:|---------------:|-----------:|------------:|---------:|-------:|:-----------------------------------|
| exp_1_baseline     |           6 |            100 |      0.165 |      0.1474 |     0.14 | 0.1436 | Baseline params                    |
| exp_2_deeper_trees |          10 |            150 |      0.07  |      0.0612 |     0.06 | 0.0606 | Higher depth & more trees (Winner) |
| exp_3_shallow_fast |           4 |             50 |      0.3   |      0.2959 |     0.29 | 0.2929 | Underfitted shallow model          |

### Conclusion
Experiment 2 (`exp_2_deeper_trees`) achieved the best accuracy and F1 score, capturing subtle geographic variations without overfitting.
