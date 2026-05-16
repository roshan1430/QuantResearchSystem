# Model Comparison Report

## Ranking

```text
        model    mae   rmse  directional_accuracy  cv_mae  cv_rmse  cv_directional_accuracy  cumulative_return  sharpe_ratio  sortino_ratio  max_drawdown    cagr  volatility  win_rate  turnover
random_forest 0.0039 0.0042                 0.750  0.0030   0.0033                   0.5833             0.0128        5.6952        10.0359       -0.0072  0.4912      0.0706     0.625     0.875
         lstm 0.0379 0.0476                 0.625  0.0421   0.0498                   0.7500            -0.0019       -0.7327        -1.2709       -0.0072 -0.0596      0.0800     0.500     1.125
  transformer 0.1598 0.2036                 0.750  0.1124   0.1504                   0.3333            -0.0021       -0.8803        -1.3485       -0.0072 -0.0639      0.0724     0.500     1.625
      xgboost 0.0046 0.0053                 0.375  0.0031   0.0038                   0.6667            -0.0230      -11.2622       -13.6036       -0.0215 -0.5195      0.0648     0.125     1.375
```

## Takeaways

- Best Sharpe ratio: `random_forest`.
- Best directional accuracy: `random_forest`.
- Lowest RMSE: `random_forest`.
