# Experiment Report: offline_sample_random_forest

## Overview

- Model: `random_forest`
- Walk-forward MAE: `0.003907`
- Walk-forward RMSE: `0.004154`
- Directional accuracy: `75.00%`
- Sharpe ratio: `5.695`
- Sortino ratio: `10.036`
- CAGR: `49.12%`

## Cross-Validation Summary

```text
 fold train_start  train_end test_start   test_end    mae   rmse  directional_accuracy
    1  2024-02-02 2024-03-07 2024-03-08 2024-03-13 0.0021 0.0026                  0.75
    2  2024-02-02 2024-03-13 2024-03-14 2024-03-19 0.0043 0.0046                  0.25
    3  2024-02-02 2024-03-19 2024-03-20 2024-03-25 0.0027 0.0028                  0.75
```

## Backtest Metrics

```text
 cumulative_return  sharpe_ratio  sortino_ratio  max_drawdown   cagr  volatility  win_rate  turnover
            0.0128        5.6952        10.0359       -0.0072 0.4912      0.0706     0.625     0.875
```

## Regime Detection Summary

```text
        observations  avg_volatility  avg_momentum  avg_range_ratio
regime                                                             
stress             5          0.0804        0.0583           0.0129
range             13          0.0782        0.0671           0.0124
trend             20          0.0682        0.0739           0.0121
```

## Performance By Regime

```text
        cumulative_return  sharpe_ratio  sortino_ratio  max_drawdown    cagr  volatility  win_rate  turnover  observations
regime                                                                                                                    
trend              0.0679       16.1752        57.2629       -0.0030  1.2885      0.0513      0.85    0.4250          20.0
stress            -0.0028       -4.8831        -6.4609       -0.0049 -0.1308      0.0286      0.40    0.4000           5.0
range             -0.0251      -22.7325       -22.7325       -0.0244 -0.3895      0.0217      0.00    0.4231          13.0
```

## Statistical Diagnostics

```text
 adf_pvalue  adf_statistic  close_return_correlation  realized_volatility cointegration_pvalue
        1.0        -0.4564                       1.0               0.0635                 None
```

## Explainability Snapshot

```text
              feature  mean_abs_shap      importance_source
               rsi_14         0.5087 permutation_importance
     volume_zscore_21         0.0854 permutation_importance
           momentum_5         0.0477 permutation_importance
         return_lag_5         0.0432 permutation_importance
         return_lag_3         0.0379 permutation_importance
realized_volatility_5         0.0334 permutation_importance
          drawdown_21         0.0250 permutation_importance
          range_ratio         0.0171 permutation_importance
         volume_lag_3         0.0161 permutation_importance
         volume_lag_5         0.0093 permutation_importance
```
