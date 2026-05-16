# Backtesting Report

## Strategy Summary

```text
 cumulative_return  sharpe_ratio  sortino_ratio  max_drawdown   cagr  volatility  win_rate  turnover
            0.0128        5.6952        10.0359       -0.0072 0.4912      0.0706     0.625     0.875
```

## Strategy Comparison

```text
                     strategy  cumulative_return  sharpe_ratio  sortino_ratio  max_drawdown   cagr  volatility  win_rate  turnover
     adaptive_regime_strategy             0.0128        5.6952        10.0359       -0.0072 0.4912      0.0706     0.625    0.8750
         directional_baseline             0.0128        5.6952        10.0359       -0.0072 0.4912      0.0706     0.625    0.8750
full_sample_regime_diagnostic             0.0382        4.4441        11.6825       -0.0278 0.2821      0.0563     0.500    0.4211
```

## Regime-Level Performance

```text
        cumulative_return  sharpe_ratio  sortino_ratio  max_drawdown    cagr  volatility  win_rate  turnover  observations
regime                                                                                                                    
trend              0.0679       16.1752        57.2629       -0.0030  1.2885      0.0513      0.85    0.4250          20.0
stress            -0.0028       -4.8831        -6.4609       -0.0049 -0.1308      0.0286      0.40    0.4000           5.0
range             -0.0251      -22.7325       -22.7325       -0.0244 -0.3895      0.0217      0.00    0.4231          13.0
```
