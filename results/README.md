# Results Directory

## Contents

### metrics/
- `evaluation_results.csv`: Detailed performance metrics for each company
- `sector_performance.csv`: Average performance by sector
- `day_performance.csv`: Performance by prediction day (T+1, T+2, T+3)

### visualizations/
- `training_history.png`: Loss and MAE curves during training
- `sector_comparison.png`: MAPE comparison across sectors
- `prediction_samples.png`: Sample prediction visualizations

## Key Findings

1. **Best Performing Sector**: Banking (MAPE: 2.05%)
2. **Most Challenging Sector**: Technology (MAPE: 12.92%)
3. **Prediction Accuracy**: Decreases from T+1 to T+3
4. **Average R² Score**: 0.94 across all sectors

## Performance Summary

| Metric | T+1 | T+2 | T+3 |
|--------|-----|-----|-----|
| MAPE | 3.8% | 5.2% | 6.6% |
| R² | 0.96 | 0.94 | 0.92 |