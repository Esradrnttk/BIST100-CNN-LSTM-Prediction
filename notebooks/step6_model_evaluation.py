# Step 6: Model Değerlendirme

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sonuç DataFrame'i
results_df = pd.DataFrame(columns=['Şirket', 'Gün', 'MAE', 'MSE', 'RMSE', 'R²', 'MAPE'])

# Şirket ve sektör tanımları
companies = ['garan', 'ısctr', 'karel', 'logo', 'bımas', 'mgros', 'tuprs', 'petkm', 'arclk', 'froto']
sectors = {
    'Bankacılık': ['garan', 'ısctr'],
    'Teknoloji': ['karel', 'logo'],
    'Perakende': ['bımas', 'mgros'],
    'Enerji': ['tuprs', 'petkm'],
    'Otomotiv/Dayanıklı Tüketim': ['arclk', 'froto']
}

# Her şirket için değerlendirme
for company in companies:
    X_test = datasets[company]['X_test']
    y_test = datasets[company]['y_test']
    scaler_y = datasets[company]['scaler_y']
    
    # Tahmin
    y_pred = cnn_lstm_model.predict(X_test, verbose=0)
    
    # Ters ölçeklendirme
    y_test_inverse = scaler_y.inverse_transform(y_test)
    y_pred_inverse = scaler_y.inverse_transform(y_pred)
    
    # Metrikler
    for day in range(y_test.shape[1]):
        mae = mean_absolute_error(y_test_inverse[:, day], y_pred_inverse[:, day])
        mse = mean_squared_error(y_test_inverse[:, day], y_pred_inverse[:, day])
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_inverse[:, day], y_pred_inverse[:, day])
        mape = np.mean(np.abs((y_test_inverse[:, day] - y_pred_inverse[:, day]) / y_test_inverse[:, day])) * 100
        
        # Sonuç ekle
        new_row = {
            'Şirket': company,
            'Gün': f'T+{day+1}',
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R²': r2,
            'MAPE': mape,
            'Sektör': next((s for s, c in sectors.items() if company in c), None)
        }
        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)

# Sonuçları kaydet
results_df.to_csv(os.path.join(output_dir, 'model_evaluation_results.csv'), index=False)
print("Değerlendirme tamamlandı!")
print(results_df.head())

# Sektör analizi
sector_results = results_df.groupby('Sektör')[['MAE', 'MAPE', 'R²']].mean()
print("\nSektör Bazlı Performans:")
print(sector_results)
