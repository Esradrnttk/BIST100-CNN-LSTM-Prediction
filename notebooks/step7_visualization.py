# Step 7: Görselleştirme ve Raporlama

import matplotlib.pyplot as plt
import seaborn as sns
import os

# Klasör oluştur
base_dir = '/content/drive/MyDrive/data/BIST100_CNN_LSTM_Model/'
graphs_dir = os.path.join(base_dir, 'graphs')
os.makedirs(graphs_dir, exist_ok=True)

# Stil ayarları
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

# Şirket isimleri
company_names = {
    'garan': 'GARANTİ BANKASI',
    'ısctr': 'İŞ BANKASI',
    'karel': 'KAREL ELEKTRONİK',
    'logo': 'LOGO YAZILIM',
    'bımas': 'BİM MAĞAZALAR',
    'mgros': 'MİGROS',
    'tuprs': 'TÜPRAŞ',
    'petkm': 'PETKİM',
    'arclk': 'ARÇELİK',
    'froto': 'FORD OTOSAN'
}

# Sektör MAPE grafiği
plt.figure(figsize=(12, 8))
sector_mape = results_df.groupby(['Sektör', 'Gün'])['MAPE'].mean().unstack()

sector_mape.plot(kind='bar', width=0.8)
plt.title('Sektörlere Göre MAPE Değerleri', fontsize=16)
plt.xlabel('Sektör', fontsize=14)
plt.ylabel('MAPE (%)', fontsize=14)
plt.legend(title='Tahmin Günü')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, 'sector_mape_comparison.png'), dpi=300)
plt.show()

# R² ısı haritası
plt.figure(figsize=(10, 8))
pivot_r2 = results_df.pivot_table(values='R²', index='Şirket', columns='Gün')
sns.heatmap(pivot_r2, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.8, vmax=1.0)
plt.title('Şirket ve Gün Bazında R² Skorları')
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, 'r2_heatmap.png'), dpi=300)
plt.show()

print("Görselleştirme tamamlandı!")
print(f"Grafikler kaydedildi: {graphs_dir}")
