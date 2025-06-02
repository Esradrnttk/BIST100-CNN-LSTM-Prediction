# Step 2: Veri Setlerinin Yüklenmesi ve Birleştirilmesi

import glob
import pandas as pd
from datetime import datetime

# Veri seti klasörleri
train_files = glob.glob(data_path + '*_train.csv')
val_files = glob.glob(data_path + '*_val.csv')
test_files = glob.glob(data_path + '*_test.csv')

# Veri setlerinin yüklenmesi için yardımcı fonksiyon
def load_datasets(file_paths):
    data_dict = {}
    for file_path in file_paths:
        # Dosya adından veri seti adını çıkar
        file_name = file_path.split('/')[-1]
        dataset_name = file_name.split('_')[0]
        
        # CSV dosyasını oku
        df = pd.read_csv(file_path, sep=',', decimal='.', on_bad_lines='warn', encoding='utf-8')
        
        # TARİH sütununu datetime'a çevir
        if 'TARİH' in df.columns:
            df['TARİH'] = pd.to_datetime(df['TARİH'], format='%Y-%m-%d')
            df.set_index('TARİH', inplace=True)
        elif 'Tarih' in df.columns:
            df['Tarih'] = pd.to_datetime(df['Tarih'], format='%Y-%m-%d')
            df.set_index('Tarih', inplace=True)
        
        # Veri setini sözlüğe ekle
        data_dict[dataset_name] = df
    
    return data_dict

# Train, validation ve test setlerini yükle
train_data = load_datasets(train_files)
val_data = load_datasets(val_files)
test_data = load_datasets(test_files)

# Makroekonomik verileri günlük veriye dönüştürme
start_date = train_data['garan'].index.min()
end_date = train_data['garan'].index.max()
all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
daily_index = pd.DataFrame(index=all_dates)

for key in ['enflasyon', 'politika faizi']:
    if key in train_data:
        train_data[key] = daily_index.join(train_data[key]).ffill()

# Veri setlerini şirket bazında birleştirme
def prepare_company_data(company, datasets, macro_keys):
    """Bir şirket için tüm verileri birleştirir"""
    df = datasets[company].copy()
    
    # Makroekonomik ve döviz kuru verileri ekle
    for key in macro_keys:
        if key in datasets:
            macro_df = datasets[key].copy()
            macro_df.columns = [f'{key}_{col}' for col in macro_df.columns]
            df = df.join(macro_df)
    
    return df

# Birleştirme için makroekonomik ve döviz kuru anahtarları
macro_keys = ['usdtry', 'eurtry', 'enflasyon', 'politika faizi']

# Her şirket için veri hazırla
train_sets = {}
val_sets = {}
test_sets = {}

companies = ['garan', 'ısctr', 'karel', 'logo', 'bımas', 'mgros', 'tuprs', 'petkm', 'arclk', 'froto']

for company in companies:
    if company in train_data:
        train_sets[company] = prepare_company_data(company, train_data, macro_keys)
    if company in val_data:
        val_sets[company] = prepare_company_data(company, val_data, macro_keys)
    if company in test_data:
        test_sets[company] = prepare_company_data(company, test_data, macro_keys)

# Veri temizleme ve eksik değerleri doldurma
def clean_and_prepare_data(data_dict):
    for company in data_dict:
        # Sütun adlarındaki boşlukları temizle
        data_dict[company].columns = data_dict[company].columns.str.strip()
        
        # Eksik değerleri doldur
        for col in data_dict[company].columns:
            if col.startswith('enflasyon_') or col.startswith('politika faizi_'):
                data_dict[company][col] = data_dict[company][col].ffill()
        
        # Kalan tüm eksik değerleri medyan ile doldur
        data_dict[company] = data_dict[company].fillna(data_dict[company].median())
        
        # Veri tipini kontrol et ve düzelt
        for col in data_dict[company].columns:
            if data_dict[company][col].dtype == 'object':
                try:
                    data_dict[company][col] = pd.to_numeric(data_dict[company][col])
                except:
                    pass
    
    return data_dict

# Veri setlerini temizle
train_sets = clean_and_prepare_data(train_sets)
val_sets = clean_and_prepare_data(val_sets)
test_sets = clean_and_prepare_data(test_sets)
