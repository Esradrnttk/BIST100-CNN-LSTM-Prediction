#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BIST 100 CNN-LSTM Hisse Senedi Tahmin Modeli
=============================================
Author: Esra Derin Tetik (tetik24@itu.edu.tr)
Course: BBL 514E - Deep Learning (Spring 2025)
Advisor: Prof. Dr. Behçet Uğur Töreyin

Bu kod, BIST 100'de işlem gören 10 farklı şirketin (5 sektörden) 
gelecek 3 günlük fiyat tahminlerini CNN-LSTM hibrit modeli ile yapmaktadır.

Veri Seti İçeriği:
- 10 Şirket: GARAN, ISCTR (Bankacılık), KAREL, LOGO (Teknoloji), 
             BIMAS, MGROS (Perakende), TUPRS, PETKM (Enerji), 
             ARCLK, FROTO (Otomotiv/Dayanıklı Tüketim)
- Teknik Göstergeler: RSI(14), MACD(26,12), Bollinger Bantları, MA(7,21,50)
- Makroekonomik Veriler: USDTRY, EURTRY, Enflasyon, Politika Faizi
"""

import os
import glob
import pickle
import warnings
warnings.filterwarnings('ignore')

# Temel kütüphaneler
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Görselleştirme
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Makine öğrenmesi
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# TensorFlow ve Keras
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Conv1D, MaxPooling1D, LSTM, Reshape, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ===========================
# GPU OPTİMİZASYONU
# ===========================
print("GPU Kontrolü ve Optimizasyon")
print("="*60)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        print(f"Kullanılabilir GPU sayısı: {len(gpus)}")
        
        # A100 için optimizasyon
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print(f"Mixed precision policy: {policy.name}")
        
    except RuntimeError as e:
        print(f"GPU ayarları yapılırken hata: {e}")
else:
    print("GPU bulunamadı, CPU kullanılacak.")

# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Worker thread optimization
tf.config.threading.set_inter_op_parallelism_threads(8)
tf.config.threading.set_intra_op_parallelism_threads(8)

# ===========================
# GLOBAL PARAMETRELERr
# ===========================
# Veri parametreleri
SEQ_LENGTH = 10  # 10 günlük geçmiş veri
PRED_LENGTH = 3  # 3 günlük tahmin

# Model parametreleri
MODEL_PARAMS = {
    'filters': 32,          # CNN filtre sayısı
    'kernel_size': 1,       # CNN kernel boyutu
    'lstm_units': 64,       # LSTM birim sayısı
    'dropout_rate': 0.3,    # Dropout oranı
}

# Eğitim parametreleri
TRAIN_PARAMS = {
    'epochs': 100,
    'batch_size': 64,
    'learning_rate': 0.001,
    'patience': 20,
    'reduce_lr_patience': 10,
    'min_lr': 0.0001,
}

# Şirket listesi
COMPANIES = ['garan', 'isctr', 'karel', 'logo', 'bimas', 'mgros', 'tuprs', 'petkm', 'arclk', 'froto']

# Sektör tanımlamaları
SECTORS = {
    'Bankacılık': ['garan', 'isctr'],
    'Teknoloji': ['karel', 'logo'],
    'Perakende': ['bimas', 'mgros'],
    'Enerji': ['tuprs', 'petkm'],
    'Otomotiv/Dayanıklı Tüketim': ['arclk', 'froto']
}

# Şirket isimleri (Türkçe)
COMPANY_NAMES = {
    'garan': 'GARANTİ BANKASI',
    'isctr': 'İŞ BANKASI',
    'karel': 'KAREL ELEKTRONİK',
    'logo': 'LOGO YAZILIM',
    'bimas': 'BİM MAĞAZALAR',
    'mgros': 'MİGROS',
    'tuprs': 'TÜPRAŞ',
    'petkm': 'PETKİM',
    'arclk': 'ARÇELİK',
    'froto': 'FORD OTOSAN'
}

# ===========================
# YARDIMCI FONKSİYONLAR
# ===========================

def swish(x):
    """Swish aktivasyon fonksiyonu: f(x) = x * sigmoid(x)"""
    return x * tf.keras.backend.sigmoid(x)

# Custom aktivasyon fonksiyonunu kaydet
tf.keras.utils.get_custom_objects().update({'swish': tf.keras.layers.Activation(swish)})

def load_datasets(file_paths):
    """CSV dosyalarını yükler ve veri sözlüğü oluşturur."""
    data_dict = {}
    
    for file_path in file_paths:
        try:
            file_name = os.path.basename(file_path)
            dataset_name = file_name.split('_')[0].lower()
            
            # CSV okuma
            separators = [',', ';', '\t', '|']
            df = None
            
            for sep in separators:
                try:
                    df = pd.read_csv(file_path, sep=sep, decimal='.', 
                                   on_bad_lines='warn', encoding='utf-8')
                    if len(df.columns) > 1:
                        break
                except:
                    continue
                    
            if df is None:
                print(f"UYARI: {file_name} dosyası okunamadı")
                continue
                
            # Sütun temizleme
            df.columns = df.columns.str.strip()
            
            # Tarih sütunu
            date_columns = ['TARİH', 'Tarih', 'TARIH', 'tarih', 'DATE', 'Date']
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
                    df.set_index(col, inplace=True)
                    break
                    
            data_dict[dataset_name] = df
            print(f"✓ {dataset_name}: {df.shape[0]} satır, {df.shape[1]} sütun yüklendi")
            
        except Exception as e:
            print(f"HATA: {file_name} yüklenirken hata: {str(e)}")
            
    return data_dict

def prepare_company_data(company, datasets, macro_keys):
    """Bir şirket için tüm verileri birleştirir."""
    if company not in datasets:
        return None
        
    df = datasets[company].copy()
    
    # Makroekonomik verileri ekle
    for key in macro_keys:
        if key in datasets:
            macro_df = datasets[key].copy()
            macro_df.columns = [f'{key}_{col}' for col in macro_df.columns]
            df = df.join(macro_df, how='left')
            
    return df

def create_sequences(df, target_column, seq_length, pred_length=1, scale=True):
    """Zaman serisi verilerini CNN-LSTM için uygun formata dönüştürür."""
    # Hedef sütun kontrolü
    if target_column not in df.columns:
        alt_names = ['SON', 'Son', 'KAPANIŞ', 'Kapanış', 'CLOSE', 'Close']
        for alt in alt_names:
            if alt in df.columns:
                target_column = alt
                break
    
    data = df.copy()
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    data = data[numeric_cols]
    
    target = data[target_column].values
    
    # Ölçeklendirme
    if scale:
        scaler_X = MinMaxScaler(feature_range=(0, 1))
        scaler_y = MinMaxScaler(feature_range=(0, 1))
        
        data_scaled = scaler_X.fit_transform(data)
        target_scaled = scaler_y.fit_transform(target.reshape(-1, 1)).flatten()
    else:
        data_scaled = data.values
        target_scaled = target
        scaler_X = None
        scaler_y = None
    
    # Sequence oluştur
    X, y = [], []
    
    for i in range(len(data_scaled) - seq_length - pred_length + 1):
        X.append(data_scaled[i:i+seq_length])
        if pred_length == 1:
            y.append(target_scaled[i+seq_length])
        else:
            y.append(target_scaled[i+seq_length:i+seq_length+pred_length])
    
    return np.array(X), np.array(y), scaler_X, scaler_y

def reshape_for_cnn(X):
    """Veriyi CNN katmanı için yeniden şekillendirir."""
    return X.reshape(X.shape[0], X.shape[1], 1, X.shape[2])

# ===========================
# CNN-LSTM MODEL
# ===========================

def create_cnn_lstm_model(input_shape, output_shape, filters=32, kernel_size=1, 
                         lstm_units=64, dropout_rate=0.3):
    """
    CNN-LSTM hibrit modelini oluşturur.
    
    Model Mimarisi:
    1. Input Layer
    2. Reshape Layer (4D -> 3D)
    3. Conv1D Layer (özellik çıkarımı)
    4. MaxPooling1D Layer
    5. LSTM Layer (temporal patterns)
    6. Dropout Layer (regularization)
    7. Dense Output Layer
    """
    
    # Input layer
    input_layer = Input(shape=input_shape, name='input_layer')
    
    # Reshape
    x = Reshape((input_shape[0], input_shape[1] * input_shape[2]), 
                name='reshape_layer')(input_layer)
    
    # 1D Convolutional layer
    x = Conv1D(filters=filters, 
               kernel_size=kernel_size, 
               padding='same',
               activation='swish',
               name='conv1d_layer')(x)
    
    # Max pooling
    x = MaxPooling1D(pool_size=1, 
                     padding='same',
                     name='maxpool_layer')(x)
    
    # LSTM layer
    x = LSTM(units=lstm_units, 
             activation='swish',
             return_sequences=False,
             name='lstm_layer')(x)
    
    # Dropout
    x = Dropout(dropout_rate, name='dropout_layer')(x)
    
    # Output layer
    output_layer = Dense(output_shape, 
                        activation='linear',
                        name='output_layer')(x)
    
    # Create model
    model = Model(inputs=input_layer, outputs=output_layer, name='CNN_LSTM_Model')
    
    return model

# ===========================
# DEĞERLENDİRME FONKSİYONLARI
# ===========================

def evaluate_model(model, X_test, y_test, scaler_y, company_name=""):
    """Model performansını değerlendirir."""
    # Tahmin yap
    y_pred = model.predict(X_test, verbose=0)
    
    # Ölçeklendirmeyi geri al
    y_test_orig = scaler_y.inverse_transform(y_test)
    y_pred_orig = scaler_y.inverse_transform(y_pred)
    
    # Metrikler
    results = []
    for day in range(y_pred_orig.shape[1]):
        mae = mean_absolute_error(y_test_orig[:, day], y_pred_orig[:, day])
        mse = mean_squared_error(y_test_orig[:, day], y_pred_orig[:, day])
        rmse = np.sqrt(mse)
        
        # R² hesapla
        if np.var(y_test_orig[:, day]) > 0:
            r2 = r2_score(y_test_orig[:, day], y_pred_orig[:, day])
        else:
            r2 = 0.0
            
        # MAPE hesapla
        mask = y_test_orig[:, day] != 0
        if np.any(mask):
            mape = np.mean(np.abs((y_test_orig[mask, day] - y_pred_orig[mask, day]) / 
                                 y_test_orig[mask, day])) * 100
        else:
            mape = 0.0
        
        results.append({
            'Company': company_name,
            'Day': f'T+{day+1}',
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        })
    
    return results, y_test_orig, y_pred_orig

# ===========================
# GÖRSELLEŞTİRME FONKSİYONLARI
# ===========================

def plot_training_history(history):
    """Eğitim geçmişini görselleştirir."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Train Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Model Loss (MAE)')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # MSE
    axes[0, 1].plot(history.history['mean_squared_error'], label='Train MSE')
    axes[0, 1].plot(history.history['val_mean_squared_error'], label='Val MSE')
    axes[0, 1].set_title('Model MSE')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MSE')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning Rate
    if 'lr' in history.history:
        axes[1, 0].plot(history.history['lr'])
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('LR')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_predictions(y_true, y_pred, company_name, save_path=None):
    """Tahmin sonuçlarını görselleştirir."""
    plt.figure(figsize=(15, 8))
    
    # Son 50 gün
    days_to_show = min(50, len(y_true))
    x_axis = np.arange(days_to_show)
    
    # Gerçek değerler
    plt.plot(x_axis, y_true[-days_to_show:, 0], 
            'o-', label='Gerçek Değer', color='black', linewidth=2)
    
    # Tahminler
    colors = ['#1e88e5', '#ff5722', '#43a047']
    for i in range(3):
        plt.plot(x_axis, y_pred[-days_to_show:, i], 
                '--', label=f'T+{i+1} Tahmini', color=colors[i], linewidth=2)
    
    plt.title(f'{company_name} - Test Seti Tahmin Performansı (Son {days_to_show} Gün)')
    plt.xlabel('Gün')
    plt.ylabel('Hisse Fiyatı (TL)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    return plt.gcf()

# ===========================
# ANA FONKSİYON
# ===========================

def main():
    """Ana çalıştırma fonksiyonu"""
    print("BIST 100 CNN-LSTM Hisse Senedi Fiyat Tahmin Modeli")
    print("="*60)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Başlangıç zamanı: {datetime.now()}")
    
    # Veri yükleme ve model eğitimi kodları buraya eklenecek
    # Bu bir template/framework olarak kullanılabilir
    
    print("\nModel template hazır!")
    print("Veri setinizi yükleyerek modeli eğitebilirsiniz.")
    
if __name__ == "__main__":
    main()
