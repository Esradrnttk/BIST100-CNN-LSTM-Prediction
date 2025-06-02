# Step 1: Google Drive Bağlantısı ve Veri Seti Keşfi

from google.colab import drive
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Google Drive'a bağlanma
drive.mount('/content/drive')

# Veri seti dizini
data_path = '/content/drive/MyDrive/data/data_sets/'

# Dizin içeriğini listeleme
print("Dizin İçeriği:")
for root, dirs, files in os.walk(data_path):
    level = root.replace(data_path, '').count(os.sep)
    indent = ' ' * 4 * level
    print(f"{indent}{os.path.basename(root)}/")
    sub_indent = ' ' * 4 * (level + 1)
    for file in files:
        print(f"{sub_indent}{file}")

# Örnek bir hisse senedi veri seti
sample_stock = pd.read_csv(data_path + 'garan_train.csv', sep=',', decimal='.',
                           on_bad_lines='warn', encoding='utf-8')

# Örnek bir ekonomik gösterge veri seti
sample_macro = pd.read_csv(data_path + 'usdtry_train.csv', sep=',', decimal='.',
                          on_bad_lines='warn', encoding='utf-8')

# Hisse senedi veri setinin ilk 5 satırını görelim
print("\nHisse Senedi Veri Seti (İlk 5 Satır):")
print(sample_stock.head())
print("\nHisse Senedi Veri Seti Bilgileri:")
print(sample_stock.info())

# Ekonomik gösterge veri setinin ilk 5 satırını görelim
print("\nEkonomik Gösterge Veri Seti (İlk 5 Satır):")
print(sample_macro.head())
print("\nEkonomik Gösterge Veri Seti Bilgileri:")
print(sample_macro.info())
