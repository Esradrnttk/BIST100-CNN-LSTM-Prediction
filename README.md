# BIST 100 Stock Price Prediction using CNN-LSTM

## 📊 BBL 514E - Deep Learning (Spring 2025)

### 👥 Team Members
- **Esra Durnaoğlu** - Model Development & Implementation
- **[Team Member 2]** - Data Preprocessing & Analysis
- **[Team Member 3]** - Evaluation & Visualization

## 🎯 Project Overview
This project implements a hybrid CNN-LSTM deep learning model for predicting stock prices of BIST 100 companies.

### 🏢 Companies Analyzed
- **Banking**: Garanti (GARAN), İş Bank (ISCTR)
- **Technology**: Karel Electronics (KAREL), Logo Software (LOGO)
- **Retail**: BİM (BIMAS), Migros (MGROS)
- **Energy**: Tüpraş (TUPRS), Petkim (PETKM)
- **Automotive**: Arçelik (ARCLK), Ford Otosan (FROTO)

## 🧠 Model Architecture
```
Input (10 days × 39 features)
    ↓
Conv1D (32 filters, kernel=1, activation=swish)
    ↓
MaxPooling1D (pool_size=1)
    ↓
LSTM (64 units, activation=swish)
    ↓
Dropout (0.3)
    ↓
Dense Output (3 days prediction)
```

## 📈 Performance Metrics
- **Average MAPE**: ~5.2%
- **Average R²**: ~0.94
- **Best Sector**: Banking (MAPE: 2.05%)

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the Model
```bash
python code/main.py
```

## 📊 Results Summary
| Sector | MAPE (%) | R² Score |
|--------|----------|----------|
| Banking | 2.05 | 0.975 |
| Technology | 12.92 | 0.891 |
| Retail | 5.43 | 0.943 |
| Energy | 4.21 | 0.952 |
| Automotive | 6.87 | 0.928 |

## 📄 License
This project is licensed under the MIT License.

---
*Last Updated: June 02, 2025*