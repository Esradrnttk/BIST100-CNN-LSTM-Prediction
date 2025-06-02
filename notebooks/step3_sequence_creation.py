# Step 3: CNN-LSTM için Sequence Oluşturma

from sklearn.preprocessing import MinMaxScaler

def create_sequences(df, target_column, seq_length, pred_length=1, scale=True):
    """
    Zaman serisi verilerini X (input) ve y (output) dizilerine dönüştürür
    """
    data = df.copy()
    target = data[target_column].values
    
    # Verileri ölçeklendirelim (0-1 arasına normalize et)
    if scale:
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        data_scaled = scaler_X.fit_transform(data)
        target_scaled = scaler_y.fit_transform(target.reshape(-1, 1)).flatten()
    else:
        data_scaled = data.values
        target_scaled = target
        scaler_X = None
        scaler_y = None
    
    X, y = [], []
    
    # Dizileri oluştur
    data_len = len(data_scaled)
    for i in range(data_len - seq_length - pred_length + 1):
        X.append(data_scaled[i:i+seq_length])
        y.append(target_scaled[i+seq_length:i+seq_length+pred_length])
    
    return np.array(X), np.array(y), scaler_X, scaler_y

# Veri setini CNN giriş formatına dönüştür
def reshape_for_cnn(X):
    """Veri setini CNN için yeniden şekillendir"""
    return X.reshape(X.shape[0], X.shape[1], 1, X.shape[2])

# Sequence parametreleri
seq_length = 10  # 10 günlük geçmiş veri
pred_length = 3  # 3 günlük tahmin

# Her şirket için veri setleri hazırla
datasets = {}
for company in companies:
    print(f"{company} için veri setleri hazırlanıyor...")
    
    # Train seti
    X_train, y_train, scaler_X, scaler_y = create_sequences(
        train_sets[company], 'SON', seq_length, pred_length, scale=True
    )
    X_train_cnn = reshape_for_cnn(X_train)
    
    # Validation seti
    X_val, y_val, _, _ = create_sequences(
        val_sets[company], 'SON', seq_length, pred_length, scale=True
    )
    X_val_cnn = reshape_for_cnn(X_val)
    
    # Test seti
    X_test, y_test, _, _ = create_sequences(
        test_sets[company], 'SON', seq_length, pred_length, scale=True
    )
    X_test_cnn = reshape_for_cnn(X_test)
    
    # Veri setlerini kaydet
    datasets[company] = {
        'X_train': X_train_cnn,
        'y_train': y_train,
        'X_val': X_val_cnn,
        'y_val': y_val,
        'X_test': X_test_cnn,
        'y_test': y_test,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y
    }

# Tüm şirketlerin verilerini birleştir
X_train_all = np.concatenate([datasets[company]['X_train'] for company in companies])
y_train_all = np.concatenate([datasets[company]['y_train'] for company in companies])
X_val_all = np.concatenate([datasets[company]['X_val'] for company in companies])
y_val_all = np.concatenate([datasets[company]['y_val'] for company in companies])
