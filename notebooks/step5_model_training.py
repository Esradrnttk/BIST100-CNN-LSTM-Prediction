# Step 5: Model Eğitimi

import os
import pickle
import matplotlib.pyplot as plt

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001),
    ModelCheckpoint('cnn_lstm_best_model.h5', save_best_only=True, monitor='val_loss')
]

# Çıktı dizini
output_dir = '/content/drive/MyDrive/data/model_outputs/'
os.makedirs(output_dir, exist_ok=True)
history_dir = os.path.join(output_dir, 'histories')
os.makedirs(history_dir, exist_ok=True)

# Model eğitimi
history = cnn_lstm_model.fit(
    X_train_all, y_train_all,
    validation_data=(X_val_all, y_val_all),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=callbacks,
    verbose=1
)

# Eğitim tarihçesini kaydet
with open(os.path.join(history_dir, 'general_model_history.pkl'), 'wb') as f:
    pickle.dump(history.history, f)

# Eğitim grafiği
plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Model Kaybı (MAE)')
plt.xlabel('Epochs')
plt.ylabel('Kayıp')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mean_squared_error'], label='Eğitim MSE')
plt.plot(history.history['val_mean_squared_error'], label='Doğrulama MSE')
plt.title('Model MSE')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'general_model_training.png'), dpi=300)
plt.show()