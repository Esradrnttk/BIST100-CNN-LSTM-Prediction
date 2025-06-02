# Step 4: CNN-LSTM Model Oluşturma

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Conv1D, MaxPooling1D, LSTM, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# GPU ayarları
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print('GPU kullanımı aktif')

# Swish aktivasyon fonksiyonu
def swish(x):
    return x * tf.keras.backend.sigmoid(x)

tf.keras.utils.get_custom_objects().update({'swish': tf.keras.layers.Activation(swish)})

# CNN-LSTM model fonksiyonu
def create_cnn_lstm_model(input_shape, output_shape, filters=32, kernel_size=1,
                         lstm_units=64, dropout_rate=0.3):
    # Input layer
    input_layer = Input(shape=input_shape)
    
    # Reshape
    x = Reshape((input_shape[0], input_shape[1] * input_shape[2]))(input_layer)
    
    # 1D CNN
    x = Conv1D(filters=filters, kernel_size=kernel_size, padding='same',
               activation='swish')(x)
    
    # MaxPooling
    x = MaxPooling1D(pool_size=1, padding='same')(x)
    
    # LSTM
    x = LSTM(units=lstm_units, activation='swish', return_sequences=False)(x)
    
    # Dropout
    x = Dropout(dropout_rate)(x)
    
    # Output
    output_layer = Dense(output_shape)(x)
    
    # Model
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Parametreler
epochs = 100
batch_size = 64
learning_rate = 0.001

# Model oluştur
input_shape = X_train_all.shape[1:]
output_shape = y_train_all.shape[1]

cnn_lstm_model = create_cnn_lstm_model(
    input_shape=input_shape,
    output_shape=output_shape,
    filters=32,
    kernel_size=1,
    lstm_units=64,
    dropout_rate=0.3
)

# Model derleme
cnn_lstm_model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss='mean_absolute_error',
    metrics=['mean_absolute_error', 'mean_squared_error']
)

cnn_lstm_model.summary()