# BIST 100 CNN-LSTM Demo
# Simplified version for quick testing

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Model parameters
SEQ_LENGTH = 10
PRED_LENGTH = 3

def create_model(input_shape):
    """Create simplified CNN-LSTM model"""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(32, 1, activation='relu', input_shape=input_shape),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(PRED_LENGTH)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Example usage
if __name__ == "__main__":
    # Generate sample data
    X = np.random.rand(100, SEQ_LENGTH, 5)  # 100 samples, 10 timesteps, 5 features
    y = np.random.rand(100, PRED_LENGTH)    # 100 samples, 3 outputs
    
    # Create and train model
    model = create_model((SEQ_LENGTH, 5))
    model.summary()
    
    print("\nDemo model created successfully!")
    print("Use with real BIST 100 data for actual predictions.")
