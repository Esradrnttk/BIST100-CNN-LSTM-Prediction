#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BIST 100 CNN-LSTM Stock Price Prediction
# Author: Esra Durnaoğlu

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Model configuration
SEQ_LENGTH = 10
PRED_LENGTH = 3
CNN_FILTERS = 32
LSTM_UNITS = 64
DROPOUT_RATE = 0.3

print('BIST 100 CNN-LSTM Model')
print('Run with actual data for predictions')
