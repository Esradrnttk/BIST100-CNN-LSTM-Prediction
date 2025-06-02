# Models Directory

## Trained Models

This directory contains trained CNN-LSTM models.

## Model Files
- `best_model.h5`: Best performing model from training
- `final_model.h5`: Final model after all epochs

## Loading Models

```python
from tensorflow.keras.models import load_model

# Load the model
model = load_model('best_model.h5')

# Make predictions
predictions = model.predict(prepared_data)
```

## Model Architecture Summary
- Input: (10, 39) - 10 days, 39 features
- Conv1D: 32 filters, kernel size 1
- LSTM: 64 units
- Dropout: 0.3
- Output: 3 units (3-day prediction)

## Note
Due to file size, trained models are not included in the repository.
Please train the model using the provided code or contact authors for pre-trained weights.