'''
TAG: HOWTO; AI, Tensorflow
There is a problem with tensorflow because it's delivered with its own version of keras.
That's why it is mandatory that we never use keras components directly.
Instead we use the corresponding class available in tensorflow.python.keras.*
'''
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.python
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.python.keras.layers import Embedding, Dense
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.python.keras.optimizers import adam_v2
from keras.src.utils import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
'''
There is a problem with tensorflow data_adapter._is_distributed_dataset function.
Because of that we need to override this function with another one.
The problem is that an original class reference was wrong and we needed to provide a new one.
See: https://stackoverflow.com/questions/77125999/google-colab-tensorflow-distributeddatasetinterface-error
'''
from tensorflow.python.keras.engine import data_adapter

def _is_distributed_dataset(ds):
    return isinstance(ds, data_adapter.input_lib.DistributedDatasetSpec)

data_adapter._is_distributed_dataset = _is_distributed_dataset

data = pd.read_csv('dataset_paper_f_cleaned_minmax.csv')

pep_seq = data['sequence'].tolist()
pep_rt = data['retention_time'].tolist()
print("imported successfully")

# Create a dictionary to map amino acids to integer
amino_acid_dict = {aa: idx for idx, aa in enumerate("ACDEFGHIKLMNPQRSTVWY", start=1)}

# Convert sequences to numerical features
numeric_seq = [[amino_acid_dict.get(aa, -1) for aa in seq] for seq in pep_seq]

# Pad sequences to ensure uniform input size
max_length = max(len(seq) for seq in numeric_seq)
X_seq = pad_sequences(numeric_seq, maxlen=max_length, padding='post', value=0)

# Converting to numpy arrays
y = np.array(pep_rt)

# Datasplitting
X_seq_train, X_seq_test, y_train, y_test = train_test_split(X_seq, y, test_size=0.2, random_state=42)
print("data was split")

# Reshape sequence data for LSTM
X_seq_train = np.expand_dims(X_seq_train, axis=-1)
X_seq_test = np.expand_dims(X_seq_test, axis=-1)

# Define the model
model = Sequential([
    Embedding(input_dim=21, output_dim=16, input_length=max_length, mask_zero=True),
    #LSTM(128, return_sequences=True),
    LSTM(32, return_sequences=True),
    LSTM(16),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='linear')
])

optimizer = adam_v2.Adam(learning_rate=0.001)

# Compile the model
model.compile(optimizer=optimizer, loss='mse', metrics='mae')

# Train the model with early stopping and reducing LR on plateau
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    min_delta=0.0001
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=2,
    verbose=1
)

history = model.fit(
    X_seq_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_seq_test, y_test),
    callbacks=[early_stopping, lr_scheduler],
    verbose=1
)

# Save trained model
# model.save('final_seq_model.keras')
# print("saved")

# Evaluate the model
mse, mae = model.evaluate(X_seq_test, y_test, verbose=1)
print("Test MAE: ", mae)
print("Test MSE: ", mse)

mse, mae = model.evaluate(X_seq_train, y_train, verbose=1)
print("Train MAE: ", mae)
print("Train MSE: ", mse)

plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('RNN model MAE')
plt.ylabel('MAE')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='upper right')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('RNN model MSE')
plt.ylabel('MSE')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='upper right')
plt.show()

y_pred = model.predict(X_seq_test)
r2sc = r2_score(y_test, y_pred)
print(r2sc)
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel("Actual RT")
plt.ylabel("Predicted RT")
plt.title("RNN model Validation Data")
plt.show()

y_pred = model.predict(X_seq_train)
r2sc = r2_score(y_train, y_pred)
print(r2sc)
plt.figure(figsize=(8, 8))
plt.scatter(y_train, y_pred, alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', linewidth=2)
plt.xlabel("Actual RT")
plt.ylabel("Predicted RT")
plt.title("RNN model Training Data")
plt.show()
