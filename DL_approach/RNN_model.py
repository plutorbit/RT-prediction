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
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers.core import Dense, Masking
from tensorflow.python.keras.layers.merge import Concatenate
from tensorflow.python.keras.layers.recurrent import LSTM
from keras.src.utils import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

######################
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

import tensorflow.python.keras as tf_keras
tf_keras.__version__="3.0"
######################

data = pd.read_csv('grouped_cystein5_feature_ex.csv')

pep_seq = data['sequence'].tolist()
pep_rt = data['retention_time'].tolist()
pep_hydro = data['hydrophobicity'].tolist()
pep_aroma = data['aromaticity'].tolist()
pep_cCount = data['cystein_count'].tolist()
pep_isMod = data['is_modified'].tolist()
pep_seq_len = data['seq_length'].tolist()
pep_mWeight = data['molecular_weight'].tolist()
pep_instabIdx = data['instability_index'].tolist()
pep_isoPoint = data['isoelectric_point'].tolist()

print("imported successfully")

# Create a dictionary to map amino acids to integer
amino_acid_dict = {aa: idx for idx, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}

# Convert sequences to numerical features
numeric_seq = [[amino_acid_dict.get(aa, -1) for aa in seq] for seq in pep_seq]

# Pad sequences to ensure uniform input size
max_length = max(len(seq) for seq in numeric_seq)
X_seq = pad_sequences(numeric_seq, maxlen=max_length, padding='post', value=-1)

# Converting to numpy arrays
X_additional_features = np.array(list(zip(pep_hydro, pep_aroma, pep_cCount, pep_isMod, pep_seq_len, pep_mWeight, pep_instabIdx, pep_isoPoint)))

y = np.array(pep_rt)

# Datasplitting
X_seq_train, X_seq_test, X_additional_train, X_additional_test, y_train, y_test = train_test_split(X_seq, X_additional_features, y, test_size=0.2, random_state=42)
print("data was split")

# Reshape sequence data for LSTM
X_seq_train = np.expand_dims(X_seq_train, axis=-1)
X_seq_test = np.expand_dims(X_seq_test, axis=-1)

# Define sequence input
sequence_input = Input(shape=(X_seq_train.shape[1], 1))

# Apply masking to handle padded data
masked_input = Masking(mask_value=-1)(sequence_input)

# LSTM layers for sequence data
lstm_out = LSTM(64, return_sequences=True)(masked_input)
lstm_out = LSTM(32)(lstm_out)

# Define additional features input
additional_input = Input(shape=(X_additional_train.shape[1],))

# Dense layers for additional features
dense_out = Dense(64, activation='relu')(additional_input)
dense_out = Dense(32, activation='relu')(dense_out)

# Combine both inputs
combined = Concatenate()([lstm_out, dense_out])
combined_out = Dense(64, activation='relu')(combined)
combined_out = Dense(32, activation='relu')(combined_out)
output = Dense(1)(combined_out)

print("attempting to initialize model")
# print(type(sequence_input))
# print(type(additional_input))
# print(type(output))

if True:
    # Define the model
    model = Model(inputs=[sequence_input, additional_input], outputs=output)
    print("model initialized")

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    print("model compiled")

    # Train the model with early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

    history = model.fit(
        [X_seq_train, X_additional_train], y_train,
        epochs=50,
        batch_size=32,
        validation_data=([X_seq_test, X_additional_test], y_test),
        callbacks=[early_stopping]
    )

    # Save trained model
    model.save('model.keras')
    print("saved")
else:
    print("loading model...")
    model = tf_keras.models.load_model('model_I_3epochstest.keras')
    print("succesfully loaded")

# Evaluate the model
print("test data error values...")
y_pred = model.predict([X_seq_test, X_additional_test])
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2sc = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'R^2 score: {r2sc}')
print()

print("training data error values...")
y_pred = model.predict([X_seq_train, X_additional_train])
mse = mean_squared_error(y_train, y_pred)
mae = mean_absolute_error(y_train, y_pred)
r2sc = r2_score(y_train, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'R^2 score: {r2sc}')
