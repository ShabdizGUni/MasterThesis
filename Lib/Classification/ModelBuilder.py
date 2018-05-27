import numpy as np
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint


def nn_1_layer(x, y, filepath):
    model = Sequential()
    model.add(Dense(y.shape[1], input_dim=x.shape[1], activation='relu'))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10, verbose=0, mode='auto')
    checkpointer = ModelCheckpoint(filepath=filepath + "/best_weights.hdf5", verbose=0, save_best_only=True)
    return model, monitor, checkpointer


def dnn_2_layers(x, y, filepath):
    model = Sequential()
    model.add(Dense(x.shape[1], input_dim=x.shape[1], activation='relu'))
    model.add(Dense(x.shape[1], activation='relu'))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=0, mode='auto')
    checkpointer = ModelCheckpoint(filepath=filepath + "/best_weights.hdf5", verbose=0, save_best_only=True)
    return model, monitor, checkpointer


def lstm_classifier(num_feat, classes, timesteps, batch_size, units_per_layer, layers, dropout, optimizer='adam'):
    model = Sequential()
    if type(dropout) is float:
        dropout = np.repeat(dropout, layers, axis=0)
    if type(units_per_layer) is int:
        units_per_layer = np.repeat(units_per_layer, layers, axis=0)
    for layer in range(layers):
        model.add(LSTM(
            units_per_layer[layer],
            dropout=dropout[layer], recurrent_dropout=dropout[layer],
            batch_input_shape=(batch_size, timesteps, num_feat),
            return_sequences=True, stateful=True
        ))
    model.add(Dense(classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
    return model


def build_param_model(x, y, filepath, param, dropout=0.1):
    batch_size = 10
    model = Sequential()
    diff = x.shape[2] - y.shape[2] if x.shape[2] > y.shape[2] else y.shape[2]
    model.add(LSTM(int(y.shape[2] + (diff * param)),
                   dropout=dropout, recurrent_dropout=0.1,
                   input_shape=x.shape[1:],
                   batch_input_shape=(batch_size, x.shape[1], x.shape[2]),
                   return_sequences=True
                   ))
    model.add(Dense(y.shape[2], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10, verbose=0, mode='auto')
    checkpointer = ModelCheckpoint(filepath=filepath + "/best_weights.hdf5", verbose=0, save_best_only=True)
    return model, monitor, checkpointer


def build_param_model_2a(x, y, filepath, units, dropout=0.1):
    batch_size = 10
    model = Sequential()
    diff = x.shape[2] - y.shape[2] if x.shape[2] > y.shape[2] else y.shape[2]
    model.add(LSTM(units,
                   dropout=dropout, recurrent_dropout=dropout,
                   input_shape=x.shape[1:],
                   batch_input_shape=(batch_size, x.shape[1], x.shape[2]),
                   return_sequences=True, stateful=True
                   ))
    model.add(LSTM(units,
                   dropout=dropout, recurrent_dropout=dropout,
                   input_shape=x.shape[1:],
                   batch_input_shape=(batch_size, x.shape[1], x.shape[2]),
                   return_sequences=True, stateful=True
                   ))
    model.add(Dense(y.shape[2], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10, verbose=0, mode='auto')
    checkpointer = ModelCheckpoint(filepath=filepath + "/best_weights2.hdf5", verbose=0, save_best_only=True)
    return model, monitor, checkpointer


def build_param_model_2(x, y, filepath, params, dropout=0.1):
    batch_size = 10
    model = Sequential()
    diff = x.shape[2] - y.shape[2] if x.shape[2] > y.shape[2] else y.shape[2]
    model.add(LSTM(int(y.shape[2] + (diff * params[0])),
                   dropout=dropout, recurrent_dropout=dropout,
                   input_shape=x.shape[1:],
                   batch_input_shape=(batch_size, x.shape[1], x.shape[2]),
                   return_sequences=True, stateful=True
                   ))
    model.add(LSTM(int(y.shape[2] + (diff * params[0])),
                   dropout=dropout, recurrent_dropout=dropout,
                   input_shape=x.shape[1:],
                   batch_input_shape=(batch_size, x.shape[1], x.shape[2]),
                   return_sequences=True, stateful=True
                   ))
    model.add(Dense(y.shape[2], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10, verbose=0, mode='auto')
    checkpointer = ModelCheckpoint(filepath=filepath + "/best_weights2.hdf5", verbose=0, save_best_only=True)
    return model, monitor, checkpointer


def build_param_model_3(x, y, filepath, params, dropout=0.1):
    batch_size = 10
    model = Sequential()
    diff = x.shape[2] - y.shape[2] if x.shape[2] > y.shape[2] else y.shape[2]
    model.add(LSTM(int(y.shape[2] + (diff * params[0])),
                   dropout=dropout, recurrent_dropout=dropout,
                   input_shape=x.shape[1:],
                   batch_input_shape=(batch_size, x.shape[1], x.shape[2]),
                   return_sequences=True, stateful=True
                   ))
    model.add(LSTM(int(y.shape[2] + (diff * params[0])),
                   dropout=dropout, recurrent_dropout=dropout,
                   input_shape=x.shape[1:],
                   batch_input_shape=(batch_size, x.shape[1], x.shape[2]),
                   return_sequences=True, stateful=True
                   ))
    model.add(LSTM(int(y.shape[2] + (diff * params[0])),
                   dropout=dropout, recurrent_dropout=dropout,
                   input_shape=x.shape[1:],
                   batch_input_shape=(batch_size, x.shape[1], x.shape[2]),
                   return_sequences=True
                   ))
    model.add(Dense(y.shape[2], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10, verbose=0, mode='auto')
    checkpointer = ModelCheckpoint(filepath=filepath + "/best_weights3.hdf5", verbose=0, save_best_only=True)
    return model, monitor, checkpointer


def build_param_model_4(x, y, filepath, params, dropout=0.1):
    batch_size = 10
    model = Sequential()
    diff = x.shape[2] - y.shape[2] if x.shape[2] > y.shape[2] else y.shape[2]
    model.add(LSTM(int(y.shape[2] + (diff * params[0])),
                   dropout=dropout, recurrent_dropout=dropout,
                   input_shape=x.shape[1:],
                   batch_input_shape=(batch_size, x.shape[1], x.shape[2]),
                   return_sequences=True, stateful=True
                   ))
    model.add(LSTM(int(y.shape[2] + (diff * params[0])),
                   dropout=dropout, recurrent_dropout=dropout,
                   input_shape=x.shape[1:],
                   batch_input_shape=(batch_size, x.shape[1], x.shape[2]),
                   return_sequences=True, stateful=True
                   ))
    model.add(LSTM(int(y.shape[2] + (diff * params[0])),
                   dropout=dropout, recurrent_dropout=dropout,
                   input_shape=x.shape[1:],
                   batch_input_shape=(batch_size, x.shape[1], x.shape[2]),
                   return_sequences=True
                   ))
    model.add(LSTM(int(y.shape[2] + (diff * params[0])),
                   dropout=dropout, recurrent_dropout=dropout,
                   input_shape=x.shape[1:],
                   batch_input_shape=(batch_size, x.shape[1], x.shape[2]),
                   return_sequences=True
                   ))
    model.add(Dense(y.shape[2], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10, verbose=0, mode='auto')
    checkpointer = ModelCheckpoint(filepath=filepath + "/best_weights4.hdf5", verbose=0, save_best_only=True)
    return model, monitor, checkpointer


def build_model(x, y, filepath):
    batch_size = 10
    model = Sequential()
    model.add(LSTM(int(x.shape[2]),
                   dropout=0.1, recurrent_dropout=0.1,
                   input_shape=x.shape[1:],
                   batch_input_shape=(batch_size, x.shape[1], x.shape[2]),
                   return_sequences=True
                   ))
    model.add(Dense(y.shape[2], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=5, verbose=0, mode='auto')
    checkpointer = ModelCheckpoint(filepath=filepath + "/best_weights.hdf5", verbose=0, save_best_only=True)
    return model, monitor, checkpointer
