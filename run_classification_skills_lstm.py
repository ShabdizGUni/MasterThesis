import pathlib
import numpy as np
import pandas as pd
import sys
from datetime import datetime
from Lib.Classification import DataHandling as dh
from Lib.Classification import common
from Lib.Classification import ModelBuilder as mb
from Lib.Viz import plot_accuracy_dev, plot_loss_dev, save_conf_matrix
from data.constants import PATCHES
from sklearn.preprocessing import LabelBinarizer, RobustScaler
from sklearn.model_selection import train_test_split
from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import RMSprop, SGD
from sklearn.metrics import accuracy_score, confusion_matrix

# rmsp = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
rmsp = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)


print("Skill Level Ups LSTM started!")


def setup(name):
    path = "output/lstm_skills" + "/" + name
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    print("Start Processing: " + name)
    return path


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
                   batch_input_shape=(batch_size, x.shape[1], x.shape[2])
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


def to_xy_staggered(frame, feat, target='itemId', length=None, partition=None):
    if ~length or length == 0:
        '''
        Padding Mechanism
        '''
        x = np.array([np.pad(frame[feat].values.reshape(-1, ),
                             pad_width=(0, len(feat) * (mx - len(frame))),
                             mode='constant',
                             constant_values=0).reshape(-1, len(feat)) for _, frame in gb])

        y = np.array([np.pad(frame[target].values,
                             pad_width=(0, mx - len(frame)),
                             mode='constant',
                             constant_values=0)
                      for _, frame in gb])
        '''
        Reshape the resulting Array as a 3d Array considering the 'mx' timesteps
        '''
        y = lb.transform(y.reshape(-1, )).reshape(-1, mx, len(lb.classes_))
    if length > 0:
        x = gb.head(length)[feat]
        part = length if partition is None else partition
        x = scaler.transform(x).reshape(-1, part, len(feat))
        y = np.array([gb.head(length)[target]])
        y = lb.transform(y.reshape(-1, )).reshape(-1, part, len(lb.classes_))
    return x, y


cols_collection = [
    common.columns_blank_item,
    common.columns_pre_game,
    # spell level ups (in_game) do not make sense here
    list(np.setdiff1d(common.columns_inventory, ["q_rank", "w_rank", "e_rank", "r_rank"])),
    list(np.setdiff1d(common.columns_performance, ["q_rank", "w_rank", "e_rank", "r_rank"])),
    list(np.setdiff1d(common.columns_teams, ["q_rank", "w_rank", "e_rank", "r_rank"])),
    list(np.setdiff1d(common.columns_teams, ["q_rank", "w_rank", "e_rank", "r_rank"])) + common.item_columns
]


limit = int(sys.argv[1])
set_2 = [22, 51, 81, 110, 202]
data = dh.get_skills_teams(champions=set_2, patches=PATCHES, tiers=["CHALLENGER", "MASTER", "DIAMOND", "PLATINUM"],
                           limit=limit, timeseries=True, min_purch=15)
print("Rows : "+ str(len(data)))
for it, cols in enumerate(cols_collection):
    if 'availGold' in cols: cols.remove('availGold')
    cols.remove('type')
    cols.remove('itemId')
    cols.append('skillSlot')
    print('Features for Iteration: ' + str(it+1))
    print(cols)
    df = data.copy()[cols]
    print("Iterataion No.: " + str(it+1))
    if 'tier' in df.columns: _, tier_keys = dh.factorise_column(df, 'tier')
    if 'side' in df.columns: df['side'] = df['side'].astype(str)
    if 'masteryId' in df.columns: df['masteryId'] = df['masteryId'].astype(str)
    if 'championId' in df.columns: df['championId'] = df['championId'].astype(str)
    df = dh.one_hot_encode_columns(df, drop=False)

    lb = LabelBinarizer()  # Learn ItemIds
    itemset = list(df['skillSlot'].unique())
    itemset.append(0)  # padding ID
    lb.fit(itemset)
    item_names = {1: "Q", 2: "W", 3: "E", 4: "R"}

    gb = df.groupby(['gameId', 'side'])  # Create Groupby object

    mx = gb['side'].size().max()  # Find the largest group
    avg = gb['side'].size().mean()  # Find the largest group
    min = gb['side'].size().min()  # check if partitioning would work

    print('Maximum Amount of Items bought: %2.1f' % mx)
    print('Mean Amount of Items bought: %2.1f' % avg)
    print('Min Amount of Items bought: %2.1f' % min)

    scaler = RobustScaler()
    feat = df.columns.difference(
        ['_id', 'gameId', 'side', 'frameNo', 'skillSlot', 'participantId', 'platformId', 'tier', 'type', 'patch'])
    scaler.fit(df[feat])
    df_scale = scaler.transform(df[feat])

    x, y = to_xy_staggered(df_scale, feat, target='skillSlot', length=15)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    _, x_org, _, y_org = train_test_split(gb.head(15)[df.columns].as_matrix().reshape(-1, 15, len(gb.head(15)[df.columns].columns)), y, test_size=0.1,
                                          random_state=42)
    x_org = pd.DataFrame(data=x_org.reshape(-1, len(gb.head(15)[df.columns].columns)), columns=df.columns)

    print("TWO LAYERS ")
    start = datetime.now()
    path = setup(name="LSTM_2_iteration_" + str(it+1))
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=0, mode='auto')
    checkpointer = ModelCheckpoint(filepath=path + "/best_weights2b.hdf5", verbose=0, save_best_only=True)
    print("Shape X: (", str(x.shape[0]), ", ", str(x.shape[1]), ", ", str(x.shape[2]), ")")
    model = mb.lstm_classifier(num_feat=x.shape[2], classes=y.shape[2], timesteps=x.shape[1],
                               batch_size=5, units_per_layer=[x.shape[2], x.shape[2]], layers=2, dropout=0.2,
                               optimizer='adam')
    model.summary()
    history = model.fit(x_train, y_train, epochs=200, validation_split=0.1, callbacks=[monitor, checkpointer],
                        batch_size=5, verbose=False, shuffle=False).history
    try:
        plot_accuracy_dev(history['acc'], history['val_acc'], filepath=path + "/Accuracy2b.png", title="LSTM 2b")
        plot_loss_dev(history['loss'], history['val_loss'], filepath=path + "/Loss2b.png", title="LSTM 2b")
    except Exception as e:
        print("Plotting did not work!")
        print(str(e))
        pass
    model.load_weights(path + "/best_weights2b.hdf5")
    score, acc = model.evaluate(x_test, y_test, verbose=0, batch_size=5)
    print("Test accuracy: %2.3f" % acc)
    preds = model.predict(x_test, batch_size=5)
    preds_lab = lb.inverse_transform(preds.reshape(-1, len(lb.classes_)))
    y_test_lab = lb.inverse_transform(y_test.reshape(-1, len(lb.classes_)))
    preds_f = np.delete(preds_lab, np.argwhere(preds_lab == 0))
    y_test_f = np.delete(y_test_lab, np.argwhere(y_test_lab == 0))
    preds_f = [item_names[i] for i in preds_f]
    y_test_f = [item_names[i] for i in y_test_f]
    itemLabels = list(set().union(y_test_f, preds_f))
    cm = confusion_matrix(y_test_f, preds_f, labels=itemLabels)
    df_cm = pd.DataFrame(data=cm, index=itemLabels, columns=itemLabels)
    df_cm.to_csv(path + "/confusion_matrix_2b.csv", sep=";")
    save_conf_matrix(df_cm, path, normalize=True)

    acc = accuracy_score(y_test_f, preds_f)
    x_org.loc[:, 'two_layers_2b'] = preds_f
    x_org.to_csv("LSTM_skills_predictions.csv", sep=";")
    print("Test accuracy filtered: %2.3f" % acc)
    print("Finished TWO LAYERS B after: ", str(datetime.now() - start))


print("Skill Level Ups LSTM Finished!")
