import pathlib
import numpy as np
import pandas as pd
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
    common.columns_in_game,
    common.columns_inventory,
    common.columns_performance,
    common.columns_teams
]

set_1 = [110, 202]
set_2 = [22, 51, 81, 110, 202]
data = dh.get_skills_teams(champions=set_1, patches=PATCHES, tiers=["CHALLENGER", "MASTER", "DIAMOND", "PLATINUM"],
                           limit=1000, timeseries=True, min_purch=15)
for i, cols in enumerate(cols_collection):
    cols.remove('type')
    cols.remove('itemId')
    cols.append('skillSlot')
    df = data[cols]
    # if 'tier' in df.columns: _, tier_keys = dh.factorise_column(df, 'tier')
    if 'tier' in df.columns: _, tier_keys = dh.factorise_column(df, 'tier')
    if 'side' in df.columns: df['side'] = df['side'].astype(str)
    if 'masteryId' in df.columns: df['masteryId'] = df['masteryId'].astype(str)
    if 'championId' in df.columns: df['championId'] = df['championId'].astype(str)
    df = dh.one_hot_encode_columns(df, drop=False)
    feat = df.columns.difference(
        ['_id', 'gameId', 'frameNo', 'itemId', 'participantId', 'platformId', 'tier', 'type', 'itemId_fact', 'patch'])
    feat = feat.difference(common.columns_inventory)

    lb = LabelBinarizer()  # Learn ItemIds
    itemset = list(df['itemId'].unique())
    itemset.append(0)  # padding ID
    lb.fit(itemset)
    item_names = dh.get_item_dict()

    gb = df.groupby(['gameId', 'side'])  # Create Groupby object

    mx = gb['side'].size().max()  # Find the largest group
    avg = gb['side'].size().mean()  # Find the largest group
    min = gb['side'].size().min()  # check if partitioning would work

    print('Maximum Amount of Items bought: %2.1f' % mx)
    print('Mean Amount of Items bought: %2.1f' % avg)
    print('Min Amount of Items bought: %2.1f' % min)


    scaler = RobustScaler()
    scaler.fit(df[feat])
    df_scale = scaler.transform(df[feat])

    x, y = to_xy_staggered(df_scale, feat, length=15)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    _, x_org, _, y_org = train_test_split(gb.head(15)[df.columns].as_matrix().reshape(-1, 15, len(gb.head(15)[df.columns].columns)), y, test_size=0.1,
                                          random_state=42)
    x_org = pd.DataFrame(data=x_org.reshape(-1, len(gb.head(15)[df.columns].columns)), columns=df.columns)

    # print("ONE LAYER")
    # start = datetime.now()
    # path = setup(name="LSTM_1")
    # monitor = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10, verbose=0, mode='auto')
    # checkpointer = ModelCheckpoint(filepath=path + "/best_weights.hdf5", verbose=0, save_best_only=True)
    # model = mb.lstm_classifier(num_feat=x.shape[2], classes=y.shape[2], timesteps=x.shape[1],
    #                            batch_size=5, units_per_layer=[len(feat)], layers=1, dropout=0.2,
    #                            optimizer='adam')
    # history = model.fit(x_train, y_train, epochs=1000, validation_split=0.1, callbacks=[monitor, checkpointer],
    #                     batch_size=5, verbose=False, shuffle=False).history
    # try:
    #     plot_accuracy_dev(history['acc'], history['val_acc'], filepath=path + "/Accuracy1.png", title="LSTM 1")
    #     plot_loss_dev(history['loss'], history['val_loss'], filepath=path + "/Loss1.png", title="LSTM 1")
    # except Exception as e:
    #     print("Plotting did not work!")
    #     print(str(e))
    #     pass
    # model.load_weights(path + "/best_weights.hdf5")
    # score, acc = model.evaluate(x_test, y_test, verbose=0, batch_size=5)
    # print("Test accuracy: %2.3f" % acc)
    # preds = model.predict(x_test, batch_size=5)
    # preds_lab = lb.inverse_transform(preds.reshape(-1, len(lb.classes_)))
    # y_test_lab = lb.inverse_transform(y_test.reshape(-1, len(lb.classes_)))
    # preds_f = np.delete(preds_lab, np.argwhere(preds_lab == 0))
    # y_test_f = np.delete(y_test_lab, np.argwhere(y_test_lab == 0))
    # preds_f = [item_names[i] for i in preds_f]
    # y_test_f = [item_names[i] for i in y_test_f]
    # itemLabels = list(set().union(y_test_f, preds_f))
    # cm = confusion_matrix(y_test_f, preds_f, labels=itemLabels)
    # df_cm = pd.DataFrame(data=cm, index=itemLabels, columns=itemLabels)
    # df_cm.to_csv(path + "/confusion_matrix_1.csv", sep=";")
    # acc = accuracy_score(y_test_f, preds_f)
    #
    # x_org.reset_index(drop=True)
    # x_org.loc[:, 'actual'] = y_test_f
    # x_org.loc[:, 'one_layer'] = preds_f
    #
    # print("Test accuracy filtered: %2.3f" % acc)
    # print("Finished ONE LAYER after: ", str(datetime.now() - start))
    #
    # print("TWO LAYERS A")
    # start = datetime.now()
    # path = setup(name="LSTM_2a")
    # monitor = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10, verbose=0, mode='auto')
    # checkpointer = ModelCheckpoint(filepath=path + "/best_weights2a.hdf5", verbose=0, save_best_only=True)
    # print("Shape X: (", str(x.shape[0]), ", ", str(x.shape[1]), ", ", str(x.shape[2]), ")")
    # # model, monitor, checkpointer = build_param_model_2a(x, y, path, units=units, dropout=0.3)
    # model = mb.lstm_classifier(num_feat=x.shape[2], classes=y.shape[2], timesteps=x.shape[1],
    #                            batch_size=5, units_per_layer=[128, 64], layers=2, dropout=0.2,
    #                            optimizer='adam')
    # model.summary()
    # history = model.fit(x_train, y_train, epochs=1000, validation_split=0.1, callbacks=[monitor, checkpointer],
    #                     batch_size=5, verbose=False, shuffle=False).history
    # try:
    #     plot_accuracy_dev(history['acc'], history['val_acc'], filepath=path + "/Accuracy2a.png", title="LSTM 2a")
    #     plot_loss_dev(history['loss'], history['val_loss'], filepath=path + "/Loss2a.png", title="LSTM 2a")
    # except Exception as e:
    #     print("Plotting did not work!")
    #     print(str(e))
    #     pass
    # model.load_weights(path + "/best_weights2a.hdf5")
    # score, acc = model.evaluate(x_test, y_test, verbose=0, batch_size=5)
    # print("Test accuracy: %2.3f" % acc)
    # preds = model.predict(x_test, batch_size=5)
    # preds_lab = lb.inverse_transform(preds.reshape(-1, len(lb.classes_)))
    # y_test_lab = lb.inverse_transform(y_test.reshape(-1, len(lb.classes_)))
    # preds_f = np.delete(preds_lab, np.argwhere(preds_lab == 0))
    # y_test_f = np.delete(y_test_lab, np.argwhere(y_test_lab == 0))
    # preds_f = [item_names[i] for i in preds_f]
    # y_test_f = [item_names[i] for i in y_test_f]
    # itemLabels = list(set().union(y_test_f, preds_f))
    # cm = confusion_matrix(y_test_f, preds_f, labels=itemLabels)
    # df_cm = pd.DataFrame(data=cm, index=itemLabels, columns=itemLabels)
    # df_cm.to_csv(path + "/confusion_matrix_2a.csv", sep=";")
    # acc = accuracy_score(y_test_f, preds_f)
    # x_org.loc[:, 'two_layers_2a'] = preds_f
    # print("Test accuracy filtered: %2.3f" % acc)
    # print("Finished TWO LAYERS A after: ", str(datetime.now() - start))

    print("TWO LAYERS ")
    start = datetime.now()
    path = setup(name="LSTM_2b")
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10, verbose=0, mode='auto')
    checkpointer = ModelCheckpoint(filepath=path + "/best_weights2b.hdf5", verbose=0, save_best_only=True)
    print("Shape X: (", str(x.shape[0]), ", ", str(x.shape[1]), ", ", str(x.shape[2]), ")")
    # model, monitor, checkpointer = build_param_model_2a(x, y, path, units=units, dropout=0.3)
    model = mb.lstm_classifier(num_feat=x.shape[2], classes=y.shape[2], timesteps=x.shape[1],
                               batch_size=5, units_per_layer=[x.shape[2], x.shape[2]], layers=2, dropout=0.2,
                               optimizer='adam')
    model.summary()
    history = model.fit(x_train, y_train, epochs=1000, validation_split=0.1, callbacks=[monitor, checkpointer],
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
    print("Test accuracy filtered: %2.3f" % acc)
    print("Finished TWO LAYERS B after: ", str(datetime.now() - start))


    # print("THREE LAYERS")
    # start = datetime.now()
    # path = setup(name="LSTM_3")
    # monitor = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10, verbose=0, mode='auto')
    # checkpointer = ModelCheckpoint(filepath=path + "/best_weights3.hdf5", verbose=0, save_best_only=True)
    # model = mb.lstm_classifier(num_feat=x.shape[2], classes=y.shape[2], timesteps=x.shape[1],
    #                            batch_size=5, units_per_layer=[128, 128, 64], layers=3, dropout=0.2,
    #                            optimizer='adam')
    # history = model.fit(x_train, y_train, epochs=1000, validation_split=0.1, callbacks=[monitor, checkpointer],
    #                     batch_size=5, verbose=False, shuffle=False).history
    # try:
    #     plot_accuracy_dev(history['acc'], history['val_acc'], filepath=path + "/Accuracy3.png", title="LSTM 3")
    #     plot_loss_dev(history['loss'], history['val_loss'], filepath=path + "/Loss3.png", title="LSTM 3")
    # except Exception as e:
    #     print("Plotting did not work!")
    #     print(str(e))
    #     pass
    # model.load_weights(path + "/best_weights3.hdf5")
    # score, acc = model.evaluate(x_test, y_test, verbose=0, batch_size=5)
    # print("Test accuracy: %2.3f" % acc)
    # preds = model.predict(x_test, batch_size=5)
    # preds_lab = lb.inverse_transform(preds.reshape(-1, len(lb.classes_)))
    # y_test_lab = lb.inverse_transform(y_test.reshape(-1, len(lb.classes_)))
    # preds_f = np.delete(preds_lab, np.argwhere(preds_lab == 0))
    # y_test_f = np.delete(y_test_lab, np.argwhere(y_test_lab == 0))
    # preds_f = [item_names[i] for i in preds_f]
    # y_test_f = [item_names[i] for i in y_test_f]
    # itemLabels = list(set().union(y_test_f, preds_f))
    # cm = confusion_matrix(y_test_f, preds_f, labels=itemLabels)
    # df_cm = pd.DataFrame(data=cm, index=itemLabels, columns=itemLabels)
    # df_cm.to_csv(path + "/confusion_matrix_3.csv", sep=";")
    # acc = accuracy_score(y_test_f, preds_f)
    # x_org.loc[:, 'three_layers'] = preds_f
    # print("Test accuracy filtered: %2.3f" % acc)
    # print("Finished THREE LAYERS after: ", str(datetime.now() - start))
    #
    # print("FOUR LAYERS")
    # start = datetime.now()
    # path = setup(name="LSTM_4")
    # monitor = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10, verbose=0, mode='auto')
    # checkpointer = ModelCheckpoint(filepath=path + "/best_weights4.hdf5", verbose=0, save_best_only=True)
    # print("Shape X: (", str(x.shape[0]), ", ", str(x.shape[1]), ", ", str(x.shape[2]), ")")
    # # model, monitor, checkpointer = build_param_model_2a(x, y, path, units=units, dropout=0.3)
    # model = mb.lstm_classifier(num_feat=x.shape[2], classes=y.shape[2], timesteps=x.shape[1],
    #                            batch_size=5, units_per_layer=[128, 128, 64, 64], layers=4, dropout=0.2,
    #                            optimizer='adam')
    # model.summary()
    # history = model.fit(x_train, y_train, epochs=1000, validation_split=0.1, callbacks=[monitor, checkpointer],
    #                     batch_size=5, verbose=False, shuffle=False).history
    # try:
    #     plot_accuracy_dev(history['acc'], history['val_acc'], filepath=path + "/Accuracy4.png", title="LSTM 4")
    #     plot_loss_dev(history['loss'], history['val_loss'], filepath=path + "/Loss4.png", title="LSTM 4")
    # except Exception as e:
    #     print("Plotting did not work!")
    #     print(str(e))
    #     pass
    # model.load_weights(path + "/best_weights4.hdf5")
    # score, acc = model.evaluate(x_test, y_test, verbose=0, batch_size=5)
    # print("Test accuracy: %2.3f" % acc)
    # preds = model.predict(x_test, batch_size=5)
    # preds_lab = lb.inverse_transform(preds.reshape(-1, len(lb.classes_)))
    # y_test_lab = lb.inverse_transform(y_test.reshape(-1, len(lb.classes_)))
    # preds_f = np.delete(preds_lab, np.argwhere(preds_lab == 0))
    # y_test_f = np.delete(y_test_lab, np.argwhere(y_test_lab == 0))
    # preds_f = [item_names[i] for i in preds_f]
    # y_test_f = [item_names[i] for i in y_test_f]
    # itemLabels = list(set().union(y_test_f, preds_f))
    # cm = confusion_matrix(y_test_f, preds_f, labels=itemLabels)
    # df_cm = pd.DataFrame(data=cm, index=itemLabels, columns=itemLabels)
    # df_cm.to_csv(path + "/confusion_matrix_4.csv", sep=";")
    # acc = accuracy_score(y_test_f, preds_f)
    # x_org.loc[:, 'four_layers'] = preds_f
    # print("Test accuracy filtered: %2.3f" % acc)
    # print("Finished FOUR LAYERS  after: ", str(datetime.now() - start))
    #
    # x_org.to_csv('lstm_items/results.csv', sep=";")