import pathlib
import numpy as np
import pandas as pd
from datetime import datetime
from Lib.Classification import DataHandling as dh
from Lib.Classification import common
from Lib.Classification import ModelBuilder as mb
from Lib.Viz import plot_accuracy_dev, plot_loss_dev
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
    path = "test" + "/" + name
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


def to_xy(df, feat, length=None, partition=None):
    if ~length or length == 0:
        '''
        Padding Mechanism
        '''
        x = np.array([np.pad(frame[feat].values.reshape(-1, ),
                             pad_width=(0, len(feat) * (mx - len(frame))),
                             mode='constant',
                             constant_values=0).reshape(-1, len(feat)) for _, frame in gb])

        y = np.array([np.pad(frame['itemId'].values,
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
        y = np.array([gb.head(length)['itemId']])
        y = lb.transform(y.reshape(-1, )).reshape(-1, part, len(lb.classes_))
    return x, y


df = dh.get_purchases_performance(champions=[22, 51, 81, 110, 202], patches=PATCHES, tiers=["CHALLENGER", "MASTER", "DIAMOND"],
                                  limit=10000, timeseries=True, min_purch=15)
# if 'tier' in df.columns: _, tier_keys = dh.factorise_column(df, 'tier')
if 'side' in df.columns: df['side'] = df['side'].astype(str)
if 'masteryId' in df.columns: df['masteryId'] = df['masteryId'].astype(str)
df['championId'] = df['championId'].astype(str)
df = dh.one_hot_encode_columns(df, ['side', 'tier', 'masteryId', 'championId', 'type', 'platformId'])
feat = df.columns.difference(
    ['_id', 'gameId', 'side', 'participantId', 'platformId', 'tier', 'type', 'patch', 'masteryId', 'championId',
     'itemId', 'itemId_fact'])
feat = feat.difference(common.inventory)

lb = LabelBinarizer()  # Learn ItemIds
itemset = list(df['itemId'].unique())
itemset.append(0)  # padding ID
lb.fit(itemset)

gb = df.groupby(['gameId', 'side'])  # Create Groupby object

mx = gb['side'].size().max()  # Find the largest group
avg = gb['side'].size().mean()  # Find the largest group
min = gb['side'].size().min()   # check if partitioning would work

print('Maximum Amount of Items bought: %2.1f' % mx)
print('Mean Amount of Items bought: %2.1f' % avg)
print('Min Amount of Items bought: %2.1f' % min)

scaler = RobustScaler()
scaler.fit(df[feat])
df = scaler.transform(df[feat])

item_names = dh.get_item_dict()
x, y = to_xy(df, feat, length=15, partition=5)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

# for 1 Layer:
# for param in [0.5, 1, 1.5, 2, 2.5, 3]:
#     print("Try with Factor: ", str(param))
#     model, monitor, checkpointer = build_param_model(x, y, setup(name="LSTM"), param)
#     history = model.fit(x_train, y_train, epochs=1000, validation_split=0.1, callbacks=[monitor, checkpointer],
#                         batch_size=10, verbose=False).history
#     score, acc = model.evaluate(x_test, y_test, verbose=0, batch_size=10)
#     print("Test accuracy: %2.3f" % acc)
#     preds = model.predict(x_test, batch_size=10)
#     preds_lab = lb.inverse_transform(preds.reshape(-1, len(lb.classes_)))
#     y_test_lab = lb.inverse_transform(y_test.reshape(-1, len(lb.classes_)))
#     preds_f = np.delete(preds_lab, np.argwhere(preds_lab == 0))
#     y_test_f = np.delete(y_test_lab, np.argwhere(y_test_lab == 0))
#     item_names = dh.get_item_dict()
#     preds_f = [item_names[i] for i in preds_f]
#     y_test_f = [item_names[i] for i in y_test_f]
#     itemLabels = list(set().union(y_test_f, preds_f))
#
#     acc = accuracy_score(y_test_f, preds_f)
#     print("Test accuracy filtered: %2.3f" % acc)

# print("ONE LAYER")
# start = datetime.now()
# path = setup(name="LSTM_1")
# model, monitor, checkpointer = build_param_model(x, y, path, param=1, dropout=0.3)
# history = model.fit(x_train, y_train, epochs=1000, validation_split=0.1, callbacks=[monitor, checkpointer],
#                     batch_size=10, verbose=False, shuffle=False).history
# try:
#     plot_accuracy_dev(history['acc'], history['val_acc'], filepath=path + "/Accuracy1.png", title="LSTM 1")
#     plot_loss_dev(history['loss'], history['val_loss'], filepath=path + "/Loss1.png", title="LSTM 1")
# except Exception as e:
#     print("Plotting did not work!")
#     print(str(e))
#     pass
# model.load_weights(path + "/best_weights.hdf5")
# score, acc = model.evaluate(x_test, y_test, verbose=0, batch_size=10)
# print("Test accuracy: %2.3f" % acc)
# preds = model.predict(x_test, batch_size=10)
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
# print("Test accuracy filtered: %2.3f" % acc)
# print("Finished ONE LAYER after: ", str(datetime.now() - start))

for units in [100, 150, 200, 250]:
    print("TWO LAYERS")
    start = datetime.now()
    path = setup(name="LSTM_2")
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10, verbose=0, mode='auto')
    checkpointer = ModelCheckpoint(filepath=path + "/best_weights.hdf5", verbose=0, save_best_only=True)
    print("Shape X: (", str(x.shape[0]), ", ", str(x.shape[1]), ", ",  str(x.shape[2]), ")")
    # model, monitor, checkpointer = build_param_model_2a(x, y, path, units=units, dropout=0.3)
    model = mb.lstm_classifier(num_feat=x.shape[2], classes=y.shape[2], timesteps=x.shape[1],
                               batch_size=5, units_per_layer=100, layers=2, dropout=0.3,
                               optimizer='adam')
    model.summary()
    history = model.fit(x_train, y_train, epochs=1000, validation_split=0.1, callbacks=[monitor, checkpointer],
                        batch_size=5, verbose=False).history
    try:
        plot_accuracy_dev(history['acc'], history['val_acc'], filepath=path + "/Accuracy2.png", title="LSTM 2")
        plot_loss_dev(history['loss'], history['val_loss'], filepath=path + "/Loss2.png", title="LSTM 2")
    except Exception as e:
        print("Plotting did not work!")
        print(str(e))
        pass
    model.load_weights(path + "/best_weights2.hdf5")
    score, acc = model.evaluate(x_test, y_test, verbose=0, batch_size=1)
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
    df_cm.to_csv(path + "/confusion_matrix_2.csv", sep=";")
    acc = accuracy_score(y_test_f, preds_f)
    print("Test accuracy filtered: %2.3f" % acc)
    print("Finished TWO LAYERS after: ", str(datetime.now() - start))


# print("THREE LAYERS")
# start = datetime.now()
# path = setup(name="LSTM_3")
# model, monitor, checkpointer = build_param_model_3(x, y, path, params=[1, 0.5, 0.25], dropout=0.3)
# history = model.fit(x_train, y_train, epochs=1000, validation_split=0.1, callbacks=[monitor, checkpointer],
#                     batch_size=10, verbose=False, shuffle=False).history
# try:
#     plot_accuracy_dev(history['acc'], history['val_acc'], filepath=path + "/Accuracy3.png", title="LSTM 3")
#     plot_loss_dev(history['loss'], history['val_loss'], filepath=path + "/Loss3.png", title="LSTM 3")
# except Exception as e:
#     print("Plotting did not work!")
#     print(str(e))
#     pass
# model.load_weights(path + "/best_weights3.hdf5")
# score, acc = model.evaluate(x_test, y_test, verbose=0, batch_size=10)
# print("Test accuracy: %2.3f" % acc)
# preds = model.predict(x_test, batch_size=10)
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
# print("Test accuracy filtered: %2.3f" % acc)
# print("Finished THREE LAYERS after: ", str(datetime.now() - start))


# print("FOUR LAYERS")
# start = datetime.now()
# path = setup(name="LSTM_4")
# model, monitor, checkpointer = build_param_model_4(x, y, path, [1, 0.5, 0.25, 0.125], dropout=0.3)
# history = model.fit(x_train, y_train, epochs=1000, validation_split=0.1, callbacks=[monitor, checkpointer],
#                     batch_size=10, verbose=False, shuffle=False).history
# try:
#     plot_accuracy_dev(history['acc'], history['val_acc'], filepath=path + "/Accuracy4.png", title="LSTM 4")
#     plot_loss_dev(history['loss'], history['val_loss'], filepath=path + "/Loss4.png", title="LSTM 4")
# except Exception as e:
#     print("Plotting did not work!")
#     print(str(e))
#     pass
# model.load_weights(path + "/best_weights4.hdf5")
# score, acc = model.evaluate(x_test, y_test, verbose=0, batch_size=10)
# print("Test accuracy: %2.3f" % acc)
# preds = model.predict(x_test, batch_size=10)
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
# print("Test accuracy filtered: %2.3f" % acc)
# print("Finished FOUR LAYERS after: ", str(datetime.now() - start))
#
# print("Finished!")


# model =
# # try:
# #     plot_accuracy_dev(history['acc'], history['val_acc'], "")
# #     plot_loss_dev(history['loss'], history['val_loss'], "")
# # except Exception:
# #     print("Plotting did not work!")
# #     pass
#
# preds = model.predict(x_test, batch_size=10)
# preds = lb.inverse_transform(preds.reshape(-1, len(lb.classes_)))
# y_test = lb.inverse_transform(y_test.reshape(-1, len(lb.classes_)))
# preds_f = np.delete(preds, np.argwhere(preds == 0))
# y_test_f = np.delete(y_test, np.argwhere(y_test == 0))
# item_names = dh.get_item_dict()
# preds_f = [item_names[i] for i in preds_f]
# y_test_f = [item_names[i] for i in y_test_f]
# itemLabels = list(set().union(y_test_f, preds_f))
#
# acc = accuracy_score(y_test_f, preds_f)
# item_names = dh.get_item_dict()
# print(str(acc))
#
# cm = confusion_matrix(y_test_f, preds_f, labels=itemLabels)
# df_cm = pd.DataFrame(data=cm, index=list(set().union(y_test_f, preds_f)), columns=list(set().union(y_test_f, preds_f)))
# df_cm.to_csv("confusion_matrix.csv", sep=";")
#
# plot_confusion_matrix(cm, names=itemLabels)
# plt.savefig("confusion_matrix.png")
# plt.clf()
