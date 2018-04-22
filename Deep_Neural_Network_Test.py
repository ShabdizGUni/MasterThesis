import numpy as np
import pandas as pd
from data.constants import *
from sqlalchemy import *
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def get_data(col) -> pd.DataFrame:
    data = list(col.find())
    keys = data[1].keys()
    return pd.DataFrame(data, columns=keys)


# PREPARE DATA
engine = create_engine('mysql+pymysql://root:Milch4321@localhost:3306/leaguestats', encoding='latin1',
                       echo=False)
# item information
item_names = {key: value for (key, value) in list(engine.execute('SELECT * FROM itemkeys'))}

item_infos = pd.read_sql("SELECT * FROM ITEMSTATS", engine)
item_infos["patch"] = item_infos["version"].str.extract("(\d.\d+)")
item_infos = item_infos.rename(columns={"key": "itemId"})


# events and frames
events = get_data(db.jhin_training_set)
frames = get_data(db.jhin_frames_test).drop(['_id'], axis=1)

# factorize categorical columns
platform_fact, platform_keys = pd.factorize(events.platformId)
events['platform_fact'] = platform_fact

encoder = LabelEncoder()
encoder.fit(events['itemId'])

type_fact, type_key = pd.factorize(events.type)
events['type_fact'] = type_fact

# join events and item information
events = pd.merge(events, frames, how='inner', left_on=['gameId', 'participantId', 'platformId', 'frameNo'],
                  right_on=['gameId', 'participantId', 'platformId', 'frameNo'], copy=False)


events = pd.merge(events, item_infos[["itemId", "patch", "goldSell", "goldBase", "goldTotal"]],
                  how="inner", on=["itemId", "patch"], copy=False)
events = events.drop(columns=["patch"])

# calculate gold
# noinspection PyTypeChecker
events["Cost"] = np.where(events["type"] == "ITEM_PURCHASED", events["goldBase"], events["goldBase"] * -1)
# noinspection PyTypeChecker
events["Cost"] = np.where(events["type"] == "ITEM_SOLD", events["goldSell"] * -1, events["Cost"])

events['frameCost'] = events.groupby(['gameId', 'platformId', 'frameNo'])['Cost'].cumsum()
events['availGold'] = events['currentGold'] - \
                      events.groupby(['gameId', 'platformId', 'frameNo'])['frameCost'].shift(1).fillna(0)

# filter columns
events = events.drop(columns=["frameCost", "Cost", "goldBase", "goldSell", "goldTotal", "currentGold"])
Y = np_utils.to_categorical(events['itemId'])
features = events.columns.difference(['_id', 'gameId', 'frameNo', 'participantId', 'itemId', 'is_train', 'platformId',
                                      'type'])
f_cols = [x for x in events.columns if x in features]
events = events[f_cols + ['itemId']]
X = events[f_cols]


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=3110990)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=3110990)

# GET CLASSIFICATION ROLLING

data_dim = len(features)
timesteps = 1
num_classes = len(set(events['itemId']))
batch_size = 145


model = Sequential()
model.add(Dense(120, input_shape=x.shape[1], activation='relu'))
model.add(Dense(10))
model.add(Dense(y.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


#
# model = Sequential()
# model.add(LSTM(56, return_sequences=True, stateful=True,
#                batch_input_shape=(batch_size, timesteps, data_dim)))
# model.add(LSTM(56, return_sequences=True, stateful=True))
# model.add(Flatten())
# model.add(Dense(56, input_dim=56,  activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(X_train.values.reshape(X_train.shape[0], timesteps, data_dim), y_train,
          batch_size=batch_size, epochs=5, shuffle=False,
          validation_data=(X_train.values.reshape(X_train.shape[0], timesteps, data_dim), y_val))

model.predict(x_test, test_y)

# 10 k fold cross validation
estimator = KerasClassifier(build_fn=model, epochs=200, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, shuffle=True, random_state=31101990)
results = cross_val_score(estimator, events[f_cols], events['item_fact'], cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
