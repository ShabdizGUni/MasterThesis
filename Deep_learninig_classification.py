import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from data.constants import *
from sqlalchemy import *
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score


def plot_confusion_matrix(cm, names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Encode text values to indexes(i.e. [1],[2],[3] for red,green,blue).
def encode_text_index(df, name):
    le = preprocessing.LabelEncoder()
    df[name] = le.fit_transform(df[name])
    return le.classes_


# Convert a Pandas dataframe to the x,y inputs that TensorFlow needs
def to_xy(df, target):
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)
    # find out the type of the target column.  Is it really this hard? :(
    target_type = df[target].dtypes
    target_type = target_type[0] if hasattr(target_type, '__iter__') else target_type
    # Encode to int for classification, float otherwise. TensorFlow likes 32 bits.
    if target_type in (np.int64, np.int32):
        # Classification
        dummies = pd.get_dummies(df[target])
        return df.as_matrix(result).astype(np.float32), dummies.as_matrix().astype(np.float32)
    else:
        # Regression
        return df.as_matrix(result).astype(np.float32), df.as_matrix([target]).astype(np.float32)


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
# filter columns
features = events.columns.difference(['_id', 'gameId', 'frameNo', 'participantId', 'itemId', 'is_train', 'platformId',
                                      'type'])
f_cols = [x for x in events.columns if x in features]
events = events[f_cols + ['itemId']]

# path = "./data/"
#
# filename = os.path.join(path, "iris.csv")
# # df = pd.read_csv(filename, na_values=['NA', '?'])

df = events
species = encode_text_index(df, "itemId")
x, y = to_xy(df, "itemId")

# Split into train/test
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42)

# model = Sequential()
# model.add(Dense(120, input_dim=x.shape[1], activation='relu'))
# model.add(Dense(60))
# model.add(Dense(y.shape[1], activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam')

#LSTM
model = Sequential()
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, input_dim=1))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=0, mode='auto')
checkpointer = ModelCheckpoint(filepath="best_weights.hdf5", verbose=0, save_best_only=True)  # save best model
#model.fit(x_train, y_train, validation_data=(x_test, y_test), callbacks=[monitor, checkpointer], verbose=0, epochs=1000)

model.fit(x_train.reshape(x_train.shape[0], x_train.shape[1], 1), y_train, epochs=2)

model.load_weights('best_weights.hdf5')  # load weights from best model

#model.predict(x_test)
model.predict(x_test.reshape(x_test.shape[0], x_test.shape[1], 1))

# 10 k fold cross validation
estimator = KerasClassifier(build_fn=model, epochs=200, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, shuffle=True, random_state=31101990)
#results = cross_val_score(estimator, x_test, y_test, cv=kfold)
results = cross_val_score(estimator, x_test.reshape(x_test.shape[0], x_test.shape[1], 1), y_test, cv=kfold)
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

pred = model.predict(x_test.reshape(x_test.shape[0], x_test.shape[1], 1))
#pred = model.predict(x_test)
pred = np.argmax(pred, axis=1)
y_test2 = np.argmax(y_test, axis=1)

precision = accuracy_score(y_test2, pred)
print("Accuracy: " + str(precision))

# Compute confusion matrix
cm = confusion_matrix(y_test2, pred)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
plt.figure()
plot_confusion_matrix(cm, species)

# Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized, species, title='Normalized confusion matrix')

plt.show()

