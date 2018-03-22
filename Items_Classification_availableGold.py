import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from sqlalchemy import *
from Lib.Classification.classifiers import *
from sklearn.metrics import classification as metrics
from pprint import pprint
from data.constants import *
from sklearn.model_selection import cross_val_score
from datetime import datetime as datetime
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


def get_data(col) -> pd.DataFrame:
    data = list(col.find())
    keys = data[1].keys()
    return pd.DataFrame(data, columns=keys)


names = ["Nearest Neighbors",
         # "Linear SVM",
         # "RBF SVM",
         # "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes"
         # "QDA"
         ]

classifiers = [
    KNeighborsClassifier(3),
    # SVC(kernel="linear", C=0.025), takes too long
    # SVC(gamma=2, C=1), takes too long
    # GaussianProcessClassifier(1.0 * RBF(1.0), n_jobs=-1), # needs too much memory
    DecisionTreeClassifier(),  # max_depth=5
    # n_jobs = -1 : number of processor cores
    RandomForestClassifier(n_jobs=-1),  # max_depth=5, n_estimators=10, max_features=1,
    MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
                  beta_2=0.999, early_stopping=False, epsilon=1e-08,
                  hidden_layer_sizes=(30, 30, 30), learning_rate='constant',
                  learning_rate_init=0.001, max_iter=200, momentum=0.9,
                  nesterovs_momentum=True, power_t=0.5, random_state=None,
                  shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
                  verbose=False, warm_start=False),
    AdaBoostClassifier(),
    GaussianNB()
    # QuadraticDiscriminantAnalysis()
]

engine = create_engine('mysql+pymysql://root:Milch4321@localhost:3306/leaguestats', encoding='latin1',
                       echo=False)
item_names = {key: value for (key, value) in list(engine.execute('SELECT * FROM itemkeys'))}

events = get_data(db.jhin_training_set)
frames = get_data(db.jhin_frames_test).drop(['_id'], axis=1)

platform_fact, platform_keys = pd.factorize(events.platformId)
events['platform_fact'] = platform_fact

item_fact, item_keys = pd.factorize(events.itemId)
events['item_fact'] = item_fact

type_fact, type_key = pd.factorize(events.type)
events['type_fact'] = type_fact

# patch_fact, patch_key = pd.factorize(events.patch)
# events['patch_fact'] = patch_fact

events = pd.merge(events, frames, how='inner', left_on=['gameId', 'participantId', 'platformId', 'frameNo'],
                  right_on=['gameId', 'participantId', 'platformId', 'frameNo'], copy=False)


item_infos = pd.read_sql("SELECT * FROM ITEMSTATS", engine)
item_infos["patch"] = item_infos["version"].str.extract("(\d.\d+)")
item_infos = item_infos.rename(columns={"key": "itemId"})

events = pd.merge(events, item_infos[["itemId", "patch", "goldSell", "goldBase", "goldTotal"]],
                  how="inner", on=["itemId", "patch"], copy=False)
events = events.drop(columns=["patch"])

# noinspection PyTypeChecker
events["Cost"] = np.where(events["type"] == "ITEM_PURCHASED", events["goldBase"], events["goldBase"] * -1)
# noinspection PyTypeChecker
events["Cost"] = np.where(events["type"] == "ITEM_SOLD", events["goldSell"] * -1, events["Cost"])

events['frameCost'] = events.groupby(['gameId', 'platformId', 'frameNo'])['Cost'].cumsum()
events['availGold'] = events['currentGold'] - \
                      events.groupby(['gameId', 'platformId', 'frameNo'])['frameCost'].shift(1).fillna(0)
events = events.drop(columns=["frameCost", "Cost", "goldBase", "goldSell", "goldTotal", "currentGold"])

events['is_train'] = np.random.uniform(0, 1, len(events)) <= .9
train, test = events[events['is_train']], events[~events['is_train']]

features = events.columns.difference(['_id', 'gameId', 'itemId', 'is_train', 'platformId', 'type', 'item_fact'])

run_classifiers("availableGold", events, features, item_names, item_keys)
