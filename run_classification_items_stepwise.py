from data.constants import PATCHES
from Lib import util
import Lib.Classification.DataHandling as dh
import Lib.Classification.Classifiers_items as cl
import Lib.Classification.common as common
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import seaborn as sns
import numpy as np
import Lib.Classification.DataHandling as dh
import sys
import graphviz as gv
import time
import pathlib
from Lib.Viz import plot_confusion_matrix, plot_loss_dev, plot_accuracy_dev
from datetime import datetime as datetime
from sklearn import preprocessing as pp
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification as metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten
from keras.utils import plot_model
from sklearn.model_selection import KFold, cross_val_score
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, auc, recall_score, precision_score, f1_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib as mpl
from Lib.Classification import ModelBuilder as mb


exp_path = util.setup("output/stepwise")


def dnn_2(name, df):
    # if 'tier' in df.columns: _, tier_keys = dh.factorise_column(df, 'tier')
    if 'side' in df.columns: df['side'] = df['side'].astype(str)
    if 'masteryId' in df.columns: df['masteryId'] = df['masteryId'].astype(str)
    df['championId'] = df['championId'].astype(str)
    itemLabels = dh.encode_text_index(df, "itemId")
    x, y = dh.prepare_dataset_2(df)

    for idx, i in enumerate([2, 6, 11, 14]):
        name = 'Deep Learning 1' + ' ' + "Iteration " + str(idx)
        path = util.setup(exp_path + "/" + name)
        train = df.index[df.patch.isin(PATCHES[:i])].tolist()
        test = df.index[df.patch.isin(PATCHES[i:])].tolist()

        x_train, y_train = x[train], y[train]

        # x_train, _, y_train, _ = train_test_split(
        #     x[train], y[train], test_size=0.1, random_state=42
        # )

        _, x_test, _, y_test = train_test_split(
            x[test], y[test], test_size=0.1, random_state=42
        )

        start = datetime.now()
        model, monitor, checkpointer = mb.dnn_2_layers(x, y, path)
        history = model.fit(x_train, y_train, validation_split=0.33, callbacks=[monitor, checkpointer], verbose=0,
                            epochs=300).history
        plot_accuracy_dev(history['acc'], history['val_acc'], path + "/acc.png", title="Deep Learning 2")
        plot_loss_dev(history['loss'], history['val_loss'], path + "/loss.png", title="Deep Learning 2")
        model.load_weights(path + "/best_weights.hdf5")
        pred = model.predict(x_test)
        pred = np.argmax(pred, axis=1)
        y_test2 = np.argmax(y_test, axis=1)

        item_names = dh.get_item_dict()

        preds = [item_names[i] for i in itemLabels[[p for p in pred]]]
        actual = [item_names[i] for i in itemLabels[[p for p in y_test2]]]

        cm = confusion_matrix(actual, preds, labels=list(set().union(actual, preds)))
        df_cm = pd.DataFrame(data=cm, index=list(set().union(actual, preds)), columns=list(set().union(actual, preds)))
        df_cm.to_csv(path + "/confusion_matrix.csv", sep=";")
        plot_confusion_matrix(cm, names=list(set().union(actual, preds)))
        plt.savefig(path + "/confusion_matrix.png")
        plt.clf()

        precision = accuracy_score(actual, preds)
        print("Accuracy: " + str(precision))

        # define 10-fold cross validation test harness
        kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        cvscores = []
        for (train, _), (_, test) in zip(kfold.split(x[train], y[train]), kfold.split(x[test], y[test])):
            # create model
            # model, monitor, checkpointer = mb.dnn_2_layers(x, y, path)
            # Fit the model
            # model.fit(x[train], y[train], epochs=300, validation_split=0.33, verbose=0, callbacks=[monitor, checkpointer])
            # evaluate the model
            scores = model.evaluate(x[test], y[test], verbose=0)
            print("%s: %.2f%% , %s: %.2f%%" % (model.metrics_names[1], scores[1] * 100, model.metrics_names[0], scores[0]))
            cvscores.append(scores[1] * 100)
        print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
        time_sec = (datetime.now() - start).seconds
        print("Finished " + name + " after " + str(time_sec) + " seconds.")

# Ashe, Ezreal, Caitlyn, Varus, Jhin
Jhin = [202]
Ashe = [22]
Ezreal = [51]
Caitlyn = [81]
Varus = [110]
champions = [110]
limit = 100000
tiers = ["CHALLENGER", "MASTER", "DIAMOND", "PLATINUM"]

df = dh.get_purchase_teams(champions=champions, patches=PATCHES, tiers=tiers, limit=limit)
print(df.group_by(by=['patch', 'championId']).size())

dnn_2("stepwise", df[common.columns_teams])
