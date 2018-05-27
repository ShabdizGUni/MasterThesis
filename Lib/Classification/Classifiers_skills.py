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
from Lib.Viz import save_conf_matrix, plot_loss_dev, plot_accuracy_dev
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
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
import matplotlib as mpl

mpl.rcParams.update(mpl.rcParamsDefault)
pd.set_option('display.width', 1000)

def save_test_result(path, x_test, actual, predicted):
    test_df = x_test.copy()
    test_df.reset_index(drop=True)
    test_df.loc[:, "actual"] = actual.tolist()
    test_df.loc[:, "predicted"] = predicted.tolist()
    test_df.to_csv(path + "/test_result.csv", sep=";")


def save_crosstab(path, actual, predicted):
    crosstab = pd.crosstab(actual, predicted, rownames=['actual'], colnames=['predicted'])
    crosstab.to_csv(path + "/crosstab.csv", sep=";")


def build_model1(x, y, filepath):
    model = Sequential()
    model.add(Dense(y.shape[1], input_dim=x.shape[1], activation='relu'))
    model.add(Dense(y.shape[1], input_dim=x.shape[1], activation='relu'))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=0, mode='auto')
    checkpointer = ModelCheckpoint(filepath=filepath + "/best_weights.hdf5", verbose=0, save_best_only=True)
    return model, monitor, checkpointer


def build_model2(x, y, filepath):
    model = Sequential()
    model.add(Dense(y.shape[1], input_dim=x.shape[1], activation='relu'))
    model.add(Dense(y.shape[1], input_dim=x.shape[1], activation='relu'))
    model.add(Dense(y.shape[1], input_dim=x.shape[1], activation='relu'))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=0, mode='auto')
    checkpointer = ModelCheckpoint(filepath=filepath + "/best_weights.hdf5", verbose=0, save_best_only=True)
    return model, monitor, checkpointer


class Classifier_skills:
    def __init__(self, exp_name, data):
        self.exp_name = exp_name
        self.data = data
        self.path = "output/skills/" + self.exp_name
        self.random_state = 42
        self.item_names = dh.get_item_dict()
        pathlib.Path(self.path).mkdir(parents=True, exist_ok=True)
        print("Experiment initlized")

    def setup(self, name):
        path = self.path + "/" + name
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        print("Start Processing: " + name)
        return path

    def run_naive_bayes(self, data):
        name = 'Naive Bayes'
        df = data.copy()
        item_names = dh.get_item_dict()
        _, platform_keys = dh.factorise_column(df, 'platformId')
        # _, item_keys = dh.factorise_column(df, 'skillSlot')
        if 'tier' in df.columns: _, tier_keys = dh.factorise_column(df, 'tier')
        if 'side' in df.columns: df['side'] = df['side'].astype('category')
        if 'masteryId' in df.columns: df['masteryId'] = df['masteryId'].astype('category')
        if 'championId' in df.columns:  df['championId'] = df['championId'].astype("category")
        features = df.columns.difference(
            ['_id', 'gameId', 'participantId', 'frameNo', 'skillSlot', 'platformId', 'tier', 'type', 'skillSlot', 'patch'])
        x, y = df[[x for x in df.columns if x in features]], df['skillSlot']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=self.random_state)
        start = datetime.now()
        path = self.setup(name)
        clf = GaussianNB()
        clf.fit(x_train, y_train)
        train_preds = clf.predict(x_train)
        test_preds = clf.predict(x_test)
        train_acc, test_acc = metrics.accuracy_score(y_train, train_preds), metrics.accuracy_score(
            y_test, test_preds)
        print("Train/Test Accuracy:  %.2f%% / %.2f%%" % (train_acc * 100,  test_acc * 100))

        kfold = KFold(n_splits=10, shuffle=True, random_state=self.random_state)
        results = cross_val_score(clf, x_test, y_test, cv=kfold)
        print("Mean Accuracy and St.Deviation: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

        spells = ["", "Q", "W", "E", "R"]
        actual = [spells[y] for y in y_test]
        predicted = [spells[p] for p in test_preds]

        # save_crosstab(path, actual, predicted)
        # save_test_result(path, x_test, actual, predicted)

        cnf_matrix = metrics.confusion_matrix(actual, predicted, labels=spells[1:])
        df_cm = pd.DataFrame(cnf_matrix, index=spells[1:], columns=spells[1:])
        df_cm.to_csv(path + "/confusion_matrix.csv", sep=";")
        save_conf_matrix(df_cm, path, normalize=True, figsize=(6, 4))

        time_sec = (datetime.now() - start).seconds
        print("Finished " + name + " after " + str(time_sec) + " seconds.")
        return predicted

    def run_decision_tree(self, data):
        df = data.copy()
        name = 'Decision Tree'
        if 'tier' in df.columns: _, tier_keys = dh.factorise_column(df, 'tier')
        if 'side' in df.columns: df['side'] = df['side'].astype('category')
        if 'masteryId' in df.columns: df['masteryId'] = df['masteryId'].astype('category')
        if 'championId' in df.columns:  df['championId'] = df['championId'].astype("category")
        _, platform_keys = dh.factorise_column(df, 'platformId')
        # _, item_keys = dh.factorise_column(df, 'skillSlot')
        features = df.columns.difference(
            ['_id', 'gameId', 'participantId',' frameNo', 'skillSlot', 'platformId', 'tier', 'type', 'skillSlot', 'patch'])
        # prepare data set
        x, y = df[[x for x in df.columns if x in features]], df['skillSlot']
        # Split into train/test
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

        start = datetime.now()
        path = self.setup(name)

        clf = DecisionTreeClassifier(
            max_depth=10,
            random_state=self.random_state)
        clf.fit(x_train, y_train)
        train_preds = clf.predict(x_train)
        test_preds = clf.predict(x_test)
        train_acc, test_acc = metrics.accuracy_score(y_train, train_preds), metrics.accuracy_score(
            y_test, test_preds)
        print("Train/Test Accuracy:  %.2f%% / %.2f%%" % (train_acc * 100,  test_acc * 100))
        tree.export_graphviz(clf, out_file=path + "tree.dot")
        dot_data = tree.export_graphviz(clf,
                                        feature_names=features,
                                        class_names=["1", "2", "3", "4"],
                                        out_file=None)
        # dot_data = tree.export_graphviz(clf, out_file=None)
        graph = gv.Source(dot_data, format="svg")
        graph.render(path + "/DecisionTree")

        importances = clf.feature_importances_
        feat_imp = pd.DataFrame(list(zip(x_train.columns, importances)), columns=["Feature_Name", "Importance"])
        feat_imp.to_csv(path + '/Feature_Importances.csv', sep=';')
        indices = np.argsort(importances)[::-1]
        indices = indices[:20]
        feat_num = len(indices) if len(indices) < 20 else 20
        # Print the feature ranking
        print("Feature ranking:")
        # for f in range(x_train.shape[1]):
        for f in range(feat_num):
            print("%d. feature %s (%f)" % (f + 1, features[indices[f]], importances[indices[f]]))

        # Plot the feature importances of the forest
        plt.clf()
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(feat_num), importances[indices],
                color="r", align="center")
        plt.xticks(range(feat_num), features[indices])
        plt.xlim([-1, 20])
        plt.xticks(rotation='vertical')
        plt.tight_layout()
        plt.savefig(path + '/feature_importance.svg')

        kfold = KFold(n_splits=10, shuffle=True, random_state=self.random_state)
        results = cross_val_score(clf, x_test, y_test, cv=kfold)
        print("Mean Accuracy and St.Deviation: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

        spells = ["", "Q", "W", "E", "R"]
        actual = [spells[y] for y in y_test]
        predicted = [spells[p] for p in test_preds]

        # save_crosstab(path, actual, predicted)
        # save_test_result(path, x_test, actual, predicted)

        cnf_matrix = metrics.confusion_matrix(actual, predicted, labels=spells[1:])
        df_cm = pd.DataFrame(cnf_matrix, index=spells[1:], columns=spells[1:])
        df_cm.to_csv(path + "/confusion_matrix.csv", sep=";")
        save_conf_matrix(df_cm, path, normalize=True, figsize=(6,4))

        time_sec = (datetime.now() - start).seconds
        print("Finished " + name + " after " + str(time_sec) + " seconds.")
        return predicted

    def run_random_forest(self, data):
        df = data.copy()
        name = 'Random Forest'
        if 'tier' in df.columns: _, tier_keys = dh.factorise_column(df, 'tier')
        if 'side' in df.columns: df['side'] = df['side'].astype('category')
        if 'masteryId' in df.columns: df['masteryId'] = df['masteryId'].astype('category')
        if 'championId' in df.columns:  df['championId'] = df['championId'].astype("category")
        _, platform_keys = dh.factorise_column(df, 'platformId')
        # _, item_keys = dh.factorise_column(df, 'skillSlot')
        features = df.columns.difference(
            ['_id', 'gameId', 'participantId', 'frameNo', 'skillSlot', 'platformId', 'tier', 'type', 'skillSlot',
             'patch'])
        # prepare data set
        x, y = df[[x for x in df.columns if x in features]], df['skillSlot']
        # Split into train/test
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=self.random_state)

        start = datetime.now()
        path = self.setup(name)

        clf = RandomForestClassifier(
            max_depth=10,
            random_state=self.random_state)
        clf.fit(x_train, y_train)
        train_preds = clf.predict(x_train)
        test_preds = clf.predict(x_test)
        train_acc, test_acc = metrics.accuracy_score(y_train, train_preds), metrics.accuracy_score(
            y_test, test_preds)
        print("Train/Test Accuracy:  %.2f%% / %.2f%%" % (train_acc * 100, test_acc * 100))

        importances = clf.feature_importances_
        std = np.std([tree.feature_importances_ for tree in clf.estimators_],
                     axis=0)
        feat_imp = pd.DataFrame(list(zip(x_train.columns, importances)), columns=["Feature_Name", "Importance"])
        feat_imp.to_csv(path + '/Feature_Importances.csv', sep=';')
        indices = np.argsort(importances)[::-1]
        indices = indices[:20]
        # Print the feature ranking
        print("Feature ranking:")

        # for f in range(x_train.shape[1]):
        feat_num = len(indices) if len(indices) < 20 else 20
        for f in range(feat_num):
            print("%d. feature %s (%f)" % (f + 1, features[indices[f]], importances[indices[f]]))

        # Plot the feature importances of the forest
        plt.clf()
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(feat_num), importances[indices],
                color="r", yerr=std[indices], align="center")
        plt.xticks(range(feat_num), features[indices])
        plt.xlim([-1, feat_num])
        plt.xticks(rotation='vertical')
        plt.tight_layout()
        plt.savefig(path + '/feature_importance.svg')

        kfold = KFold(n_splits=10, shuffle=True, random_state=self.random_state)
        results = cross_val_score(clf, x_test, y_test, cv=kfold)
        print("Mean Accuracy and St.Deviation: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

        spells = ["", "Q", "W", "E", "R"]
        actual = [spells[y] for y in y_test]
        predicted = [spells[p] for p in test_preds]

        # save_crosstab(path, actual, predicted)
        # save_test_result(path, x_test, actual, predicted)

        cnf_matrix = metrics.confusion_matrix(actual, predicted, labels=spells[1:])
        df_cm = pd.DataFrame(cnf_matrix, index=spells[1:], columns=spells[1:])
        df_cm.to_csv(path + "/confusion_matrix.csv", sep=";")
        save_conf_matrix(df_cm, path, normalize=True, figsize=(6, 4))

        time_sec = (datetime.now() - start).seconds
        print("Finished " + name + " after " + str(time_sec) + " seconds.")
        return predicted

    def run_neural_network(self, data):
        name = 'Neural Network'
        df = data.copy()
        if 'tier' in df.columns: _, tier_keys = dh.factorise_column(df, 'tier')
        if 'side' in df.columns: df['side'] = df['side'].astype(str)
        if 'masteryId' in df.columns: df['masteryId'] = df['masteryId'].astype(str)
        if 'championId' in df.columns:  df['championId'] = df['championId'].astype(str)
        dh.one_hot_encode_columns(df, ['side', 'tier', 'masteryId', 'championId', 'type', 'platformId'])
        # _, item_keys = dh.factorise_column(df, 'skillSlot')
        features = df.columns.difference(
            ['_id', 'gameId', 'skillSlot', 'frameNo', 'participantId', 'platformId', 'tier', 'type', 'skillSlot', 'patch'])
        # prepare data set
        y = df['skillSlot']
        x = df[[x for x in df.columns if x in features]]
        scaler = MinMaxScaler()
        scaler.fit(x)
        x_norm = scaler.transform(x)
        x_train, x_test, y_train, y_test = train_test_split(x_norm, y, test_size=0.1, random_state=self.random_state)

        start = datetime.now()
        path = self.setup(name)

        clf = MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
                            beta_2=0.999, early_stopping=True, epsilon=1e-08,
                            hidden_layer_sizes=(120,), learning_rate='adaptive',
                            learning_rate_init=0.001, max_iter=100, momentum=0.9,
                            nesterovs_momentum=True, power_t=0.5, random_state=self.random_state,
                            shuffle=False, solver='adam', tol=0.0001, validation_fraction=0.33,
                            verbose=False, warm_start=False)

        clf.fit(x_train, y_train)
        train_preds = clf.predict(x_train)
        test_preds = clf.predict(x_test)
        train_acc, test_acc = metrics.accuracy_score(y_train, train_preds), metrics.accuracy_score(
            y_test, test_preds)
        print("Train/Test Accuracy:  %.2f%% / %.2f%%" % (train_acc * 100, test_acc * 100))

        kfold = KFold(n_splits=10, shuffle=True, random_state=self.random_state)
        results = cross_val_score(clf, x_test, y_test, cv=kfold)
        print("Mean Accuracy and St.Deviation: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

        spells = ["", "Q", "W", "E", "R"]
        actual = [spells[y] for y in y_test]
        predicted = [spells[p] for p in test_preds]

        # save_crosstab(path, actual, predicted)
        # save_test_result(path, x_test, actual, predicted)

        cnf_matrix = metrics.confusion_matrix(actual, predicted, labels=spells[1:])
        df_cm = pd.DataFrame(cnf_matrix, index=spells[1:], columns=spells[1:])
        df_cm.to_csv(path + "/confusion_matrix.csv", sep=";")
        save_conf_matrix(df_cm, path, normalize=True, figsize=(6,4))

        time_sec = (datetime.now() - start).seconds
        print("Finished " + name + " after " + str(time_sec) + " seconds.")
        return predicted

    def run_deep_learning_1(self, data):
        name = 'Deep Learning 1'
        df = data.copy()
        # if 'tier' in df.columns: _, tier_keys = dh.factorise_column(df, 'tier')
        if 'side' in df.columns: df['side'] = df['side'].astype(str)
        if 'masteryId' in df.columns: df['masteryId'] = df['masteryId'].astype(str)
        if 'championId' in df.columns:  df['championId'] = df['championId'].astype(str)
        df['skillSlot'] = df['skillSlot'].astype(str)
        labels = dh.encode_text_index(df, "skillSlot")
        x, y = dh.prepare_dataset(df, target='skillSlot')
        # Split into train/test
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.1, random_state=self.random_state)

        start = datetime.now()
        path = self.setup(name)
        model1, monitor, checkpointer = build_model1(x, y, path)
        history = model1.fit(x_train, y_train, validation_split=0.33, callbacks=[monitor, checkpointer], verbose=0,
                             epochs=100).history
        plot_accuracy_dev(history['acc'], history['val_acc'], path + "/acc.png", title="Deep Learning 2")
        plot_loss_dev(history['loss'], history['val_loss'], path + "/loss.svg", title="Deep Learning 2")
        model1.load_weights(path + "/best_weights.hdf5")
        pred = model1.predict(x_test)
        pred = np.argmax(pred, axis=1)
        y_test2 = np.argmax(y_test, axis=1)

        spells = {"1": "Q", "2": "W", "3": "E", "4": "R"}
        predicted = [spells[i] for i in labels[[p for p in pred]]]
        actual = [spells[i] for i in labels[[p for p in y_test2]]]

        # save_crosstab(path, actual, predicted)
        # save_test_result(path, x_test, actual, predicted)

        cnf_matrix = metrics.confusion_matrix(actual, predicted, labels=list(spells.values()))
        df_cm = pd.DataFrame(cnf_matrix, index=list(spells.values()), columns=list(spells.values()))
        df_cm.to_csv(path + "/confusion_matrix.csv", sep=";")
        save_conf_matrix(df_cm, path, normalize=True, figsize=(6,4))

        precision = accuracy_score(actual, predicted)
        print("Accuracy: " + str(precision))

        # define 10-fold cross validation test harness
        kfold = KFold(n_splits=10, shuffle=True, random_state=self.random_state)
        cvscores = []
        for train, test in kfold.split(x, y):
            # create model
            model, monitor, checkpointer = build_model1(x, y, path)
            # Fit the model
            model.fit(x[train], y[train], epochs=100, validation_split=0.33, verbose=0, callbacks=[monitor, checkpointer])
            # evaluate the model
            scores = model.evaluate(x[test], y[test], verbose=0)
            print("%s: %.2f%% , %s: %.2f%%" % (model.metrics_names[1], scores[1] * 100, model.metrics_names[0], scores[0]))
            cvscores.append(scores[1] * 100)
        print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
        time_sec = (datetime.now() - start).seconds
        print("Finished " + name + " after " + str(time_sec) + " seconds.")
        return predicted

    def run_deep_learning_2(self, data):
        name = 'Deep Learning 2'
        df = data.copy()
        # if 'tier' in df.columns: _, tier_keys = dh.factorise_column(df, 'tier')
        if 'side' in df.columns: df['side'] = df['side'].astype(str)
        if 'masteryId' in df.columns: df['masteryId'] = df['masteryId'].astype(str)
        if 'championId' in df.columns: df['championId'] = df['championId'].astype(str)
        df['skillSlot'] = df['skillSlot'].astype(str)
        labels = dh.encode_text_index(df, "skillSlot")

        x, y = dh.prepare_dataset(df, 'skillSlot')
        # Split into train/test
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.1, random_state=self.random_state)

        start = datetime.now()
        path = self.setup(name)
        model1, monitor, checkpointer = build_model2(x, y, path)
        history = model1.fit(x_train, y_train, validation_split=0.33, callbacks=[monitor, checkpointer], verbose=0,
                             epochs=100).history
        plot_accuracy_dev(history['acc'], history['val_acc'], path + "/accuracy.svg", title="Deep Learning 2")
        plot_loss_dev(history['loss'], history['val_loss'], path + "/loss.svg", title="Deep Learning 2")
        model1.load_weights(path + "/best_weights.hdf5")
        pred = model1.predict(x_test)
        pred = np.argmax(pred, axis=1)
        y_test2 = np.argmax(y_test, axis=1)

        spells = {"1": "Q", "2": "W", "3": "E", "4": "R"}
        predicted = [spells[i] for i in labels[[p for p in pred]]]
        actual = [spells[i] for i in labels[[p for p in y_test2]]]

        # save_crosstab(path, actual, predicted)
        # save_test_result(path, x_test, actual, predicted)

        cnf_matrix = metrics.confusion_matrix(actual, predicted, labels=list(spells.values()))
        df_cm = pd.DataFrame(cnf_matrix, index=list(spells.values()), columns=list(spells.values()))
        df_cm.to_csv(path + "/confusion_matrix.csv", sep=";")
        save_conf_matrix(df_cm, path, normalize=True, figsize=(6, 4))

        precision = accuracy_score(actual, predicted)
        print("Accuracy: " + str(precision))

        # define 10-fold cross validation test harness
        kfold = KFold(n_splits=10, shuffle=True, random_state=self.random_state)
        cvscores = []
        for train, test in kfold.split(x, y):
            # create model
            model, monitor, checkpointer = build_model1(x, y, path)
            # Fit the model
            model.fit(x[train], y[train], epochs=100, validation_split=0.33, verbose=0, callbacks=[monitor, checkpointer])
            # evaluate the model
            scores = model.evaluate(x[test], y[test], verbose=0)
            print("%s: %.2f%% , %s: %.2f%%" % (model.metrics_names[1], scores[1] * 100, model.metrics_names[0], scores[0]))
            cvscores.append(scores[1] * 100)
        print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
        time_sec = (datetime.now() - start).seconds
        print("Finished " + name + " after " + str(time_sec) + " seconds.")
        return predicted

    def run_clfs(self):
        df = self.data.copy()
        # prepare data set
        # normalised and one hot encoded
        features = df.columns.difference(['_id', 'gameId', 'participantId', 'skillSlot', 'platformId', 'type', 'skillSlot', 'patch'])
        naive_bayes_pred = self.run_naive_bayes(df)
        decision_tree_pred = self.run_decision_tree(df)
        random_forest_pred = self.run_random_forest(df)
        neural_network_pred = self.run_neural_network(df)
        deep_learning_1_pred = self.run_deep_learning_1(df)
        deep_learning_2_pred = self.run_deep_learning_2(df)

        # Reconstruct testset from all methods and concatenate results
        x, y = df.drop(columns=['skillSlot']), df['skillSlot']
        _, x_test, _, y_test = train_test_split(x, y, test_size=0.1, random_state=self.random_state)
        x_test.reset_index(drop=True)
        x_test.loc[:, 'actual'] = y_test.tolist()
        x_test.loc[:, 'naive_bayes'] = naive_bayes_pred
        x_test.loc[:, 'decision_tree'] = decision_tree_pred
        x_test.loc[:, 'random_forest'] = random_forest_pred
        x_test.loc[:, 'neural_network'] = neural_network_pred
        x_test.loc[:, 'deep_learning_1'] = deep_learning_1_pred
        x_test.loc[:, 'deep_learning_2'] = deep_learning_2_pred
        x_test.to_csv(self.path + '/results.csv', sep=";")
        print("Finished!")
