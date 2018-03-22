import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import graphviz as gv
import time
import pathlib
from datetime import datetime as datetime
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification as metrics
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree as tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


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
    DecisionTreeClassifier(random_state=31101990),  # max_depth=5
    # n_jobs = -1 : number of processor cores
    RandomForestClassifier(n_jobs=-1, random_state=31101990),  # max_depth=5, n_estimators=10, max_features=1,
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


def run_classifiers(exp_name, events, features, item_names, item_keys):
    events['is_train'] = np.random.uniform(0, 1, len(events)) <= .9
    train, test = events[events['is_train']], events[~events['is_train']]

    for name, clf in zip(names, classifiers):
        start = datetime.now()

        path = "output_" + exp_name + "/" + name
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

        print("Start Processing: " + name)

        #sys.stdout = open("output_" + exp_name + "/" + 'console.txt', 'w')

        clf.fit(train[[x for x in events.columns if x in features]], train['item_fact'])

        preds = item_keys[clf.predict(test[[x for x in events.columns if x in features]])]

        precision = metrics.accuracy_score(test['itemId'], preds)
        print("Accuracy: " + str(precision))

        actual = pd.Series([item_names[i] for i in test['itemId']])
        predicted = pd.Series([item_names[p] for p in preds])

        test_df = events[~events['is_train']].copy()
        test_df.reset_index(drop=True)
        test_df.loc[:, "actual"] = actual.tolist()
        test_df.loc[:, "predicted"] = predicted.tolist()
        test_df.to_csv(path + "/test_result.csv", sep=";")

        crosstab = pd.crosstab(actual, predicted, rownames=['actual'], colnames=['predicted'])
        crosstab.to_csv(path + "/crosstab.csv", sep=";")
        cnf_matrix = metrics.confusion_matrix(actual, predicted, labels=[item_names[i] for i in item_keys])

        # Plot non-normalized confusion matrix
        df_cm = pd.DataFrame(cnf_matrix, index=[item_names[i] for i in item_keys],
                             columns=[item_names[i] for i in item_keys])
        df_cm.to_csv(path + "/confusion_matrix.csv", sep=";")
        plt.figure(figsize=(10, 7))
        sns.set(font_scale=0.25)
        sns_plot = sns.heatmap(df_cm).get_figure()
        sns_plot.savefig(path + "/confusion_matrix.pdf", format='pdf')
        # plt.savefig("output/confusion_matrix_" + name + ".pdf", format='pdf')

        # if name == "Decision Tree":
        #     # tree.export_graphviz(clf,out_file="output_" + exp_name + "/" + name + "_tree.dot")
        #     # dot_data = tree.export_graphviz(clf,
        #     #                                 feature_names=features,
        #     #                                 class_names=[item_names[i] for i in item_keys],
        #     #                                 out_file=None)
        #     dot_data = tree.export_graphviz(clf, out_file=None)
        #     graph = gv.Source(dot_data, format="svg")
        #     graph.render(path + "/DecisionTree")
        #     time.sleep(10)

        time_sec = (datetime.now() - start).seconds
        print("Finished " + name + " after " + str(time_sec) + " seconds.")


