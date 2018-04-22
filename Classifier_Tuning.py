import numpy as np
import pandas as pd
import graphviz as gv
from data.constants import PATCHES
from Lib.Classification import DataHandling as dh
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import tree as tree
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten, Dropout


# important for replicability
random_state = 42
# Ashe, Ezreal, Caitlyn, Varus, Jhin
champions = [22, 51, 81, 110, 202]
limit = 200000
data = dh.get_purchases_in_game(champions=champions, patches=PATCHES, limit=limit)

# DECISION TREE:
#   parameters to tune:
#       Min Samples Leaf
#       min_samples_split
np.random.seed(random_state)
low = np.arange(0, 110, step=10)*10
high = np.arange(0, 110, step=10)*10
points = np.random.uniform(low=low, high=high, size=[2, 11]).T
np.random.shuffle(points[:, 1])
print(points)
points = pd.DataFrame(data=points)


for point in points.itertuples():
    df = data.copy()
    name = 'Decision Tree'
    item_names = dh.get_item_dict()
    if 'tier' in df.columns: _, tier_keys = dh.factorise_column(df, 'tier')
    if 'side' in df.columns: df['side'] = df['side'].astype('category')
    if 'masteryId' in df.columns: df['masteryId'] = df['masteryId'].astype('category')
    df['championId'] = df['championId'].astype('category')
    _, platform_keys = dh.factorise_column(df, 'platformId')
    _, type_keys = dh.factorise_column(df, 'type')
    _, item_keys = dh.factorise_column(df, 'itemId')
    features = df.columns.difference(
        ['_id', 'gameId', 'participantId', 'itemId', 'platformId', 'tier', 'type', 'itemId_fact', 'patch'])
    # prepare data set
    x, y = df[[x for x in df.columns if x in features]], df['itemId_fact']
    # Split into train/test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=random_state)
    clf = DecisionTreeClassifier(random_state=random_state,
                                 min_samples_leaf=int(point[1]) if point[1] != 0 else 1)
                                 #min_samples_split=point[2] if point[2] != 0.00 else 2)
    kfold = KFold(n_splits=10, shuffle=True, random_state=random_state)
    results = cross_val_score(clf, x_test, y_test, cv=kfold)
    print("Results for Config min_samples_leaf :" + str(point[1])) #+ " and min_samples_split: " + str(point[2]))
    print("Mean Accuracy and St.Deviation: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
    del df

# clf = DecisionTreeClassifier(random_state=random_state,
# #                              min_samples_leaf=0.001)
# # clf.fit(x_train, y_train)
# # dot_data = tree.export_graphviz(clf,
# #                                 feature_names=features,
# #                                 class_names=[item_names[i] for i in item_keys],
# #                                 out_file=None)
# # # dot_data = tree.export_graphviz(clf, out_file=None)
# # graph = gv.Source(dot_data, format="svg")
# # graph.render()