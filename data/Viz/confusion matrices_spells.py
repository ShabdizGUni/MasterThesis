import data.constants as const
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import Lib.util as util
from Lib.Viz import save_conf_matrix
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from sklearn.metrics import roc_curve, auc

rslt = pd.read_csv("C:\\Users\\Shabdiz\\PycharmProjects\\MasterThesis\\output\\skills_vm\\In-Game Choices\\results.csv",
                   sep=";")
rslt['patch'] = rslt['patch'].astype(str)
spell_dict = {
    1: "Q",
    2: "W",
    3: "E",
    4: "R"
}

actual = [spell_dict[r] for r in rslt['actual']]
preds_nb = rslt['naive_bayes'].tolist()
preds_dt = rslt['decision_tree'].tolist()
preds_rf = rslt['random_forest'].tolist()
preds_dl1 = rslt['deep_learning_1'].tolist()
preds_dl2 = rslt['deep_learning_2'].tolist()

df_cm_nb = pd.DataFrame(data=confusion_matrix(actual, preds_nb, labels=list(set().union(actual, preds_nb))),
                        columns=list(set().union(actual, preds_nb)), index=list(set().union(actual, preds_nb)))
save_conf_matrix(df_cm_nb, name='naive_bayes', path=util.setup("ConfigMatrices_Skills"), normalize=True)

df_cm_dt = pd.DataFrame(data=confusion_matrix(actual, preds_dt, labels=list(set().union(actual, preds_dt))),
                        columns=list(set().union(actual, preds_dt)), index=list(set().union(actual, preds_dt)))
save_conf_matrix(df_cm_nb, name='decision_tree', path=util.setup("ConfigMatrices_Skills"), normalize=True)

df_cm_rf = pd.DataFrame(data=confusion_matrix(actual, preds_rf, labels=list(set().union(actual, preds_rf))),
                        columns=list(set().union(actual, preds_rf)), index=list(set().union(actual, preds_rf)))
save_conf_matrix(df_cm_nb, name='random_forest', path=util.setup("ConfigMatrices_Skills"), normalize=True)

df_cm_dl1 = pd.DataFrame(data=confusion_matrix(actual, preds_dl1, labels=list(set().union(actual, preds_dl1))),
                         columns=list(set().union(actual, preds_dl1)), index=list(set().union(actual, preds_dl1)))
save_conf_matrix(df_cm_nb, name='deep_learning_1', path=util.setup("ConfigMatrices_Skills"), normalize=True)

df_cm_dl2 = pd.DataFrame(data=confusion_matrix(actual, preds_dl2, labels=list(set().union(actual, preds_dl2))),
                         columns=list(set().union(actual, preds_dl2)), index=list(set().union(actual, preds_dl2)))
save_conf_matrix(df_cm_nb, name='deep_learning_2', path=util.setup("ConfigMatrices_Skills"), normalize=True)

# accuaracy scores:
nb_acc = accuracy_score(actual, preds_nb)
print("Naive Bayes Acc: " + str(nb_acc))
dt_acc = accuracy_score(actual, preds_dt)
print("Decision Tree Acc: " + str(dt_acc))
rf_acc = accuracy_score(actual, preds_rf)
print("Random Forest Acc: " + str(rf_acc))
dl1_acc = accuracy_score(actual, preds_dl1)
print("Deep Learning 1 Acc: " + str(dl1_acc))
dl2_acc = accuracy_score(actual, preds_dl2)
print("Deep Learning 2 Acc: " + str(dl2_acc))

champions_dict = {
    22: "Ashe",
    51: "Caitlyn",
    81: "Ezreal",
    110: "Varus",
    202: "Jhin"
}
champions = [22, 51, 110, 202]
for c in champions:
    actual = [spell_dict[r] for r in rslt[rslt.championId == c]['actual']]
    preds_dt = rslt[rslt.championId == c]['decision_tree'].tolist()
    df_cm_dt_c = pd.DataFrame(data=confusion_matrix(actual, preds_dt, labels=list(set().union(actual, preds_dt))),
                              columns=list(set().union(actual, preds_dt)), index=list(set().union(actual, preds_dt)))
    save_conf_matrix(df_cm_dt_c, name='decision_tree ' + champions_dict[c], path=util.setup("ConfigMatrices_Skills"),
                     normalize=True)

Varus_post_actual = [spell_dict[r] for r in
                     rslt[(rslt.championId == 110) & (rslt.patch.isin(["7.14", "7.15", "7.16", "7.17", "7.18"]))][
                         'actual'].tolist()]
Varus_post_pred = rslt[(rslt.championId == 110) & (rslt.patch.isin(["7.14", "7.15", "7.16", "7.17", "7.18"]))][
    'decision_tree'].tolist()

Varus_pre_actual = [spell_dict[r] for r in
                    rslt[(rslt.championId == 110) & (rslt.patch.isin(["7.1", "7.2", "7.3", "7.4"]))][
                        'actual'].tolist()]
Varus_pre_pred = rslt[(rslt.championId == 110) & (rslt.patch.isin(["7.1", "7.2", "7.3", "7.4"]))][
    'decision_tree'].tolist()

df_cm_v_post = pd.DataFrame(data=confusion_matrix(Varus_post_actual, Varus_post_pred,
                                                  labels=list(set().union(Varus_post_actual, Varus_post_pred))),
                            columns=list(set().union(Varus_post_actual, Varus_post_pred)),
                            index=list(set().union(Varus_post_actual, Varus_post_pred)))
dl2_acc_v_post = accuracy_score(Varus_post_actual, Varus_post_pred)
print("Varus Lethality Era: " + str(dl2_acc_v_post))
save_conf_matrix(df_cm_v_post, name='decision_tree Varus post 7.14', path=util.setup("ConfigMatrices_Skills"),
                 normalize=True, figsize=(5,3))

df_cm_v_pre = pd.DataFrame(
    data=confusion_matrix(Varus_pre_actual, Varus_pre_pred, labels=list(set().union(Varus_pre_actual, Varus_pre_pred))),
    columns=list(set().union(Varus_pre_actual, Varus_pre_pred)),
    index=list(set().union(Varus_pre_actual, Varus_pre_pred)))

dl2_acc_v_pre = accuracy_score(Varus_pre_actual, Varus_pre_pred)
print("Varus Lethality Era: " + str(dl2_acc_v_pre))
save_conf_matrix(df_cm_v_pre, name='decision_tree Varus pre 7.14', path=util.setup("ConfigMatrices_Skills"),
                 normalize=True, figsize=(5,3))
