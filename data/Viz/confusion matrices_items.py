import data.constants as const
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import Lib.util as util
from Lib.Viz import save_conf_matrix
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from sklearn.metrics import roc_curve,auc


rslt = pd.read_csv("C:\\Users\\Shabdiz\\PycharmProjects\\MasterThesis\\output\\items\\Patch Context\\results.csv", sep=";")
rslt['patch'] = rslt['patch'].astype(str)
# all champions

actual = rslt['actual']
preds = rslt['deep_learning_2']

#
# print("deep_learning_2 Accuracy: " + str(accuracy_score(actual, preds)))
# print("deep_learning_2 Recall: " + str(recall_score(actual, preds)))
# print("deep_learning_2 Precision: " + str(precision_score(actual, preds)))
# df_cm = pd.DataFrame(data=confusion_matrix(actual, preds, labels=list(set().union(actual, preds))),
#                      columns=list(set().union(actual, preds)), index=list(set().union(actual, preds)))
# save_conf_matrix(df_cm, name='Deep Learning 2', path=util.setup("ConfigMatrices"), normalize=True)
#
# preds = rslt['neural_network']
# print("neural_network Accuracy: " + str(accuracy_score(actual, preds)))
# print("neural_network Recall: " + str(recall_score(actual, preds)))
# print("neural_network Precision: " + str(precision_score(actual, preds)))
# save_conf_matrix(df_cm, name='Deep Learning 1', path=util.setup("ConfigMatrices"), normalize=True)
#
#
# preds = rslt['deep_learning_1']
# print("Deep Learning 1 Accuracy: " + str(accuracy_score(actual, preds)))
# print("Deep Learning 1 Recall: " + str(recall_score(actual, preds)))
# print("Deep Learning 1 Precision: " + str(precision_score(actual, preds)))
# save_conf_matrix(df_cm, name='Deep Learning 1', path=util.setup("ConfigMatrices"), normalize=True)
#
# preds = rslt['decision_tree']
# print("decision_tree Accuracy: " + str(accuracy_score(actual, preds)))
# print("decision_tree Recall: " + str(recall_score(actual, preds)))
# print("decision_treePrecision: " + str(precision_score(actual, preds)))
# save_conf_matrix(df_cm, name='Decision Trees', path=util.setup("ConfigMatrices"), normalize=True)
#
# preds = rslt['random_forest']
# print("random_forest Accuracy: " + str(accuracy_score(actual, preds)))
# print("random_forest Recall: " + str(recall_score(actual, preds)))
# print("random_forestPrecision: " + str(precision_score(actual, preds)))



ids = [22, 51, 81, 110, 202]
champ_dict = {
    22: "Ashe",
    51: "Caitlyn",
    81: "Ezreal",
    110: "Varus",
    202: "Jhin"
}

preds = rslt.deep_learning_2.tolist()
actual = rslt.actual.tolist()
acc = accuracy_score(actual, preds)
print("Total Accuracy: " + str(acc))

accuracy_over_time = pd.DataFrame(data={
    "patches": const.PATCHES * 5,
    "champion": [champ_dict[item] for item in ids for i in range(20)],
    "acc": 0
    },
    columns=["patches", "champion", "acc"]
)
for id in ids:
    for patch in const.PATCHES:
        aux = rslt[(rslt.championId == id) & (rslt.patch == patch)]
        preds = aux.deep_learning_2.tolist()
        actual = aux.actual.tolist()
        acc = 0 if math.isnan(accuracy_score(actual, preds)) else accuracy_score(actual, preds)
        accuracy_over_time.loc[(accuracy_over_time.patches == patch) & (accuracy_over_time.champion == champ_dict[id]), "acc"] = acc
        print("Champion : " + str(id) + " accuracy: " + str(acc))

plt.clf()
plt.figure()
patches_b = const.PATCHES
patches_b.remove("7.10")
patches_b.remove("7.1")
accuracy_over_time['acc'] = accuracy_over_time['acc']*100
ax = sns.pointplot(x="patches", y="acc", hue="champion", order=patches_b, data=accuracy_over_time)
ax.set(ylim=(0, 100))
ax.legend(loc='lower left', #bbox_to_anchor=(0.5, 1.05),
          fancybox=True, shadow=True, ncol=2)
plt.ylabel("Test Accuracy in %")
ax.set_xticklabels(rotation=90, labels=patches_b)
plt.tight_layout()
plt.savefig("test_tips.png")

#Ashe
Ashe = rslt[rslt.championId == 22]
actual = Ashe.actual.tolist()
preds = Ashe.deep_learning_2.tolist()
Ashe_acc_all = accuracy_score(actual, preds)

df_cm = pd.DataFrame(data= confusion_matrix(actual, preds, labels=list(set().union(actual, preds))),
                     columns=list(set().union(actual, preds)), index=list(set().union(actual, preds)))

save_conf_matrix(df_cm, path="Ashe", normalize=True, name="Ashe_all_season")

Ashe_pre74 = Ashe[Ashe.patch.isin(["6.23", "6.24", "7.1", "7.2", "7.3", "7.4", "7.5", "7.6", "7.7", "7.8"])]
actual_pre74 = Ashe_pre74.actual.tolist()
preds_pre74 = Ashe_pre74.deep_learning_2.tolist()
Ashe_acc_pre74 = accuracy_score(actual_pre74, preds_pre74)
print("Ashe Acc pre patch 7.4: " + str(Ashe_acc_pre74))
df_cm = pd.DataFrame(data= confusion_matrix(actual_pre74, preds_pre74, labels=list(set().union(actual_pre74, preds_pre74))),
                     columns=list(set().union(actual_pre74, preds_pre74)), index=list(set().union(actual_pre74, preds_pre74)))
save_conf_matrix(df_cm, path="Ashe", normalize=True, name="Ashe_pre_74")


Ashe_post711 = Ashe[~Ashe.patch.isin(["7.8", "7.9", "7.10", "7.11", "7.12", "7.13", "7.14", "7.15", "7.16", "7.17", "7.18"])]
actual_post711 = Ashe_post711.actual.tolist()
preds_post711 = Ashe_post711.deep_learning_2.tolist()
Ashe_acc_post711 = accuracy_score(actual_post711, preds_post711)
print("Ashe Acc post patch 7.11: " + str(Ashe_acc_post711))
df_cm = pd.DataFrame(data= confusion_matrix(actual_post711, preds_post711, labels=list(set().union(actual_post711, preds_post711))),
                     columns=list(set().union(actual_post711, preds_post711)), index=list(set().union(actual_post711, preds_post711)))
save_conf_matrix(df_cm, path="Ashe", normalize=True, name="Ashe_post_711")




# Ezreal
Ezreal = rslt[rslt.championId == 81]
actual = Ezreal.actual.tolist()
preds = Ezreal.deep_learning_2.tolist()
Ezreal_acc_all = accuracy_score(actual, preds)

df_cm = pd.DataFrame(data= confusion_matrix(actual, preds, labels=list(set().union(actual, preds))),
                     columns=list(set().union(actual, preds)), index=list(set().union(actual, preds)))

save_conf_matrix(df_cm, path="Ezreal", normalize=True, name="Ezreal_all_season")

Ezreal_pre74 = Ezreal[Ezreal.patch.isin(["6.23", "6.24", "7.1", "7.2", "7.3", "7.4", "7.5", "7.6", "7.7", "7.8"])]
actual_pre74 = Ezreal_pre74.actual.tolist()
preds_pre74 = Ezreal_pre74.deep_learning_2.tolist()
Ezreal_acc_pre74 = accuracy_score(actual_pre74, preds_pre74)
print("Ezreal Acc pre patch 7.4: " + str(Ezreal_acc_pre74))
df_cm = pd.DataFrame(data= confusion_matrix(actual_pre74, preds_pre74, labels=list(set().union(actual_pre74, preds_pre74))),
                     columns=list(set().union(actual_pre74, preds_pre74)), index=list(set().union(actual_pre74, preds_pre74)))
save_conf_matrix(df_cm, path="Ezreal", normalize=True, name="Ezreal_pre_74")


Ezreal_post711 = Ezreal[~Ezreal.patch.isin(["6.23", "6.24", "7.1", "7.2", "7.3", "7.4", "7.5", "7.6", "7.7", "7.8", "7.9", "7.10", "7.15", "7.16", "7.17", "7.18"])]
actual_post711 = Ezreal_post711.actual.tolist()
preds_post711 = Ezreal_post711.deep_learning_2.tolist()
Ezreal_acc_post711 = accuracy_score(actual_post711, preds_post711)
print("Ezreal Acc post patch 7.11: " + str(Ezreal_acc_post711))
df_cm = pd.DataFrame(data= confusion_matrix(actual_post711, preds_post711, labels=list(set().union(actual_post711, preds_post711))),
                     columns=list(set().union(actual_post711, preds_post711)), index=list(set().union(actual_post711, preds_post711)))
save_conf_matrix(df_cm, path="Ezreal", normalize=True, name="Ezreal_post_711")


# Caitlyn
Caitlyn = rslt[rslt.championId == 51]
actual = Caitlyn.actual.tolist()
preds = Caitlyn.deep_learning_2.tolist()
Caitlyn_acc_all = accuracy_score(actual, preds)

df_cm = pd.DataFrame(data=confusion_matrix(actual, preds, labels=list(set().union(actual, preds))),
                     columns=list(set().union(actual, preds)), index=list(set().union(actual, preds)))

save_conf_matrix(df_cm, path="Caitlyn", normalize=True, name="Caitlyn_all_season")

Caitlyn_pre74 = Caitlyn[Caitlyn.patch.isin(["6.23", "6.24", "7.1", "7.2", "7.3", "7.4", "7.5", "7.6", "7.7", "7.8"])]
actual_pre74 = Caitlyn_pre74.actual.tolist()
preds_pre74 = Caitlyn_pre74.deep_learning_2.tolist()
Caitlyn_acc_pre74 = accuracy_score(actual_pre74, preds_pre74)
print("Caitlyn Acc pre patch 7.4: " + str(Caitlyn_acc_pre74))
df_cm74 = pd.DataFrame(data=confusion_matrix(actual_pre74, preds_pre74, labels=list(set().union(actual_pre74, preds_pre74))),
                     columns=list(set().union(actual_pre74, preds_pre74)), index=list(set().union(actual_pre74, preds_pre74)))
df_cm74.to_csv('Caitlyn_Conf_matrix_pre74.csv', sep=";")
save_conf_matrix(df_cm, path="Caitlyn", normalize=True, name="Caitlyn_pre_74")


Caitlyn_post711 = Caitlyn[~Caitlyn.patch.isin(["6.23", "6.24", "7.1", "7.2", "7.3", "7.4", "7.5", "7.6", "7.7", "7.8", "7.9", "7.10", "7.15", "7.16", "7.17", "7.18"])]
actual_post711 = Caitlyn_post711.actual.tolist()
preds_post711 = Caitlyn_post711.deep_learning_2.tolist()
Caitlyn_acc_post711 = accuracy_score(actual_post711, preds_post711)
print("Caitlyn Acc post patch 7.11: " + str(Caitlyn_acc_post711))
df_cm711 = pd.DataFrame(data=confusion_matrix(actual_post711, preds_post711, labels=list(set().union(actual_post711, preds_post711))),
                        columns=list(set().union(actual_post711, preds_post711)), index=list(set().union(actual_post711, preds_post711)))
df_cm74.to_csv('Caitlyn_Conf_matrix_pre74.csv', sep=";")
save_conf_matrix(df_cm, path="Caitlyn", normalize=True, name="Caitlyn_post_711")

# Jhin
Jhin = rslt[rslt.championId == 202]
actual = Jhin.actual.tolist()
preds = Jhin.deep_learning_2.tolist()
Jhin_acc_all = accuracy_score(actual, preds)

df_cm = pd.DataFrame(data= confusion_matrix(actual, preds, labels=list(set().union(actual, preds))),
                     columns=list(set().union(actual, preds)), index=list(set().union(actual, preds)))

save_conf_matrix(df_cm, path="Jhin", normalize=True, name="Jhin_all_season")

Jhin_pre74 = Jhin[Jhin.patch.isin(["6.23", "6.24", "7.1", "7.2", "7.3", "7.4", "7.5", "7.6", "7.7", "7.8"])]
actual_pre74 = Jhin_pre74.actual.tolist()
preds_pre74 = Jhin_pre74.deep_learning_2.tolist()
Jhin_acc_pre74 = accuracy_score(actual_pre74, preds_pre74)
print("Jhin Acc pre patch 7.4: " + str(Jhin_acc_pre74))
df_cm = pd.DataFrame(data= confusion_matrix(actual_pre74, preds_pre74, labels=list(set().union(actual_pre74, preds_pre74))),
                     columns=list(set().union(actual_pre74, preds_pre74)), index=list(set().union(actual_pre74, preds_pre74)))
save_conf_matrix(df_cm, path="Jhin", normalize=True, name="Jhin_pre_74")


Jhin_post711 = Jhin[~Jhin.patch.isin(["6.23", "6.24", "7.1", "7.2", "7.3", "7.4", "7.5", "7.6", "7.7", "7.8", "7.9", "7.10", "7.15", "7.16", "7.17", "7.18"])]
actual_post711 = Jhin_post711.actual.tolist()
preds_post711 = Jhin_post711.deep_learning_2.tolist()
Jhin_acc_post711 = accuracy_score(actual_post711, preds_post711)
print("Jhin Acc post patch 7.11: " + str(Jhin_acc_post711))
df_cm = pd.DataFrame(data= confusion_matrix(actual_post711, preds_post711, labels=list(set().union(actual_post711, preds_post711))),
                     columns=list(set().union(actual_post711, preds_post711)), index=list(set().union(actual_post711, preds_post711)))
save_conf_matrix(df_cm, path="Jhin", normalize=True, name="Jhin_post_711")





# Varus
Varus = rslt[rslt.championId == 110]
actual = Varus.actual.tolist()
preds = Varus.deep_learning_2.tolist()
varus_acc_all = accuracy_score(actual, preds)

df_cm = pd.DataFrame(data= confusion_matrix(actual, preds, labels=list(set().union(actual, preds))),
                     columns=list(set().union(actual, preds)), index=list(set().union(actual, preds)))

save_conf_matrix(df_cm, path="Varus", normalize=True, name="Varus_all_season")

actual_pre74 = Varus_pre74.actual.tolist()
preds_pre74 = Varus_pre74.deep_learning_2.tolist()
varus_acc_pre74 = accuracy_score(actual_pre74, preds_pre74)
print("Varus Acc pre patch 7.4: " + str(varus_acc_pre74))
df_cm = pd.DataFrame(data= confusion_matrix(actual_pre74, preds_pre74, labels=list(set().union(actual_pre74, preds_pre74))),
                     columns=list(set().union(actual_pre74, preds_pre74)), index=list(set().union(actual_pre74, preds_pre74)))
save_conf_matrix(df_cm, path="Varus", normalize=True, name="Varus_pre_74")


Varus_post711 = Varus[~Varus.patch.isin(["6.23", "6.24", "7.1", "7.2", "7.3", "7.4", "7.5", "7.6", "7.7", "7.8", "7.9", "7.10", "7.15", "7.16", "7.17", "7.18"])]
actual_post711 = Varus_post711.actual.tolist()
preds_post711 = Varus_post711.deep_learning_2.tolist()
varus_acc_post711 = accuracy_score(actual_post711, preds_post711)
print("Varus Acc post patch 7.11: " + str(varus_acc_post711))
df_cm_post711 = pd.DataFrame(data= confusion_matrix(actual_post711, preds_post711, labels=list(set().union(actual_post711, preds_post711))),
                     columns=list(set().union(actual_post711, preds_post711)), index=list(set().union(actual_post711, preds_post711)))
df_cm_post711.to_csv('Varus_Conf_matrix_post711.csv', sep=";")
save_conf_matrix(df_cm, path="Varus", normalize=True, name="Varus_post_711")

