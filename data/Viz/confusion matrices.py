import numpy as np
import pandas as pd
from Lib.Viz import save_conf_matrix
from sklearn.metrics import confusion_matrix, accuracy_score


rslt = pd.read_csv("C:\\Users\\Shabdiz\\PycharmProjects\\MasterThesis\\output\\items\\Team Performance depending\\results.csv", sep=";")

# Varus
Varus = rslt[rslt.championId == 110]

actual = Varus.actual.tolist()
preds = Varus.deep_learning_1.tolist()

varus_acc_all = accuracy_score(actual, preds)

df_cm = pd.DataFrame(data= confusion_matrix(actual, preds, labels=list(set().union(actual, preds))),
                     columns=list(set().union(actual, preds)), index=list(set().union(actual, preds)))

save_conf_matrix(df_cm, path="Varus", normalize=True, name="Varus_all_season")

Varus_pre74 = Varus[Varus.patch.isin(["7.1", "7.2", "7.3"])]
actual_pre74 = Varus.actual.tolist()
preds_pre74 = Varus.deep_learning_1.tolist()
varus_acc_pre74 = accuracy_score(actual_pre74, preds_pre74)
print("Varus Acc pre patch 7.4: " + str(varus_acc_pre74))
df_cm = pd.DataFrame(data= confusion_matrix(actual_pre74, preds_pre74, labels=list(set().union(actual_pre74, preds_pre74))),
                     columns=list(set().union(actual_pre74, preds_pre74)), index=list(set().union(actual_pre74, preds_pre74)))
save_conf_matrix(df_cm, path="Varus", normalize=True, name="Varus_pre_74")


Varus_post711 = Varus[~Varus.patch.isin(["7.1", "7.2", "7.3", "7.4", "7.5", "7.6", "7.7", "7.8", "7.9", "7.10"])]
actual_post711 = Varus.actual.tolist()
preds_post711 = Varus.deep_learning_1.tolist()
varus_acc_post711 = accuracy_score(actual_post711, preds_post711)
print("Varus Acc post patch 7.11: " + str(varus_acc_post711))
df_cm = pd.DataFrame(data= confusion_matrix(actual_post711, preds_post711, labels=list(set().union(actual_post711, preds_post711))),
                     columns=list(set().union(actual_post711, preds_post711)), index=list(set().union(actual_post711, preds_post711)))
save_conf_matrix(df_cm, path="Varus", normalize=True, name="Varus_post_711")



# Ezreal
Ezreal = rslt[rslt.championId == 81]

actual = Ezreal.actual.tolist()
preds = Ezreal.deep_learning_1.tolist()

Ezreal_acc_all = accuracy_score(actual, preds)

df_cm = pd.DataFrame(data= confusion_matrix(actual, preds, labels=list(set().union(actual, preds))),
                     columns=list(set().union(actual, preds)), index=list(set().union(actual, preds)))

save_conf_matrix(df_cm, path="Ezreal", normalize=True, name="Ezreal_all_season")

Ezreal_pre74 = Ezreal[Ezreal.patch.isin(["7.1", "7.2", "7.3"])]
actual_pre74 = Ezreal.actual.tolist()
preds_pre74 = Ezreal.deep_learning_1.tolist()
Ezreal_acc_pre74 = accuracy_score(actual_pre74, preds_pre74)
print("Ezreal Acc pre patch 7.4: " + str(Ezreal_acc_pre74))
df_cm = pd.DataFrame(data= confusion_matrix(actual_pre74, preds_pre74, labels=list(set().union(actual_pre74, preds_pre74))),
                     columns=list(set().union(actual_pre74, preds_pre74)), index=list(set().union(actual_pre74, preds_pre74)))
save_conf_matrix(df_cm, path="Ezreal", normalize=True, name="Ezreal_pre_74")


Ezreal_post711 = Ezreal[~Ezreal.patch.isin(["7.1", "7.2", "7.3", "7.4", "7.5", "7.6", "7.7", "7.8", "7.9", "7.10"])]
actual_post711= Ezreal.actual.tolist()
preds_post711 = Ezreal.deep_learning_1.tolist()
Ezreal_acc_post711 = accuracy_score(actual_post711, preds_post711)
print("Ezreal Acc post patch 7.11: " + str(Ezreal_acc_post711))
df_cm = pd.DataFrame(data= confusion_matrix(actual_post711, preds_post711, labels=list(set().union(actual_post711, preds_post711))),
                     columns=list(set().union(actual_post711, preds_post711)), index=list(set().union(actual_post711, preds_post711)))
save_conf_matrix(df_cm, path="Ezreal", normalize=True, name="Ezreal_post_711") 




# Caitlyn
Caitlyn = rslt[rslt.championId == 51]

actual = Caitlyn.actual.tolist()
preds = Caitlyn.deep_learning_1.tolist()

Caitlyn_acc_all = accuracy_score(actual, preds)

df_cm = pd.DataFrame(data= confusion_matrix(actual, preds, labels=list(set().union(actual, preds))),
                     columns=list(set().union(actual, preds)), index=list(set().union(actual, preds)))

save_conf_matrix(df_cm, path="Caitlyn", normalize=True, name="Caitlyn_all_season")

Caitlyn_pre74 = Caitlyn[Caitlyn.patch.isin(["7.1", "7.2", "7.3"])]
actual_pre74 = Caitlyn.actual.tolist()
preds_pre74 = Caitlyn.deep_learning_1.tolist()
Caitlyn_acc_pre74 = accuracy_score(actual_pre74, preds_pre74)
print("Caitlyn Acc pre patch 7.4: " + str(Caitlyn_acc_pre74))
df_cm = pd.DataFrame(data= confusion_matrix(actual_pre74, preds_pre74, labels=list(set().union(actual_pre74, preds_pre74))),
                     columns=list(set().union(actual_pre74, preds_pre74)), index=list(set().union(actual_pre74, preds_pre74)))
save_conf_matrix(df_cm, path="Caitlyn", normalize=True, name="Caitlyn_pre_74")


Caitlyn_post711 = Caitlyn[~Caitlyn.patch.isin(["7.1", "7.2", "7.3", "7.4", "7.5", "7.6", "7.7", "7.8", "7.9", "7.10"])]
actual_post711= Caitlyn.actual.tolist()
preds_post711 = Caitlyn.deep_learning_1.tolist()
Caitlyn_acc_post711 = accuracy_score(actual_post711, preds_post711)
print("Caitlyn Acc post patch 7.11: " + str(Caitlyn_acc_post711))
df_cm = pd.DataFrame(data= confusion_matrix(actual_post711, preds_post711, labels=list(set().union(actual_post711, preds_post711))),
                     columns=list(set().union(actual_post711, preds_post711)), index=list(set().union(actual_post711, preds_post711)))
save_conf_matrix(df_cm, path="Caitlyn", normalize=True, name="Caitlyn_post_711") 


