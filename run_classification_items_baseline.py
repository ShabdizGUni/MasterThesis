from data.constants import PATCHES
import Lib.Classification.DataHandling as dh
import Lib.Classification.Classifiers_items as cl
import Lib.Classification.common as common
import sys


print("Items Base started!")

limit = int(sys.argv[1])
# Ashe, Ezreal, Caitlyn, Varus, Jhin
champions = [22, 51, 81, 110, 202]
tiers = ["CHALLENGER", "MASTER", "DIAMOND", "PLATINUM"]

df = dh.get_purchase_teams(champions=champions, patches=PATCHES, tiers=tiers, limit=limit, timeseries=True, min_purch=10)


print("Number of Records: %d" % (len(df)))
print("_________________________________")
print('Features for Blank: ')
print(df.columns)
clf = cl.Classifier_items('Blank', df[common.columns_blank_item])
clf.run_clfs()

print('Features for Pre-Game Choices: ')
print(df[common.columns_pre_game].columns)
clf = cl.Classifier_items('Pre-Game Choices', df[common.columns_pre_game])
clf.run_clfs()

print('Features for In-Game Choices: ')
print(df[common.columns_in_game].columns)
clf = cl.Classifier_items('In-Game Choices', df[common.columns_in_game])
clf.run_clfs()

print('Features for "Inventory Choices": ')
print(df[common.columns_inventory].columns)
clf = cl.Classifier_items('Inventory Choices', df[common.columns_inventory])
clf.run_clfs()

print('Features for Performance Dependent Choices: ')
print(df[common.columns_performance].columns)
clf = cl.Classifier_items('Performance depending', df[common.columns_performance])
clf.run_clfs()

print('Features for Team Performance: ')
print(df.columns)
clf = cl.Classifier_items('Team Performance depending', df[common.columns_teams])
clf.run_clfs()

print('Features for Patch Context: ')
print(df.columns)
clf = cl.Classifier_items('Patch Context', df[common.columns_teams + common.item_columns])
clf.run_clfs()
del df


print("Items Baseline Finished!")
