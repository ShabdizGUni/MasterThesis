from data.constants import PATCHES
import Lib.Classification.DataHandling as dh
import Lib.Classification.Classifiers_skills as cl
import Lib.Classification.common as common
import sys

limit = int(sys.argv[1])
# Ashe, Ezreal, Caitlyn, Varus, Jhin
champions = [22, 51, 81, 110, 202]
tiers = ["CHALLENGER", "MASTER", "DIAMOND", "PLATINUM"]

df = dh.get_purchase_teams(champions=champions, patches=PATCHES, tiers=["CHALLENGER", "MASTER", "DIAMOND", "PLATINUM"],
                           limit=limit, timeseries=True, min_purch=15)
print("Number of Records: %d" % (len(df)))
print("_________________________________")
print('Features for Blank: ')
print(df.columns)
clf = cl.Classifier_skills('Blank', df[common.columns_blank_item])
clf.run_clfs()
# del df

# df = dh.get_purchase_teams(champions=champions, patches=PATCHES, tiers=tiers, limit=limit)
print('Features for Pre-Game Choices: ')
print(df.columns)
clf = cl.Classifier_skills('Pre-Game Choices', df[common.columns_pre_game])
clf.run_clfs()
# del df

# df = dh.get_purchase_teams(champions=champions, patches=PATCHES, tiers=tiers, limit=limit)
print('Features for In-Game Choices: ')
print(df.columns)
clf = cl.Classifier_skills('In-Game Choices', df[common.columns_in_game])
clf.run_clfs()
# del df

print('Features for "Inventory Choices": ')
print(df.columns)
clf = cl.Classifier_skills('Inventory Choices', df[common.columns_inventory])
clf.run_clfs()
# del df

# df = dh.get_purchase_teams(champions=champions, patches=PATCHES, tiers=tiers, limit=limit)
print('Features for Performance Dependent Choices: ')
print(df.columns)
clf = cl.Classifier_skills('Performance depending', df[common.columns_performance])
clf.run_clfs()
# del df

# df = dh.get_purchase_teams(champions=champions, patches=PATCHES, tiers=tiers, limit=limit)
print('Features for  Team Performance: ')
print(df.columns)
clf = cl.Classifier_skills('Team Performance depending', df[common.columns_teams])
clf.run_clfs()
del df
