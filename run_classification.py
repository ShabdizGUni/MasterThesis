from data.constants import PATCHES
import Lib.Classification.DataHandling as dh
import Lib.Classification.Classifiers as cl
import Lib.Classification.common as common

# Ashe, Ezreal, Caitlyn, Varus, Jhin
champions = [110, 202]
limit = 100000
tiers = ["CHALLENGER", "MASTER", "DIAMOND", "PLATINUM"]

df = dh.get_purchase_teams(champions=champions, patches=PATCHES, tiers=tiers, limit=limit)
print("Number of Records: %d" % (len(df)))
print("_________________________________")
# print('Features for Blank: ')
# print(df.columns)
# clf = cl.Classifier('Blank', df[common.columns_blank])
# clf.run_clfs()
# # del df
#
# # df = dh.get_purchase_teams(champions=champions, patches=PATCHES, tiers=tiers, limit=limit)
# print('Features for Pre-Game Choices: ')
# print(df.columns)
# clf = cl.Classifier('Pre-Game Choices', df[common.columns_pre_game])
# clf.run_clfs()
# # del df
#
# # df = dh.get_purchase_teams(champions=champions, patches=PATCHES, tiers=tiers, limit=limit)
# print('Features for In-Game Choices: ')
# print(df.columns)
# clf = cl.Classifier('In-Game Choices', df[common.columns_in_game])
# clf.run_clfs()
# # del df

# print('Features for "Inventory Choices": ')
# print(df.columns)
# clf = cl.Classifier('Inventory Choices', df[common.columns_inventory])
# clf.run_clfs()
# # del df
#
# # df = dh.get_purchase_teams(champions=champions, patches=PATCHES, tiers=tiers, limit=limit)
# print('Features for Performance Dependent Choices: ')
# print(df.columns)
# clf = cl.Classifier('Performance depending', df[common.columns_performance])
# clf.run_clfs()
# # del df

# df = dh.get_purchase_teams(champions=champions, patches=PATCHES, tiers=tiers, limit=limit)
print('Features for  Team Performance: ')
print(df.columns)
clf = cl.Classifier('Team Performance depending', df[common.columns_teams])
clf.run_clfs()
del df
