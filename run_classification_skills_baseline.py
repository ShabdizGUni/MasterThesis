from data.constants import PATCHES
import Lib.Classification.DataHandling as dh
import Lib.Classification.Classifiers_skills as cl
import Lib.Classification.common as common

print("Skill Level Ups Baseline started!")

# Ashe, Ezreal, Caitlyn, Varus, Jhin
# champions = [22, 51, 83, 110, 202]
champions = [22, 51]
limit = 1000
tiers = ["CHALLENGER", "MASTER", "DIAMOND", "PLATINUM"]

df = dh.get_skills_teams(champions=champions, patches=PATCHES, tiers=tiers, limit=limit, timeseries=True, min_purch=10)
print("Number of Records: %d" % (len(df)))
print("_________________________________")
print('Features for Blank: ')
cols = common.columns_blank_item
if 'type' in cols: cols.remove("type")
if 'itemId' in cols: cols.remove("itemId")
cols.append('skillSlot')
print(df[cols].columns)
# clf = cl.Classifier_skills('Blank', df[cols])
# clf.run_clfs()

print('Features for Pre-Game Choices: ')
cols = common.columns_pre_game
if 'type' in cols: cols.remove("type")
if 'itemId' in cols: cols.remove("itemId")
cols.append('skillSlot')
print(df.columns)
# clf = cl.Classifier_skills('Pre-Game Choices', df[cols])
# clf.run_clfs()

print('Features for In-Game Choices: ')
cols = common.columns_in_game
if 'type' in cols: cols.remove("type")
if 'itemId' in cols: cols.remove("itemId")
cols.append('skillSlot')
print(df.columns)
# clf = cl.Classifier_skills('In-Game Choices', df[cols])
# clf.run_clfs()

print('Features for "Inventory Choices": ')
cols = common.columns_inventory
if 'type' in cols: cols.remove("type")
if 'itemId' in cols: cols.remove("itemId")
cols.append('skillSlot')
print(df.columns)
# clf = cl.Classifier_skills('Inventory Choices', df[cols])
# clf.run_clfs()

print('Features for Performance Dependent Choices: ')
cols = common.columns_performance
if 'type' in cols: cols.remove("type")
if 'itemId' in cols: cols.remove("itemId")
if 'availGold' in cols: cols.remove("availGold")
cols.append('skillSlot')
print(df.columns)
# clf = cl.Classifier_skills('Performance depending', df[cols])
# clf.run_clfs()

print('Features for Team Performance Dependent Choices: ')
cols = common.columns_teams
if 'type' in cols: cols.remove("type")
if 'itemId' in cols: cols.remove("itemId")
if 'availGold' in cols: cols.remove("availGold")
cols.append('skillSlot')
print(df.columns)
clf = cl.Classifier_skills('Team Performance depending', df[cols])
clf.run_clfs()

print('Features for Patch Context: ')
cols = common.columns_teams + common.item_columns
if 'type' in cols: cols.remove("type")
if 'itemId' in cols: cols.remove("itemId")
if 'availGold' in cols: cols.remove("availGold")
cols.append('skillSlot')
clf = cl.Classifier_skills('Patch Context', df[cols])
clf.run_clfs()
del df


print("Skills Baseline Finished!")
