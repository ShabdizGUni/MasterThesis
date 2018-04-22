from data.constants import PATCHES
import Lib.Classification.DataHandling as dh
import Lib.Classification.Classifiers as cl

# Ashe, Ezreak, Caitlyn, Varus, Jhin
champions = [22, 51, 81, 110, 202]
limit = 1000

df = dh.get_purchases_blank(champions=champions, patches=PATCHES, limit=limit)
print('Features: ')
print(df.columns)
clf = cl.Classifier('Blank_lstm_test', df)
clf.run_clfs()
del df
#
# df = dh.get_purchases_pre_game(champions=champions, patches=PATCHES, limit=limit)
# print('Features: ')
# print(df.columns)
# clf = cl.Classifier('Pre-Game Choices', df)
# clf.run_clfs()
# del df
#
# df = dh.get_purchases_in_game(champions=champions, patches=PATCHES, limit=limit)
# print('Features: ')
# print(df.columns)
# clf = cl.Classifier('In-Game Choices', df)
# clf.run_clfs()
# del df
#
# df = dh.get_purchases_performance(champions=champions, patches=PATCHES, limit=limit)
# print('Features: ')
# print(df.columns)
# clf = cl.Classifier('Performance depending', df)
# clf.run_clfs()
# del df
