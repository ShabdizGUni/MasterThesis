from pymongo import *
from data.Piplines import *
import re as r
import pandas as pd
import seaborn as sb
import data._Constants as con

# def main():
pd.set_option('display.width', 1000)
db = MongoClient('mongodb://localhost:27017/').LeagueCrawler
itemsets = db.matchDetails.aggregate(Pipeline.itemsets(202, False), allowDiskUse=True)
itemsets: pd.DataFrame = pd.DataFrame(list(itemsets))

itemsets["patch"] = itemsets["gameVersion"].apply(lambda row: r.match("\d.\d+", row).group(0))

itemsets["core_items"] = \
    itemsets.apply(lambda row:
                   tuple(map(
                       lambda x: con.ADC_TIER_3_ITEMS_DICT[x],
                       filter(lambda x: x in con.ADC_TIER_3_ITEMS_LIST, row["items"])
                   )),
                   axis=1)
itemsets["num_core_items"] = itemsets.apply(lambda row: len(row['core_items']), axis=1)
itemsets["two_core_items"] = itemsets["core_items"].apply(lambda x: True if len(x) >= 2 else False)
itemsets["core_items_2"] = itemsets["core_items"].apply(lambda x: x[:2] if len(x) >= 2 else None)
itemsets["three_core_items"] = itemsets["core_items"].apply(lambda x: True if len(x) >= 3 else False)
itemsets["core_items_3"] = itemsets["core_items"].apply(lambda x: x[:3] if len(x) >= 3 else None)
itemsets["four_core_items"] = itemsets["core_items"].apply(lambda x: True if len(x) >= 4 else False)
itemsets["core_items_4"] = itemsets["core_items"].apply(lambda x: x[:4] if len(x) >= 4 else None)

# itemsetsJhin["core_items"] = \
#     itemsetsJhin.apply(lambda row: tuple(map(lambda x: con.ADC_TIER_3_ITEMS_DICT[x], row['core_items'])), axis=1)

df = itemsets

gamesCount_2_items = df[df['two_core_items']].groupby(['championId', 'platformId', 'patch']).size().reset_index(
    name='gamesPlayed_2')
gamesCount_3_items = df[df['three_core_items']].groupby(['championId', 'platformId', 'patch']).size().reset_index(
    name='gamesPlayed_3')
gamesCount_4_items = df[df['four_core_items']].groupby(['championId', 'platformId', 'patch']).size().reset_index(
    name='gamesPlayed_4')

# First two items
result_2_items = df[df["num_core_items"] >= 2] \
    .groupby(["championId", "platformId", "patch", "core_items_2"]).core_items_2 \
    .count() \
    .reset_index(name="2_items_count")

result_2 = pd.merge(result_2_items, gamesCount_2_items, on=['championId', 'platformId', 'patch'])
result_2['freq'] = result_2['2_items_count'] / result_2['gamesPlayed_2']

sb.set(font_scale=1)
g2 = sb.FacetGrid(result_2[(result_2['freq'] > 0.02)][['platformId', 'patch', 'core_items_2', 'freq']],
                  col='patch', col_wrap=5, col_order=con.PATCHES)
g2 = (g2.map(sb.barplot, 'freq', 'core_items_2', 'platformId', hue_order=["EUW1", "NA1", "BR1", "KR"], palette=sb.color_palette("pastel")).add_legend())

# alternatively:
g2 = sb.factorplot('freq', 'core_items_2', 'platformId',
                   data=result_2[(result_2['freq'] > 0.02)][['platformId', 'patch', 'core_items_2', 'freq']],
                   kind='bar',
                   col='patch',
                   col_wrap=5,
                   col_order=con.PATCHES,
                   palette=sb.color_palette("pastel"))

# First three items
result_3_items = df[df["num_core_items"] >= 2] \
    .groupby(["championId", "platformId", "patch", "core_items_3"]).core_items_3 \
    .count() \
    .reset_index(name="2_items_count")

result_3 = pd.merge(result_3_items, gamesCount_3_items, on=['championId', 'platformId', 'patch'])
result_3['freq'] = result_3['2_items_count'] / result_3['gamesPlayed_3']

sb.set(font_scale=1)
g3 = sb.FacetGrid(result_3[(result_3['freq'] > 0.02)][['patch', 'core_items_3', 'freq']],
                  col='patch', col_wrap=6, col_order=con.PATCHES)
g3 = g3.map(sb.barplot, 'freq', 'core_items_3')

# First four items
result_4_items = df[df["num_core_items"] >= 2] \
    .groupby(["championId", "platformId", "patch", "core_items_4"]).core_items_4 \
    .count() \
    .reset_index(name="2_items_count")

result_4 = pd.merge(result_4_items, gamesCount_4_items, on=['championId', 'platformId', 'patch'])
result_4['freq'] = result_4['2_items_count'] / result_4['gamesPlayed_4']

sb.set(font_scale=1)
g4 = sb.FacetGrid(result_4[(result_4['freq'] > 0.02)][['patch', 'core_items_4', 'freq']],
                 col='patch', col_wrap=6, col_order=con.PATCHES)
g4 = g4.map(sb.barplot, 'freq', 'core_items_4')
