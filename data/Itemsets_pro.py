from pymongo import *
import data.Pipelines as pl
from matplotlib import pyplot as plt
import re as r
import operator
import numpy as np
import pandas as pd
import seaborn as sns
import data.constants as con
import os

# sns.set(rc={'figure.figsize': (21.7, 11.00)})


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


pd.set_option('display.width', 1000)
db = MongoClient('mongodb://localhost:27017/').LeagueCrawler
threshold = 0.1

for _id in con.ADC_CHAMPIONS_DICT.keys():
    itemsets = db.itemsets_adc_pro.find({"championId": _id})
    champion = con.ADC_CHAMPIONS_DICT[_id]
    path = "Viz_Pro/" + champion + "/"
    PATCHES = ["7.1-7.3", "7.4-7.8", "7.9-7.13", "7.14-7.18"]

    ensure_dir(path)

    itemsets: pd.DataFrame = pd.DataFrame(list(itemsets))

    # itemsets["patch"] = itemsets["gameVersion"].apply(lambda row: r.match("\d.\d+", row).group(0))
    itemsets["patch_grp"] = np.where(itemsets['patch'].isin(["7.1", "7.2", "7.3"]), PATCHES[0],
                                     np.where(itemsets['patch'].isin(["7.5", "7.6", "7.7", "7.8"]), PATCHES[1],
                                              np.where(itemsets['patch'].isin(["7.9", "7,10", "7.11", "7.12", "7.13"]),
                                                       PATCHES[2],
                                                       np.where(itemsets['patch'].isin(["7.14", "7.15", "7.16", "7.17",
                                                                                        "7.18"]), PATCHES[3], "none")
                                                       )
                                              )
                                     )
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

    df = itemsets

    gamesCount_2_items = df[df['two_core_items']].groupby(['championId', 'patch_grp']).size().reset_index(
        name='gamesPlayed_2')
    gamesCount_3_items = df[df['three_core_items']].groupby(['championId', 'patch_grp']).size().reset_index(
        name='gamesPlayed_3')
    gamesCount_4_items = df[df['four_core_items']].groupby(['championId', 'patch_grp']).size().reset_index(
        name='gamesPlayed_4')

    # First two items
    result_2_items = df[df["num_core_items"] >= 2] \
        .groupby(["championId", "patch_grp", "core_items_2"]).core_items_2 \
        .count() \
        .reset_index(name="2_items_count")

    result_2 = pd.merge(result_2_items, gamesCount_2_items, on=['championId', 'patch_grp'])
    result_2['freq'] = result_2['2_items_count'] / result_2['gamesPlayed_2']
    num_games_2 = sum(itemsets["two_core_items"])

    # sns.set(font_scale=0.6)
    df2 = result_2[(result_2['freq'] > threshold)][['patch_grp', 'core_items_2', 'freq']]
    g2, ax = plt.subplots(figsize=(22, 10))
    g2 = sns.FacetGrid(df2,
                       col='patch_grp', col_wrap=5, col_order=PATCHES)
    g2 = (g2.map(sns.barplot, 'freq', 'core_items_2',  # 'platformId', hue_order=["EUW1", "NA1", "BR1", "KR"],
                 # palette=sns.color_palette("pastel"),
                 order=sorted(df2.core_items_2.unique(), key=operator.itemgetter(0))).add_legend())
    # g2.fig.subplots_adjust(top=0.9)
    g2.fig.suptitle(champion + "'s Itemsets - Games: " + str(num_games_2)+", Threshold:" + str(threshold), fontsize=20)
    g2.fig.tight_layout()
    g2.savefig(path + "2_items.svg")

    # alternatively:
    # g2 = sb.factorplot('freq', 'core_items_2', 'platformId',
    #                    data=result_2[(result_2['freq'] > 0.02)][['platformId', 'patch_grp', 'core_items_2', 'freq']],
    #                    kind='bar',
    #                    col='patch_grp',
    #                    col_wrap=5,
    #                    col_order=PATCHES,
    #                    palette=sb.color_palette("pastel"))

    # First three items
    result_3_items = df[df["num_core_items"] >= 3] \
        .groupby(["championId", "patch_grp", "core_items_3"]).core_items_3 \
        .count() \
        .reset_index(name="2_items_count")

    result_3 = pd.merge(result_3_items, gamesCount_3_items, on=['championId', 'patch_grp'])
    result_3['freq'] = result_3['2_items_count'] / result_3['gamesPlayed_3']
    num_games_3 = sum(itemsets["three_core_items"])

    # sns.set(font_scale=1.3)
    df3 = result_3[(result_3['freq'] > threshold)][['patch_grp', 'core_items_3', 'freq']]
    g3, ax = plt.subplots(figsize=(22, 10))
    g3 = sns.FacetGrid(df3,
                       col='patch_grp', col_wrap=5, col_order=PATCHES)
    g3 = (g3.map(sns.barplot, 'freq', 'core_items_3',  # 'platformId', hue_order=["EUW1", "NA1", "BR1", "KR"],
                 # palette=sns.color_palette("pastel"),
                 order=sorted(df3.core_items_3.unique(), key=operator.itemgetter(0))
                 ).add_legend())
    # g3.fig.subplots_adjust(top=0.9)
    g3.fig.suptitle(champion + " Itemsets - Games: " + str(num_games_3)+", Threshold:" + str(threshold), fontsize=20)
    g3.fig.tight_layout()
    g3.savefig(path + "3_items.svg")

    # First four items
    result_4_items = df[df["num_core_items"] >= 4] \
        .groupby(["championId", "patch_grp", "core_items_4"]).core_items_4 \
        .count() \
        .reset_index(name="4_items_count")

    result_4 = pd.merge(result_4_items, gamesCount_4_items, on=['championId', 'patch_grp'])
    result_4['freq'] = result_4['4_items_count'] / result_4['gamesPlayed_4']
    num_games_4 = sum(itemsets["four_core_items"])

    sns.set(font_scale=0.6)
    df4 = result_4[(result_4['freq'] > threshold)][['patch_grp', 'core_items_4', 'freq']]
    g4, ax = plt.subplots(figsize=(21.7, 11.00))
    g4 = sns.FacetGrid(df4,
                       col='patch_grp', col_wrap=5, col_order=PATCHES)
    g4 = (g4.map(sns.barplot, 'freq', 'core_items_4',  # 'platformId', hue_order=["EUW1", "NA1", "BR1", "KR"],
                 # palette=sns.color_palette("pastel"),
                 order=sorted(df4.core_items_4.unique(), key=operator.itemgetter(0))).add_legend())
    g4.fig.subplots_adjust(top=0.9)
    # g4.fig.suptitle(champion + "'s Itemsets - Games: " + str(num_games_4)+", Threshold:" + str(threshold), fontsize=20)
    # g4.tight_layout()
    g4.savefig(path + "4_items.svg")
