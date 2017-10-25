from pymongo import *
import data.Pipelines as pl
import re as r
import operator
import pandas as pd
import seaborn as sns
import data._Constants as con
import os


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

pd.set_option('display.width', 1000)
db = MongoClient('mongodb://localhost:27017/').LeagueCrawler

for _id in con.ADC_CHAMPIONS_DICT.keys():
    itemsets = db.itemsets_adc.find({"championId": _id})  # Jhin
    champion = con.ADC_CHAMPIONS_DICT[_id]
    path = "Viz/" + champion + "/"

    ensure_dir(path)

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

    sns.set(font_scale=1)
    df2 = result_2[(result_2['freq'] > 0.05)][['platformId', 'patch', 'core_items_2', 'freq']]
    g2 = sns.FacetGrid(df2,
                       col='patch', col_wrap=5, col_order=con.PATCHES)
    g2 = (g2.map(sns.barplot, 'freq', 'core_items_2', 'platformId', hue_order=["EUW1", "NA1", "BR1", "KR"],
                 palette=sns.color_palette("pastel"),
                 order=sorted(df2.core_items_2.unique(), key=operator.itemgetter(0))).add_legend())
    g2.fig.subplots_adjust(top=0.9)
    g2.fig.suptitle(champion + "'s Itemsets")
    g2.savefig(path+"2_items.pdf")

    # alternatively:
    # g2 = sb.factorplot('freq', 'core_items_2', 'platformId',
    #                    data=result_2[(result_2['freq'] > 0.02)][['platformId', 'patch', 'core_items_2', 'freq']],
    #                    kind='bar',
    #                    col='patch',
    #                    col_wrap=5,
    #                    col_order=con.PATCHES,
    #                    palette=sb.color_palette("pastel"))

    # First three items
    result_3_items = df[df["num_core_items"] >= 3] \
        .groupby(["championId", "platformId", "patch", "core_items_3"]).core_items_3 \
        .count() \
        .reset_index(name="2_items_count")

    result_3 = pd.merge(result_3_items, gamesCount_3_items, on=['championId', 'platformId', 'patch'])
    result_3['freq'] = result_3['2_items_count'] / result_3['gamesPlayed_3']


    sns.set(font_scale=0.7)
    df3 = result_3[(result_3['freq'] > 0.05)][['platformId', 'patch', 'core_items_3', 'freq']]
    g3 = sns.FacetGrid(df3,
                       col='patch', col_wrap=5, col_order=con.PATCHES)
    g3 = (g3.map(sns.barplot, 'freq', 'core_items_3', 'platformId', hue_order=["EUW1", "NA1", "BR1", "KR"],
                 palette=sns.color_palette("pastel"),
                 order=sorted(df3.core_items_3.unique(), key=operator.itemgetter(0))
                 ).add_legend())
    g3.fig.subplots_adjust(top=0.9)
    g3.fig.suptitle(champion + " Itemsets")
    g3.savefig(path+"3_items.pdf")

    # First four items
    result_4_items = df[df["num_core_items"] >= 4] \
        .groupby(["championId", "platformId", "patch", "core_items_4"]).core_items_4 \
        .count() \
        .reset_index(name="4_items_count")

    result_4 = pd.merge(result_4_items, gamesCount_4_items, on=['championId', 'platformId', 'patch'])
    result_4['freq'] = result_4['4_items_count'] / result_4['gamesPlayed_4']

    sns.set(font_scale=0.5)
    df4 = result_4[(result_4['freq'] > 0.05)][['platformId', 'patch', 'core_items_4', 'freq']]
    g4 = sns.FacetGrid(df4,
                       col='patch', col_wrap=5, col_order=con.PATCHES)
    g4 = (g4.map(sns.barplot, 'freq', 'core_items_4', 'platformId', hue_order=["EUW1", "NA1", "BR1", "KR"],
                 palette=sns.color_palette("pastel"),
                 order=sorted(df4.core_items_4.unique(), key=operator.itemgetter(0))).add_legend())
    g4.fig.subplots_adjust(top=0.9)
    g4.fig.suptitle(champion + "'s Itemsets")
    g4.savefig(path+"4_items.pdf")
