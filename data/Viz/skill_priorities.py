import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from pymongo import MongoClient
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import data.constants as con
from collections import Counter


for _id in con.ADC_CHAMPIONS_DICT.keys():
    db = MongoClient('mongodb://localhost:27017/').LeagueCrawler
    champion = con.ADC_CHAMPIONS_DICT[_id]
    path = champion + "/"
    pipeline = [
        {"$match": {
            "championId": _id
            }
        },
        {"$group": {
            "_id": {"gameId": "$gameId", "side": "$side", "patch": "$patch", "tier": "$tier"
            },
            "q": {
                "$sum": {"$cond": [
                    {"$eq": ["$skillSlot", 1]},
                    1,
                    0
                ]}
            },
            "w": {
                "$sum" : {"$cond": [
                    {"$eq": ["$skillSlot", 2]},
                    1,
                    0
                ]}
            },
            "e": {
                "$sum": {"$cond": [
                    {"$eq": ["$skillSlot", 3]},
                    1,
                    0
                ]}
            },
            "r": {
                "$sum": {"$cond": [
                    {"$eq": ["$skillSlot", 4]},
                    1,
                    0
                ]}
            }
        }},
        {"$project": {
            "gameId": "$_id.gameId", "side": "$_id.side", "patch": "$_id.patch", "tier": "$_id.tier",
            "q": "$q", "w": "$w", "e": "$e", "r": "$r"
            }
        },
        {"$group": {
            "_id": {"patch": "$patch"},
            "q" : {"$avg": "$q"},
            "w" : {"$avg": "$w"},
            "e" : {"$avg": "$e"},
            "r" : {"$avg": "$r"}
            }
        },
        {"$project": {
            "_id":0,
            "patch": "$_id.patch",
            "tier": "$_id.tier",
            "q": "$q", "w": "$w", "e":"$e", "r" : "$r"
            }
        }
    ]

    data = list(db.adc_skill_level_ups.aggregate(pipeline, allowDiskUse=True))
    frame = pd.DataFrame(data=data, columns=data[0].keys())
    frame_melt = pd.melt(frame, id_vars=['patch'], value_vars=['q', 'w', 'e', 'r'])
    g2, ax = plt.subplots(figsize=(22, 10))
    g2 = sns.FacetGrid(frame_melt, col='patch', col_wrap=5, col_order=con.PATCHES).set(ylabel="AVG Points Spent", xlabel="Spell")
    g2 = (g2.map(sns.barplot, 'variable', 'value'))
    plt.tight_layout()
    plt.savefig(path + "AVG_Points_Spent.svg")

    def count_spells(row):
        cnt = Counter()
        translate = {
            1: "Q",
            2: "W",
            3: "E",
            4: "R"
        }
        spells_trans = [translate[i] for i in row['spells']]
        for i in spells_trans: cnt[i] += 1
        output = {
            "Q": cnt['Q'],
            "W": cnt['W'],
            "E": cnt['E'],
            "R": cnt['R']
        }
        return output


    pipeline2 = [
        {"$match": {
            "championId": _id
            }
        },
        {"$group": {
            "_id": {"gameId": "$gameId", "side": "$side", "patch": "$patch", "tier": "$tier"
            },
            "spells": {"$push": "$skillSlot"}
        }},
        {"$project": {
            "gameId": "$_id.gameId", "side": "$_id.side", "patch": "$_id.patch", "tier": "$_id.tier",
            "spells": { "$slice": ["$spells", 11] } ,
            "more_than_11": { "$eq": [ {"$size": "$spells"}, 11 ] }
            }
        },
        {"$match": {
            "more_than_11": True
            }
        },
        {"$group": {
            "_id": { "patch": "$patch", "spells": "$spells" },
            "count": {"$sum": 1 }
            }
        },
        {"$project": {
            "_id": 0,
            "patch": "$_id.patch",
            "spells": "$_id.spells",
            "count": "$count"
            }
        }
    ]
    data2 = list(db.adc_skill_level_ups.aggregate(pipeline2, allowDiskUse=True))
    frame2 = pd.DataFrame(data=data2, columns=data2[0].keys())
    cnt = Counter()
    frame2['spell_prio'] = frame2.apply(lambda x: count_spells(x), axis=1)
    frame2['spells'] = frame2['spells'].astype(str)
    frame2['spell_prio'] = frame2['spell_prio'].astype(str)
    test = frame2.groupby(['patch', 'spell_prio'])['count'].sum()
    test = pd.DataFrame({'count' : frame2.groupby(['patch', 'spell_prio'])['count'].sum()}).reset_index()

    # frame_melt2 = pd.melt(frame2, id_vars=['patch', 'spells'], value_vars=['count'])
    try:
        g2, axs = plt.subplots(figsize=(8, 2), ncols=3)
        plt.rc('ytick', labelsize=6)
        g2 = sns.barplot(data=test[(test['patch'] == '7.3')].sort_values(by='count', ascending=False).head(5), x='count', y='spell_prio', ax=axs[0], color="#6593F5")
        axs[0].set(ylabel="Spell Level Ups", title="Patch 7.3")
        g2 = sns.barplot(data=test[(test['patch'] == '7.11')].sort_values(by='count', ascending=False).head(5), x='count', y='spell_prio', ax=axs[1], color="#6593F5")
        axs[1].set(ylabel="", title="Patch 7.11")
        g2 = sns.barplot(data=test[(test['patch'] == '7.18')].sort_values(by='count', ascending=False).head(5), x='count', y='spell_prio', ax=axs[2], color="#6593F5")
        axs[2].set(ylabel="", title="Patch 7.18")
        plt.tight_layout()
        plt.savefig(path + "Points_Spent_Seq.svg")
    except Exception as e:
        print("Not possible for Champion:" + str(champion))
    # g2 = (g2.map(sns.barplot, 'variable', 'value'))