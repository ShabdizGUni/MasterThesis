import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from pymongo import MongoClient
import seaborn as sns
import matplotlib.pyplot as plt
from ggplot import *

db = MongoClient('mongodb://localhost:27017/').LeagueCrawler
maria = create_engine('mysql+pymysql://root:Milch4321@localhost:3306/leaguestats', encoding='latin1',
                         echo=False)

data = db.adc_purchase_details.aggregate([
    {"$group": {
        "_id": {
            "itemId": "$itemId"
        },
        "count": {"$sum": 1}
    }},
    {"$project": {
        "_id": 0,
        "itemId": "$_id.itemId",
        "count": "$count"
    }},
], allowDiskUse=True)
data_list = list(data)
item_names = pd.read_sql(
    "select it.key as id, name, depth, "
    "   it.FlatPhysicalDamageMod as ad, "
    "   it.PercentAttackSpeedMod as \"ats\", "
    "   it.FlatCritChanceMod as crit ,"
    "   it.goldTotal as gold "
    "from itemstats_adj it "
    "where patch =\"7.18\"", maria)
frame = pd.DataFrame(data_list, columns=data_list[0].keys())
frame = pd.merge(frame, item_names, left_on='itemId', right_on='id', copy=False)
frame['stat_cat'] = np.where(frame['ad'] != 0.0, 'ad', np.where(frame['ats'] != 0.0, "attack speed", np.where(frame['crit'] != 0.0, "crit", 'none')))

df = frame[frame["count"] > 10000].sort_values(by=['count'], ascending=False).reset_index()
df['count'] = df['count']/1000
# df['stat_cat'] = '#00BFFF'
df['stat_cat'] = np.where(df['ad'] != 0.0, 'ad', np.where(df['ats'] != 0.0, "attack speed", np.where(df['crit'] != 0.0, "crit", 'none')))

a4_dims = (9, 5)
fig, ax = plt.subplots(figsize=a4_dims)
colors = ["#E32636", "#FF8C00", "#FFD300", '#00BFFF']
color_dict = {'ad': "#E32636", "attack speed": "#FF8C00", "crit": "#FFD300", "none": '#00BFFF'}
g = sns.barplot(x='name', y='count', data=df, palette=df['stat_cat'].map(color_dict))
g.set_xticklabels(labels=df['name'], rotation=90, size=8)
g.set(xlabel='Item Name and Price', ylabel='#Purchases in thosands')
ax.set(ylim=(0,4000))
for p, gold in zip(ax.patches, df['gold']):
        ax.annotate("%.d g" % gold, (p.get_x() + p.get_width() / 1.5, p.get_height()),
             ha='center', va='center', rotation=90, fontsize=7,
             xytext=(0, 15), textcoords='offset points')
fig.tight_layout()
fig.savefig("Overall")

# below 1000g
df0000 = frame[(frame['gold'] < 1000) & (frame["count"] > 10000)].sort_values(by='count', ascending=False)
df0000['count'] = df0000['count']/1000
a4_dims = (9, 5)
fig, ax = plt.subplots(figsize=a4_dims)
color_dict = {'ad': "#E32636", "attack speed": "#FF8C00", "crit": "#FFD300", "none": '#00BFFF'}
g = sns.barplot(x='name', y='count', data=df0000, palette=df0000['stat_cat'].map(color_dict))
g.set_xticklabels(labels=df0000['name'], rotation=90, size=8)
g.set(xlabel='Item Name', ylabel='#Purchases in thosands', ylim=(0, 4000))
for p, gold in zip(ax.patches, df0000['gold']):
        ax.annotate("%.d g" % gold, (p.get_x() + p.get_width() / 1.5, p.get_height()),
             ha='center', va='center', rotation=90, fontsize=7,
             xytext=(0, 15), textcoords='offset points')
fig.tight_layout()
fig.savefig("Below1000g")

# > 1000 gold
df1000 = frame[(frame['gold'] > 1000) & (frame["count"] > 10000)].sort_values(by='count', ascending=False)
df1000['count'] = df1000['count']/1000
a4_dims = (9, 5)
fig, ax = plt.subplots(figsize=a4_dims)
color_dict = {'ad': "#E32636", "attack speed": "#FF8C00", "crit": "#FFD300", "none": '#00BFFF'}
g = sns.barplot(x='name', y='count', data=df1000, palette=df1000['stat_cat'].map(color_dict))
g.set_xticklabels(labels=df1000['name'], rotation=90, size=8)
g.set(xlabel='Item Name', ylabel='#Purchases in thosands', ylim=(0, 1600))
for p, gold in zip(ax.patches, df1000['gold']):
        ax.annotate("%.d g" % gold, (p.get_x() + p.get_width() / 1.5, p.get_height()),
             ha='center', va='center', rotation=90, fontsize=7,
             xytext=(0, 15), textcoords='offset points')
fig.tight_layout()
fig.savefig("Above1000g")

#  > 2000g
df2000 = frame[(frame['gold'] > 2000) & (frame["count"] > 10000)].sort_values(by='count', ascending=False)
df2000['count'] = df2000['count']/1000
a4_dims = (9, 5)
fig, ax = plt.subplots(figsize=a4_dims)
color_dict = {'ad': "#E32636", "attack speed": "#FF8C00", "crit": "#FFD300", "none": '#00BFFF'}
g = sns.barplot(x='name', y='count', data=df2000, palette=df2000['stat_cat'].map(color_dict))
g.set_xticklabels(labels=df2000['name'], rotation=90, size=8)
g.set(xlabel='Item Name', ylabel='#Purchases in thosands')
ax.set(ylim=(0, 1000))
for p, gold in zip(ax.patches, df2000['gold']):
        ax.annotate("%.d g" % gold, (p.get_x() + p.get_width() / 1.5, p.get_height()),
             ha='center', va='center', rotation=90, fontsize=7,
             xytext=(0, 15), textcoords='offset points')
fig.tight_layout()
fig.savefig("Above2000g")
