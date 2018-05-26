import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from pymongo import MongoClient
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from ggplot import *

db = MongoClient('mongodb://localhost:27017/').LeagueCrawler
maria = create_engine('mysql+pymysql://root:Milch4321@localhost:3306/leaguestats', encoding='latin1',
                      echo=False)
# 2010: biscuit
# 2003: health potion
# 1055: dorans blade
# 1036: long sword
# 1042: dagger
# 3031: infinity edge
# 1043: recurve bow
# 1053: Vamp Scepter
# 3144: Cutlass
# 3153: BORK

# 3134: Serrated Dirk
#

data = list(db.adc_purchase_details.find(
    {
        "patch": "7.3",
        "itemId": {"$in": [2003, 1055,
                           1036, 3134, 3133, 3142,

                           ]}
    },
    {"_id": 0, "itemId": 1, "timestamp": 1, "type": 1})
)
frame = pd.DataFrame(data, columns=data[0].keys())
del data
frame['seconds'] = frame['timestamp']/1000
frame['minutes'] = round(frame['timestamp']/(1000*60), 1)


original = ["#9370DB", '#E32636', '#696969', '#ED9121', '#CFB53B']
blues = ["#6593F5", "#0F52BA", "#1034A6", "#000080"]
reds = ["#ff3333", "#e60000", "#990000", "#660000"]

a4_dims = (9, 5)
fig, ax = plt.subplots(figsize=a4_dims)
dorans_patch = mpatches.Patch(color="#9370DB", label='doran\'s blade: 450g')
healthPot_patch = mpatches.Patch(color='#E32636', label='health potion: 50g')
longSword_patch = mpatches.Patch(color='#696969', label='long sword: 350')
dirk_patch = mpatches.Patch(color='#ED9121', label='serrated dirk: 1100g')
warhammer_patch = mpatches.Patch(color="#0F52BA", label='caulfield\'s warhammer: 1100g')
yom_patch = mpatches.Patch(color='#CFB53B', label='yomouu\'s ghostblade: 2900g')
dorans_patch_sold = mpatches.Patch(edgecolor="#9370DB", linestyle="--", Fill=False, label='doran\'s blade sold')
g = sns.distplot(frame[(frame['itemId'] == 1055) & (frame['type'] == "ITEM_PURCHASED")]['minutes'], color=dorans_patch.get_facecolor(), hist=True)
sns.distplot(frame[(frame['itemId'] == 2003) & (frame['type'] == "ITEM_PURCHASED")]['minutes'], color=healthPot_patch.get_facecolor(), hist=True)
sns.distplot(frame[frame['itemId'] == 1036]['minutes'], color=longSword_patch.get_facecolor(), hist=True)
sns.distplot(frame[frame['itemId'] == 3134]['minutes'], color=dirk_patch.get_facecolor(), hist=True)
sns.distplot(frame[frame['itemId'] == 3133]['minutes'], color=warhammer_patch.get_facecolor(), hist=True)
sns.distplot(frame[frame['itemId'] == 3142]['minutes'], color=yom_patch.get_facecolor(), hist=True)
sns.distplot(frame[(frame['itemId'] == 1055) & (frame['type'] == "ITEM_SOLD")]['minutes'], color=dorans_patch_sold.get_edgecolor(), hist=True, kde_kws={"ls": "--"})
longSword_patch.get_facecolor()
plt.legend(handles=[dorans_patch, healthPot_patch, longSword_patch, dirk_patch, warhammer_patch, yom_patch, dorans_patch_sold])
g.set(xlim=(0, 45), ylim=(0, 0.5))
fig.tight_layout()
fig.savefig("Purhcases_GB.png")


