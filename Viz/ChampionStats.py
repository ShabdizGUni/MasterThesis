import pandas as pd
from ggplot import *
from pymongo import *
from data.Piplines import *

db = MongoClient('mongodb://localhost:27017/').LeagueCrawler

championCounts = pd.DataFrame(list(db.matchDetails2000.aggregate(MatchDetails2000.ChampionCounterADC, allowDiskUse=True)))
plot = ggplot(aes(x="count"), data=championCounts) + geom_bar() + facet_grid(x='patch', y='platformId')
print(plot)