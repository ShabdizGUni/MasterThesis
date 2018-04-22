from Lib.Classification.Classifiers import *
from sqlalchemy import *
from data.constants import *


def get_data(col) -> pd.DataFrame:
    data = list(col.find())
    keys = data[1].keys()
    return pd.DataFrame(data, columns=keys)


engine = create_engine('mysql+pymysql://root:Milch4321@localhost:3306/leaguestats', encoding='latin1',
                       echo=False)
item_names = {key: value for (key, value) in list(engine.execute('SELECT * FROM itemkeys'))}

events = get_data(db.jhin_training_set)
frames = get_data(db.jhin_frames_test).drop(['_id'], axis=1)

platformId_fact, platform_keys = pd.factorize(events.platformId)
events['platformId_fact'] = platformId_fact

itemId_fact, item_keys = pd.factorize(events.itemId)
events['itemId_fact'] = itemId_fact

type_fact, type_key = pd.factorize(events.type)
events['type_fact'] = type_fact

events = pd.merge(events, frames, how='inner', left_on=['gameId', 'participantId', 'platformId', 'frameNo'],
                  right_on=['gameId', 'participantId', 'platformId', 'frameNo'], copy=False)

events['is_train'] = np.random.uniform(0, 1, len(events)) <= .9
train, test = events[events['is_train']], events[~events['is_train']]

features = events.columns.difference(['_id', 'gameId', 'itemId', 'is_train', 'platformId', 'type', 'itemId_fact'])

run_classifiers("gold", events, features, item_names, item_keys)

