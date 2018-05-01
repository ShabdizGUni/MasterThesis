import pandas as pd
import numpy as np
import bson
import Lib.Classification.common as common
from pymongo import MongoClient
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from collections import deque
from data import constants as const


def get_mongo_connection():
    return MongoClient('mongodb://localhost:27017/').LeagueCrawler


def get_maria_connection():
    return create_engine('mysql+pymysql://root:Milch4321@localhost:3306/leaguestats', encoding='latin1',
                         echo=False)


# not possible to enter seed manually. gaameIds would need to be stored seperately to replicate
def get_data_frame(col, limit=None, champions=None, tiers=None, patches=None) -> pd.DataFrame:
    criteria = {}
    if tiers is not None:
        if type(tiers) is str: criteria["tier"] = tiers
        if type(tiers) is list: criteria["tier"] = {"$in": tiers}
    data = []
    if champions:
        for c in champions:
            criteria['championId'] = int(c)
            if limit:
                per_patch = limit/len(champions)
                if patches:
                    per_patch = per_patch/len(patches)
                    for p in patches:
                        criteria['patch'] = p
                        data = data + list(col.aggregate([{"$match": criteria}, {"$sample": {"size": int(per_patch)}}], allowDiskUse=True))
                else:
                    data = data + list(col.aggregate([{"$match": criteria}, {"$sample": {"size": int(per_patch)}}], allowDiskUse=True))
            else:
                data = data + list(col.find(criteria))
    else:
        data = data + (list(col.find()))
    keys = data[1].keys()
    df = pd.DataFrame(data, columns=keys)
    return pd.DataFrame(df.drop(columns=['_id']))


def get_events_as_timeseries(champions, patches, tiers, limit, min_purch):
    print("Get GameIds")
    Ids = []
    data = []
    criteria = {
        "countRel": {
            "$gte": min_purch
        }
    }
    if tiers is not None:
        if type(tiers) is str: criteria["tier"] = tiers
        if type(tiers) is list: criteria["tier"] = {"$in": tiers}
    per_crit = int(limit/(len(patches)*len(champions)))
    mongo = get_mongo_connection()
    for c in champions:
        criteria['championId'] = c
        for p in patches:
            criteria['patch'] = p
            Ids = Ids + list(mongo.adc_purchase_details_Ids.aggregate([
                {"$match": criteria},
                {"$sample": {"size": per_crit}}
            ], allowDiskUse=True))
    print("Get Purchases")
    # columns = list(mongo.adc_purchase_details.find_one())
    # frame = pd.DataFrame([], columns=columns)
    for idx, Id in enumerate(Ids):
        # new = list(mongo.adc_purchase_details.find({
        #     "gameId": Id['gameId'],
        #     "side": Id['side']}
        # ))
        data.extend(list(mongo.adc_purchase_details.find({
            "gameId": Id['gameId'],
            "side": Id['side']}
        )))
    #     print(str(idx))
    #     if (idx + 1) % 1000 == 0:
    #         print("Unload No.: ", str(int(idx / 1000)))
    #         new_frame = pd.DataFrame(data, columns=columns)
    #         for i in range(len(new_frame.index)):
    #             frame.loc[len(frame)] = new_frame.loc[i, columns]
    #         data = []
    # return frame
    keys = data[1].keys()
    df = pd.DataFrame(data, columns=keys)
    return df.drop(columns=["_id"])


def get_according_frames(mongo, events):
    gc_pairs = events.groupby(by=['gameId', 'championId'], as_index=False).first().loc[:, ['gameId', 'championId']]
    data = []
    for i, gameId, champId in gc_pairs.itertuples():
        data = data + list(mongo.adc_frames.find({"gameId": gameId, "championId": champId}))
    df = pd.DataFrame(data)
    df = df.drop(columns=['_id'])
    return df


def get_item_dict():
    engine = get_maria_connection()
    return {key: value for (key, value) in list(engine.execute('SELECT * FROM itemkeys'))}


def get_item_infos():
    engine = get_maria_connection()
    item_infos = pd.read_sql("SELECT * FROM ITEMSTATS", engine)
    item_infos["patch"] = item_infos["version"].str.extract("(\d.\d+)")
    item_infos = item_infos.rename(columns={"key": "itemId"})
    return item_infos


def get_item_stats():
    df = pd.DataFrame(const.PATCHES, columns=['patch'])
    maria = get_maria_connection()
    for name, id_, stats in common.itemStatSet:
        col_ids = [id_ + "_" + stat for stat in stats]
        select_zip = zip(stats, col_ids)
        selects = [col + " as " + col_id for col, col_id in select_zip]
        statement = "SELECT " + str.join(',', ['patch'] + selects) + " FROM itemstats_adj WHERE name = '" + name + "'"
        df2 = pd.read_sql(statement, maria)
        df = pd.merge(df, df2, left_on=['patch'], right_on=['patch'], copy=False)
    return df


def get_events(mongo):
    return get_data_frame(mongo.adc_purchase_details)


def get_frames(mongo):
    return get_data_frame(mongo.adc_frames)


def join_frames(events):
    champions = list(events.championId.unique())
    patches = list(events.patch.unique())
    db = get_mongo_connection()
    #frames = get_data_frame(db.adc_frames, champions=champions, patches=patches)
    frames = get_according_frames(get_mongo_connection(), events)
    events = pd.merge(events, frames, how='inner', left_on=['gameId', 'participantId', 'patch', 'championId', 'platformId', 'frameNo'],
                      right_on=['gameId', 'participantId', 'patch', 'championId', 'platformId', 'frameNo'], copy=False)
    return events


def compute_reward_gold(events):
    killreward = 300
    killassist = 150
    turretreward = 300
    turretassist = 125
    events["Cost"] = np.where(events["type"] == "ITEM_PURCHASED", events["goldBase"], events["goldBase"] * -1)
    # noinspection PyTypeChecker
    events["Cost"] = np.where(events["type"] == "ITEM_SOLD", events["goldSell"] * -1, events["Cost"])
    events['frameCost'] = events.groupby(['gameId', 'platformId', 'frameNo'])['Cost'].cumsum()
    events['availGold'] = events['currentGold'] - events.groupby(['gameId', 'platformId', 'frameNo'])['frameCost'] \
        .shift(1).fillna(0)
    events['killBounty'] = \
        events.sort_values(by=['platformId', 'gameId', 'frameNo', 'timestamp']).groupby(['platformId', 'gameId'])['kills'] \
        .diff(1).fillna(0) * killreward
    events['assistsBounty'] = \
        events.sort_values(by=['platformId', 'gameId', 'frameNo', 'timestamp']).groupby(['platformId', 'gameId'])['assists'] \
        .diff(1).fillna(0) * killassist
    events['turretKillBounty'] = \
        events.sort_values(by=['platformId', 'gameId', 'frameNo', 'timestamp']).groupby(['platformId', 'gameId'])['turretKills'] \
        .diff(1).fillna(0) * turretreward
    events['turretAssistBounty'] = \
        events.sort_values(by=['platformId', 'gameId', 'frameNo', 'timestamp']).groupby(['platformId', 'gameId'])['turretAssists'] \
        .diff(1).fillna(0) * turretassist
    events['availGold'] = \
        events['availGold'] + events['killBounty'] + events['assistsBounty'] + \
        events['turretKillBounty'] + events['turretAssistBounty']
    events = events.drop(columns=["frameCost", "Cost", "goldBase", "goldSell", "goldTotal", "currentGold"])
    return events


# for neural networks and deep learning
def one_hot_encode_columns(df, columns, drop=True):
    c = []
    for column in columns:
        if column in df.columns: c.append(column)
    for column in c:
        values = list(set(df[column]))
        for v in values:
            df[v] = np.where(df[column].str.contains(v), 1, 0)
        if drop: df = df.drop(columns=[column])
    return df


def encode_text_index(df, name):
    le = preprocessing.LabelEncoder()
    df[name] = le.fit_transform(df[name])
    return le.classes_


# for trees
def factorise_column(df, column):
    column_fact, column_key = pd.factorize(df[column])
    df[column + "_fact"] = column_fact
    return column_fact, column_key


def to_xy(df, target):
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)
    # find out the type of the target column.  Is it really this hard? :(
    target_type = df[target].dtypes
    target_type = target_type[0] if hasattr(target_type, '__iter__') else target_type
    # Encode to int for classification, float otherwise. TensorFlow likes 32 bits.
    if target_type in (np.int64, np.int32):
        # Classification
        dummies = pd.get_dummies(df[target])
        return df.as_matrix(result).astype(np.float32), dummies.as_matrix().astype(np.float32)
    else:
        # Regression
        return df.as_matrix(result).astype(np.float32), df.as_matrix([target]).astype(np.float32)


def prepare_dataset(df):
    columns_to_drop = ['_id', 'gameId', 'frameNo', 'participantId', 'patch']
    for column in columns_to_drop:
        if column in df.columns: df.drop(columns=[column], inplace=True)
    c = []
    for column in df.columns:
        if df[column].dtypes not in (np.int32, np.int64, np.float, np.double):
            c.append(column)
    df = one_hot_encode_columns(df, c)
    x, y = to_xy(df, 'itemId')
    scaler = MinMaxScaler()
    scaler.fit(x)
    x = scaler.transform(x)
    return x, y


def filter_data_set(events):  # Corrupting Potions stays!
    events = events[events.itemId.isin(const.ADC_RELEVANT_ITEMS)]
    # events.loc[events['itemId'] == 2010, ['itemId']] = 2003  # Health Potions and Biscuits
    # events = events.loc[events['itemId'] != 2055]  # Control Wards
    # events = events.loc[events['itemId'] != 2003]  # Health Potions and Biscuits
    # # events = events.loc[events['itemId'] != 2031]  # Refillable Potion
    # events = events.loc[~events['itemId'].isin([2011, 2140, 2139, 2138])]  # Elixir of Skill, Wrath, Sorcery and Iron
    # events = events.loc[~events['itemId'].isin([3340, 3363, 3341])]  # Warding Totem, Farsight Aleration, Sweeping Lens
    return events


# "not outliers"
def get_regular():
    mongo = get_mongo_connection()
    regular = list(mongo.adc_numberItems.find({"count": {"$gte": 10}}, {"$project": "id"}))
    return [d['_id'] for d in regular]


def get_purchases_blank(champions=None, patches=None, tiers=None, limit=None, timeseries=False, min_purch=10):
    mongo = get_mongo_connection()
    if not timeseries:
        events = get_data_frame(mongo.adc_purchase_details, limit=limit, champions=champions, patches=patches, tiers=tiers)
    else:
        events = get_events_as_timeseries(champions=champions, patches=patches, tiers=tiers, limit=limit, min_purch=min_purch)
    events.drop(columns=events.columns.difference(common.columns_blank), inplace=True)
    events = filter_data_set(events)
    print('Join Frames...')
    events = join_frames(events)
    print('Join Gold Values ...')
    item_infos = get_item_infos()
    events = pd.merge(events, item_infos[["itemId", "patch", "goldSell", "goldBase", "goldTotal"]],
                      how="left", on=["itemId", "patch"], copy=False)
    print('Calculate available Gold ...')
    events = compute_reward_gold(events)
    events = filter_data_set(events)
    print('Join Itemstats by Patch ...')
    events = pd.merge(events, get_item_stats(), how='left', on='patch', copy=False)
    return events


def get_purchases_pre_game(champions=None, patches=None, tiers=None, limit=None, timeseries=False, min_purch=10):
    mongo = get_mongo_connection()
    if not timeseries:
        events = get_data_frame(mongo.adc_purchase_details, limit=limit, champions=champions, patches=patches, tiers=tiers)
    else:
        events = get_events_as_timeseries(champions=champions, patches=patches, tiers=tiers, limit=limit, min_purch=min_purch)
    events.drop(columns=events.columns.difference(common.columns_pre_game), inplace=True)
    events = filter_data_set(events)
    print('Join Frames...')
    events = join_frames(events)
    print('Join Gold Values ...')
    item_infos = get_item_infos()
    events = pd.merge(events, item_infos[["itemId", "patch", "goldSell", "goldBase", "goldTotal"]],
                      how="left", on=["itemId", "patch"], copy=False)
    print('Calculate available Gold ...')
    events = compute_reward_gold(events)
    events = filter_data_set(events)
    print('Join Itemstats by Patch ...')
    events = pd.merge(events, get_item_stats(), how='left', on='patch', copy=False)
    return events


def get_purchases_in_game(champions=None, patches=None, tiers=None, limit=None, timeseries=False, min_purch=10):
    mongo = get_mongo_connection()
    if not timeseries:
        events = get_data_frame(mongo.adc_purchase_details, limit=limit, champions=champions, patches=patches, tiers=tiers)
    else:
        events = get_events_as_timeseries(champions=champions, patches=patches, tiers=tiers, limit=limit, min_purch=min_purch)
    events.drop(columns=events.columns.difference(common.columns_in_game), inplace=True)
    events = filter_data_set(events)
    print('Join Frames...')
    events = join_frames(events)
    print('Join Gold Values ...')
    item_infos = get_item_infos()
    events = pd.merge(events, item_infos[["itemId", "patch", "goldSell", "goldBase", "goldTotal"]],
                      how="left", on=["itemId", "patch"], copy=False)
    print('Calculate available Gold ...')
    events = compute_reward_gold(events)
    events = filter_data_set(events)
    print('Join Itemstats by Patch ...')
    events = pd.merge(events, get_item_stats(), how='left', on='patch', copy=False)
    return events


def get_purchases_inventory(champions=None, patches=None, tiers=None, limit=None, timeseries=False, min_purch=10):
    mongo = get_mongo_connection()
    if not timeseries:
        events = get_data_frame(mongo.adc_purchase_details, limit=limit, champions=champions, patches=patches, tiers=tiers)
    else:
        events = get_events_as_timeseries(champions=champions, patches=patches, tiers=tiers, limit=limit, min_purch=min_purch)
    events.drop(columns=events.columns.difference(common.columns_inventory), inplace=True)
    events = filter_data_set(events)
    print('Join Frames...')
    events = join_frames(events)
    print('Join Gold Values ...')
    item_infos = get_item_infos()
    events = pd.merge(events, item_infos[["itemId", "patch", "goldSell", "goldBase", "goldTotal"]],
                      how="left", on=["itemId", "patch"], copy=False)
    print('Calculate available Gold ...')
    events = compute_reward_gold(events)
    events = filter_data_set(events)
    print('Join Itemstats by Patch ...')
    events = pd.merge(events, get_item_stats(), how='left', on='patch', copy=False)
    return events


def get_purchases_performance(champions=None, patches=None, tiers=None, limit=None, timeseries=False, min_purch=10):
    mongo = get_mongo_connection()
    if not timeseries:
        events = get_data_frame(mongo.adc_purchase_details, limit=limit, champions=champions, patches=patches, tiers=tiers)
    else:
        events = get_events_as_timeseries(champions=champions, patches=patches, tiers=tiers, limit=limit, min_purch=min_purch)
    events.drop(columns=events.columns.difference(common.columns_performance), inplace=True)
    print('Join Frames...')
    events = join_frames(events)
    print('Join Gold Values ...')
    item_infos = get_item_infos()
    events = pd.merge(events, item_infos[["itemId", "patch", "goldSell", "goldBase", "goldTotal"]],
                      how="left", on=["itemId", "patch"], copy=False)
    print('Calculate available Gold ...')
    events = compute_reward_gold(events)
    events = filter_data_set(events)
    print('Join Itemstats by Patch ...')
    events = pd.merge(events, get_item_stats(), how='left', on='patch', copy=False)
    return events


def get_purchase_teams(champions=None, patches=None, tiers=None, limit=None, timeseries=False, min_purch=10):
    mongo = get_mongo_connection()
    if not timeseries:
        events = get_data_frame(mongo.adc_purchase_details, limit=limit, champions=champions, patches=patches)
    else:
        events = get_events_as_timeseries(champions=champions, patches=patches, tiers=tiers, limit=limit, min_purch=min_purch)
    events.drop(columns=events.columns.difference(common.columns_teams), inplace=True)
    print('Join Frames...')
    events = join_frames(events)
    print('Join Gold Values ...')
    item_infos = get_item_infos()
    events = pd.merge(events, item_infos[["itemId", "patch", "goldSell", "goldBase", "goldTotal"]],
                      how="left", on=["itemId", "patch"], copy=False)
    print('Calculate available Gold ...')
    events = compute_reward_gold(events)
    events = filter_data_set(events)
    print('Join Itemstats by Patch ...')
    events = pd.merge(events, get_item_stats(), how='left', on='patch', copy=False)
    return events


def get_data_set_0_all():
    mongo = get_mongo_connection()
    events = get_data_frame(mongo.adc_purchase_details, limit=10000)
    events.drop(columns=events.columns.difference(common.columns_blank), inplace=True)

    return events


# purchases only with timestamps
def get_data_set_0() -> pd.DataFrame:
    mongo = get_mongo_connection()
    events = get_data_frame(mongo.jhin_training_set)
    events.drop(columns=events.columns.difference(common.columns_blank), inplace=True)
    events = filter_data_set(events)
    return events


# only events
def get_data_set_1() -> pd.DataFrame:
    mongo = get_mongo_connection()
    events = get_data_frame(mongo.adc_purchase_details)
    events = events[events.itemId.isin(get_regular())]
    return events


# filtered out wards, potions and trinkets
def get_data_set_2() -> pd.DataFrame:
    events = get_data_set_1()
    return events


# combined with frames (gold, xp, etc.) UNFILTERED
def get_data_set_3a() -> pd.DataFrame:
    events = get_data_set_1()
    events = join_frames(events)
    return events


# combined with frames (gold, xp, etc.) FILTERED
def get_data_set_3b() -> pd.DataFrame:
    events = get_data_set_2()
    events = join_frames(events)
    return events


# join item infos and calculate estimated available gold
def get_data_set_4() -> pd.DataFrame:
    events = get_data_set_3a()
    item_infos = get_item_infos()
    events = pd.merge(events, item_infos[["itemId", "patch", "goldSell", "goldBase", "goldTotal"]],
                      how="inner", on=["itemId", "patch"], copy=False)
    # noinspection PyTypeChecker
    events["Cost"] = np.where(events["type"] == "ITEM_PURCHASED", events["goldBase"], events["goldBase"] * -1)
    # noinspection PyTypeChecker
    events["Cost"] = np.where(events["type"] == "ITEM_SOLD", events["goldSell"] * -1, events["Cost"])
    events['frameCost'] = events.groupby(['gameId', 'platformId', 'frameNo'])['Cost'].cumsum()
    events['availGold'] = events['currentGold'] - events.groupby(['gameId', 'platformId', 'frameNo'])['frameCost'] \
        .shift(1).fillna(0)
    events = events.drop(columns=["frameCost", "Cost", "goldBase", "goldSell", "goldTotal", "currentGold"])
    return events


# improve estimation on available gold
def get_data_set_5() -> pd.DataFrame:
    killreward = 300
    killassist = 150
    turretreward = 300
    turretassist = 125
    events = get_data_set_4()
    events['killBounty'] = \
        events.sort_values(by=['platformId', 'gameId', 'frameNo', 'timestamp']).groupby(['platformId', 'gameId'])['kills'] \
        .diff(1).fillna(0) * killreward
    events['assistsBounty'] = \
        events.sort_values(by=['platformId', 'gameId', 'frameNo', 'timestamp']).groupby(['platformId', 'gameId'])['assists'] \
        .diff(1).fillna(0) * killassist
    events['turretKillBounty'] = \
        events.sort_values(by=['platformId', 'gameId', 'frameNo', 'timestamp']).groupby(['platformId', 'gameId'])['turretKills'] \
        .diff(1).fillna(0) * turretreward
    events['turretAssistBounty'] = \
        events.sort_values(by=['platformId', 'gameId', 'frameNo', 'timestamp']).groupby(['platformId', 'gameId'])['turretAssists'] \
        .diff(1).fillna(0) * turretassist
    events['availGold'] = \
        events['availGold'] + events['killBounty'] + events['assistsBounty'] + \
        events['turretKillBounty'] + events['turretAssistBounty']
    return events


# mostly for tree classifiers
def factorise_columns(df, columns):
    for column in columns:
        column_fact, column_keys = pd.factorize(df[column])
        df[column + "_fact"] = column_fact
    return df
