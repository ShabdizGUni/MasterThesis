from pymongo import MongoClient, ASCENDING
import data.Pipelines as pl
import data.constants as const
from datetime import datetime


def main():
    mongoDB = MongoClient('mongodb://localhost:27017/').LeagueCrawler
    globalStart, start = datetime.now(), datetime.now()

    # events:
    mongoDB.matchDetails.aggregate(pl.get_adc_events(), allowDiskUse=True)
    print("Events finished after: " + str(datetime.now() - start))

    # frames:
    start = datetime.now()
    print("Start Frames after: " + str(datetime.now() - start))
    mongoDB.matchDetails.aggregate(pl.get_frames(), allowDiskUse=True)
    print("Frames finished after: " + str(datetime.now() - start))
    mongoDB.adc_frames.create_index([('championId', ASCENDING)])
    mongoDB.adc_frames.create_index([('championId', ASCENDING),
                                     ('patch', ASCENDING)])
    mongoDB.adc_frames.create_index([('gameId', ASCENDING)])

    # purchase details:
    start = datetime.now()
    print("Start Purchase Details: " + str(datetime.now()))
    mongoDB.adc_events.aggregate(pl.get_purchase_details(), allowDiskUse=True)
    print("Purchase Details finished after: " + str(datetime.now() - start))

    print("Start Indexing at: " + str(datetime.now() - start))
    mongoDB.adc_purchase_details.create_index([("championId", ASCENDING)])
    mongoDB.adc_purchase_details.create_index([("championId", ASCENDING), ("patch", ASCENDING)])
    print("Script finished after: " + str(datetime.now() - globalStart))

    mongoDB.adc_events.create_index([("championId", ASCENDING),
                                     ("gameId", ASCENDING),
                                     ("timestamp", ASCENDING)
                                     ])
    start = datetime.now()
    print("Indexing Events finished after: " + str(datetime.now() - start))

    mongoDB.adc_events.create_index([("championId", ASCENDING),
                                     ("gameId", ASCENDING),
                                     ("frameNo", ASCENDING)
                                     ])
    start = datetime.now()
    print("Indexing Frames finished after: " + str(datetime.now() - start))

    print("Start Counting Items at: ", str(datetime.now()))
    mongoDB.adc_purchase_details.aggregate(pl.get_item_count(), allowDiskUse=True)
    print("Finished Counting Items at: ", str(datetime.now() - globalStart))

    print("Start Counting Items per Match at: ", str(datetime.now()))
    # irrel_items = list(mongoDB.adc_numberItems.find({"count": {"$lt": 10}}, {"$project": "_id"}))
    # item_list = [item['_id'] for item in irrel_items]
    mongoDB.adc_purchase_details.aggregate(pl.create_purchase_detail_ids(const.ADC_RELEVANT_ITEMS),
                                           allowDiskUse=True)
    print("Finished Counting Items per Match at: ", str(datetime.now() - globalStart))

if __name__ == "__main__":
    main()
