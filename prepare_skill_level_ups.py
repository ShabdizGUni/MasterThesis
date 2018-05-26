from pymongo import MongoClient, ASCENDING
import data.Pipelines as pl
import data.constants as const
from datetime import datetime


def main():
    mongoDB = MongoClient('mongodb://localhost:27017/').LeagueCrawler
    globalStart, start = datetime.now(), datetime.now()

    # mongoDB.adc_events.aggregate(pl.get_skill_level_ups(with_frames=True), allowDiskUse=True)
    # print("Start Indexing at: " + str(datetime.now() - start))
    # mongoDB.adc_skill_level_ups.create_index([("championId", ASCENDING)])
    # mongoDB.adc_skill_level_ups.create_index([("championId", ASCENDING), ("patch", ASCENDING)])
    mongoDB.adc_skill_level_ups.create_index([("gameId", ASCENDING),
                                              ("side", ASCENDING)])
    mongoDB.adc_skill_level_ups.create_index([("gameId", ASCENDING),
                                              ("patch", ASCENDING),
                                              ("tier", ASCENDING),
                                              ("side", ASCENDING),
                                              ("championId", ASCENDING)])

    print("Start Counting Level Ups per Match at: ", str(datetime.now()))
    mongoDB.adc_skill_level_ups.aggregate(pl.create_skill_level_ups_ids(),
                                          allowDiskUse=True)
    print("Finished Counting Level Ups per Match at: ", str(datetime.now() - globalStart))

    print("Finished Skill Level Ups at: ", str(datetime.now() - globalStart))

if __name__ == "__main__":
    main()
