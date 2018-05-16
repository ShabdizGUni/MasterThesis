from pymongo import MongoClient, ASCENDING
import data.Pipelines as pl
import data.constants as const
from datetime import datetime


def main():
    mongoDB = MongoClient('mongodb://localhost:27017/').LeagueCrawler
    globalStart, start = datetime.now(), datetime.now()

    mongoDB.adc_events.aggregate(pl.get_skill_level_ups(with_frames=True), allowDiskUse=True)
    print("Finished Skill Level Ups at: ", str(datetime.now() - globalStart))

if __name__ == "__main__":
    main()
