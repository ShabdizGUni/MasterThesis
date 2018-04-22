import requests
from pymongo import MongoClient
from data.constants import PATCH_ITERATIONS


def main():
    mongoDB = MongoClient('mongodb://localhost:27017/').LeagueCrawler
    masteries = mongoDB.masteries
    for patch in PATCH_ITERATIONS:
        url = "http://ddragon.leagueoflegends.com/cdn/" + patch + "/data/en_US/mastery.json"
        print(url)
        response = requests.get(url)
        inserted_id = masteries.insert_one(response.json())
        print(inserted_id)

if __name__ == "__main__":
    main()
