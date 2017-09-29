from pymongo import MongoClient


def main():
    # patches = ["6\.23\..*", "6\.24\..*", "7\.1\..*", "7\.2\..*", "7\.3\..*", "7\.4\..*", "7\.5\..*", "7\.6\..*",
    #            "7\.7\..*", "7\.8\..*", "7\.9\..*", "7\.10\..*", "7\.11\..*", "7\.12\..*", "7\.13\..*", "7\.15\..*"]

    regions = ["EUW1", "KR", "NA1", "BR1"]
    db = MongoClient('mongodb://localhost:27017/').LeagueCrawler
    # for patch in patches:
    for region in regions:
        matches = db.matchDetails.find({"gameVersion": {"$regex": "7.14\..*"}, "platformId": region}).limit(2000)
        result = db.matchDetails2000.insert_many(matches)
        games = len(result.inserted_ids)
        print("Inserted Region "+region+": "+str(games))


if __name__ == "__main__":
    main()
