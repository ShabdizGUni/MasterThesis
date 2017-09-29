from pymongo import MongoClient
from pprint import pprint
from Lib.patches import PATCH_LIST
import re

def main():

    client = MongoClient('mongodb://localhost:27017/')
    db = client.LeagueCrawler.matchDetails
    detailedPatches = {r["_id"] : r["count"] for r in db.aggregate([
        {"$match":{
            "platformId": "NA1"
            }
        },
        {"$group": {
            "_id":"$gameVersion",
            "count":{
                "$sum":1
            }
        }}
    ]) }
    patchCycles = {p.patch: 0 for p in PATCH_LIST}
    for k, v in detailedPatches.items():
        patchCycles[re.match("\d.\d+", k, flags=0).group(0)] += v
    del (patchCycles["7.16"])
    del (patchCycles["6.22"])
    pprint(patchCycles)


if __name__ == "__main__":
    main()
