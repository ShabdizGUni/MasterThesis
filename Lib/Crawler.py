import enum
import threading
import os
import re
import json
import random
from datetime import datetime
from pymongo import MongoClient
from Lib.API.APIRequests import *
from collections import deque
from Lib.patches import *
from pprint import pprint

class EntrySummoner(enum.Enum):
    KR = "SSG Ambiton"
    EUW1 = "lur1keen"
    NA1 = "Pobelter"
    BR1 = "4LaN"


class Crawler(threading.Thread):

    def __init__(self, region: str, matchesPerPatch: int):
        threading.Thread.__init__(self)
        self.region = region.upper()
        self.matchesPerPatch = matchesPerPatch
        self.req: APIRequester = APIRequester(os.environ.get("DEV_KEY"), region=self.region)
        self.client = MongoClient('mongodb://localhost:27017/')
        self.db = self.client.LeagueCrawler.matchDetails
        self.start_time = datetime.datetime.now()


    def run(self):
        self.fetch()

    def get_matches(self) -> list:
        return self.db.distinct("gameId", {"platformId": self.region})

    def get_summoners(self) -> list:
        return self.db.distinct("participantIdentities.player.currentAccountId", {"platformId": self.region})

    def get_versions(self) -> dict:
        detailedPatches = {r["_id"]: r["count"] for r in list(self.db.aggregate([
            {"$match": {
                "platformId": self.region
            }
            },
            {"$group": {
                "_id": "$gameVersion",
                "count": {"$sum": 1}
            }
            }
        ]))}
        #print(str(self.region) + " Patches in Detail: " + str(detailedPatches))
        patchCycles = {p.patch: 0 for p in PATCH_LIST}
        for k, v in detailedPatches.items():
            patchCycles[re.match("\d.\d+", k, flags=0).group(0)] += v
        print(str(self.region) + ": " + str(patchCycles))
        return patchCycles


    def fetch(self):
        req = self.req
        fetchedMatches = self.get_matches()
        strdSum = self.get_summoners() # stored summoners in matchDetails collection
        unfetchedSummoners = deque()
        fetchedSummoners = []
        versions = self.get_versions()
        pprint(versions)
        unfetchedSummoners.append(
            self.req.requestSummonerData(self.region, EntrySummoner[self.region].value)["accountId"])
        for s in random.sample(strdSum, len(strdSum)): unfetchedSummoners.append(s) #random order of summoners
        while len(unfetchedSummoners) > 0:
            versions = self.get_versions() #update Versions in case something halfway still went wrong
            summoner = unfetchedSummoners.popleft()
            if summoner in fetchedSummoners:
                pprint("Summoner: " + str(summoner) + " already processed, continue to next")
                continue

            pprint(self.region + " : Next Summoner ID" + str(summoner))
            for attempt in range(1, 10):
                try:
                    currentMatchlist = req.requestMatchList(self.region, summoner, queue=420)
                    break
                except Exception as e:
                    pprint(self.region + " {} :".format(
                        datetime.datetime.now() - self.start_time) + ": " + e.message + " when trying to fetch Matchlist from " + str(
                        summoner))
                    time.sleep(10)
            else:
                pprint("Exceeded 10 attempts. Going to next Summoner")
                continue
            for match in currentMatchlist:
                patch = getPatchFromTimestamp(match['timestamp'])
                if match['gameId'] not in fetchedMatches and patch in versions.keys() and versions[patch] < self.matchesPerPatch and self.region == match['platformId']:
                    for attempt in range(1, 10):
                        try:
                            details = req.requestMatchData(self.region, match['gameId'])
                            details["timeline"] = req.requestMatchTimelineData(self.region, match['gameId'])
                            break
                        except Exception as e:
                            pprint(self.region + " {} :".format(
                                datetime.datetime.now() - self.start_time) + e.message + " when trying to fetch Match " + str(
                                match["gameId"]))
                            time.sleep(10)
                    else:
                        pprint("Exceeded 10 attempts. Going to next Match")
                        continue
                    ids = details["participantIdentities"]
                    for id in ids:
                        if id["player"]["currentAccountId"] not in fetchedSummoners:
                            unfetchedSummoners.append(id["player"]["currentAccountId"])
                    fetchedMatches.append(details['gameId'])
                    versions[patch] += 1
                    self.db.insert_one(details)
                    print(self.region+": "+ str(versions))
            fetchedSummoners.append(summoner)
            ("All games from Summoner: " + str(summoner) + " processd! Continue to next")
            if all(v >= self.matchesPerPatch for v in versions.values()): break
