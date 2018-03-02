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
    KR = "KSV AMBITlON"
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
        self.db = self.client.LeagueCrawler['matchDetails2000']
        self.start_time = datetime.datetime.now()


    def run(self):
        self.fetch()

    def get_matches(self) -> list:
        return self.db.distinct("gameId", {"platformId": self.region})

    def get_fetched_summoners(self) -> list:
        return self.db.fetchedSummoners.distinct("id", {"platformId": self.region})

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
        patchCycles = {p.patch: 0 for p in PATCH_LIST}
        for k, v in detailedPatches.items():
            patchCycles[re.match("\d.\d+", k, flags=0).group(0)] += v
        print(str(self.region) + ": " + str(patchCycles))
        return patchCycles


    def fetch(self):
        req = self.req
        #fetchedMatches = self.get_matches()
        strdSum = self.get_summoners() # stored summoners in matchDetails collection
        unfetchedSummoners = deque()
        fetchedSummoners = self.get_fetched_summoners()
        versions = self.get_versions()
        print(self.region+": "+ str(versions))
        iteration = 0
        if not strdSum: #first run
            unfetchedSummoners.append(
                self.req.requestSummonerData(self.region, EntrySummoner[self.region].value)["accountId"])
        else:
            for s in list(set(strdSum) - set(fetchedSummoners)): unfetchedSummoners.append(s)
        print("Number of Summoners in MatchDetails:" +"\t"+ str(len(strdSum)) + "\n" +
              "Number of Summoners already completely processed:" +"\t" + str(len(fetchedSummoners)) + "\n" +
              "Number of Summoners to process:" +"\t" + str(len(unfetchedSummoners)))

        # AS LONG AS THERE ARE SUMMONERS TO PROCESS:
        while len(unfetchedSummoners) > 0:
            summoner = unfetchedSummoners.popleft()
            if summoner in fetchedSummoners:
                pprint("Summoner: " + str(summoner) + " already processed, continue to next")
                continue
            pprint(self.region + " : Next Summoner ID" + str(summoner))

            #GET MATCHLIST OF SUMMONER
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

            # ITERATE THROUGH ALL HIS RANKED GAMES IN (PRE-)SEASON 7
            for match in currentMatchlist:
                patch = getPatchFromTimestamp(match['timestamp'])
                #match['gameId'] not in fetchedMatches
                if not list(self.db.find({"platformId" : self.region, "gameId": match['gameId']})) and patch in versions.keys() and versions[patch] < self.matchesPerPatch and self.region == match['platformId']:
                    startfetch = datetime.datetime.now()
                    for attempt in range(1, 10):
                        try:
                            details = req.requestMatchData(self.region, match['gameId'])
                            details["timeline"] = req.requestMatchTimelineData(self.region, match['gameId'])
                            break
                        except ForbiddenException as e:
                            pprint(self.region + " {} :".format(
                                datetime.datetime.now() - self.start_time) + e.message + ": Change up your API Key! Script will abort for now" )
                            return
                        except requests.HTTPError as e:
                            pprint("HTTPError when trying to fetch match "+ str(match["gameId"]) + ". Trying again!")
                            continue
                        except Exception as e:
                            pprint(self.region + " {} :".format(
                                datetime.datetime.now() - self.start_time) + str(e) + " when trying to fetch Match " + str(
                                match["gameId"]))
                            time.sleep(10)
                    else:
                        pprint("Exceeded 10 attempts. Going to next Match")
                        continue

                    idents = details["participantIdentities"]
                    part = details["participants"]


                    # STORE RELEVANT SUMMONERS
                    for id in idents:
                        #partId = id["participantId"]
                        if id["player"]["currentAccountId"] not in fetchedSummoners:
                            unfetchedSummoners.append(id["player"]["currentAccountId"])

                    details["patch"] = re.match("\d.\d+", details["gameVersion"], flags=0).group(0)
                    #fetchedMatches.append(details['gameId'])
                    versions[patch] += 1

                    objId = self.db.insert_one(details)

                    pprint(self.region +": Stored Game " + str(details["gameId"]) + " from Patch " + str(details["patch"]) + " into MongoDB Collection with _id:" +
                           str(objId.inserted_id) + "after " + str((datetime.datetime.now() - startfetch).seconds) + " seconds")
                #elif len(list(self.db.find({"platformId" : self.region, "gameId": match['gameId']}))) > 0:
                    #pprint(self.region + ": GameId " + str(match['gameId']) + " already fetched! Skip..")

            #after processing summmoner:

            fetchedSummoners.append(summoner)
            strdSum = self.db.fetchedSummoners.insert_one({"id": summoner, "platformId": self.region})

            print(self.region + ": Completed processing Summoner " +str(summoner)+ "! Stored him into the MongoDB collection with the _id: " + str(strdSum.inserted_id))
            print(self.region + ": "+ str(versions))
            iteration += 1
            if  iteration % 15 == 0: # update every 15 summoners
                versions = self.get_versions() #update Versions in case something halfway still went wrong
            if all(v >= self.matchesPerPatch for v in versions.values()):
                print(self.region + ": " + str(versions))
                print(self.region + ": Completed crawling!")
                break
