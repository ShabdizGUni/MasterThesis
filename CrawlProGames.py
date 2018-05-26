from pprint import pprint
from Lib.API.APIRequests import ProGameRequester
from pymongo import *
import pandas as pd
import re

def main():
    req = ProGameRequester()
    db = MongoClient('mongodb://localhost:27017/').LeagueCrawler

    path = 'C:\\Users\Shabdiz\PycharmProjects\MasterThesis\Input'
    urlRoot = "http://matchhistory.na.leagueoflegends.com/en/#match-details/"

    # spring = pd.read_excel(path + "\\2017spring.xlsx", sheetname="Sheet1")
    # summer = pd.read_excel(path + "\\2017summer.xlsx", sheetname="Sheet1")
    # progames = pd.concat([spring, summer])
    progames = pd.read_excel(path + "\\2017matchdataOraclesElixir.xlsx", sheet_name="Sheet1")
    # LPL Entries are hand edited

    df = progames[progames.league != "LPL"]
    urls = list(set(df.url))
    fetchedGames = list(db.matchDetailsPro.aggregate([
        {"$group": {
            "_id" : {
                "gameId" : "$gameId",
                "platformId": "$platformId"
                }
            }
        },
        {"$project": {
            "gameId": "$_id.gameId",
            "platformId" :"$_id.platformId"
            }
        }
    ]))
    print(fetchedGames)
    for url in urls:
        #Raw string notation (r"text") keeps regular expressions sane, see https://docs.python.org/3.1/library/re.html#raw-string-notation
        #tail = re.match(r".*match-details/(.*)",url).group(1)
        captures = re.match(r".*match-details/(.*)/(.*)\?gameHash=(\w*\d*)", url)
        platformId = captures.group(1)
        gameId = captures.group(2)
        gameHash = captures.group(3)
        if {"gameId": gameId, "platformId": platformId} in fetchedGames:
            print("gameId:" +str(gameId) + "," + str(platformId) + " already fetched, continue..." )
            continue
        for attempt in range(1, 50):
            try :
                response = req.requestMatchData(platformId, gameId, gameHash)
                response['timeline'] = req.requestTimelineData(platformId, gameId, gameHash)
                insert = db.matchDetailsPro.insert_one(response)
                pprint(insert.inserted_id)
                break
                print("Was not successful at game {:s}".format(gameId))
            except Exception as e:
                print("Exeption at attempt {:d}".format(attempt))
                print(e)



if __name__ == "__main__":
    main()
