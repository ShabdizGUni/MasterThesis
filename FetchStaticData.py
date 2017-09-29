from Lib.API.APIRequests import *
from data._Constants import  *
import os


def main():
    api = APIRequester(apiKey=os.environ["DEV_KEY"], region="EUW1")

    platforms = ALL_PLATFORMS
    #patches = PATCH_ITERATIONS
    #patches = ["7.1.1", "7.5.1", "7.12.1", "7.14.1"]
    patches = ["7.14.1"]
    it = 0

    for patch in patches:
        #region = platforms[it % len(platforms)]
        #champions = api.requestChampionData(region, patch)
        #db.champions.insert_one(champions)
        #it += 1
        region = platforms[it % len(platforms)]
        items = api.requestItemData(region, patch)
        db.items.insert_one(items)
        it += 1

if __name__ == "__main__":
    main()
