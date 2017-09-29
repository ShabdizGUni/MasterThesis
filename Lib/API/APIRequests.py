import requests
import time
import datetime
from pprint import pprint
from cassiopeia.type.dto.match import *

class RateLimitException(Exception):
    def __init__(self):
        super(Exception).__init__()
        self.message = "Rate Limit exceeded"

class MatchNotFoundException(Exception):
    def __init__(self):
        super(Exception).__init__()
        self.message = "Match not found"

class BadRequestException(Exception):
    def __init__(self):
        super(Exception).__init__()
        self.message = "Bad Request"

class ServiceException(Exception):
    def __init__(self):
        super(Exception).__init__()
        self.message = "Service unavailable"

class InternalServerErrorException(Exception):
    def __init__(self):
        super(Exception).__init__()
        self.message = "Internal Server Error"

class UnauthorizedException(Exception):
    def __init__(self):
        super(Exception).__init__()
        self.message = "Unauthorized"

class ForbiddenException(Exception):
    def __init__(self):
        super(Exception).__init__()
        self.message = "Forbidden"

class APIRequester(object):

    def __init__(self, apiKey:str, region:str):
        self.__apiKey:str = apiKey
        self.region = region
        self.root:str = ".api.riotgames.com/lol/"

        ##FOR GAME CONSTANTS REFER TO: https://developer.riotgames.com/game-constants.html

        # SEASONS
        #
        # SEASONNAME          ID
        # PRESEASON3          0
        # SEASON3             1
        # PRESEASON2014       2
        # SEASON2014          3
        # PRESEASON2015       4
        # SEASON2015          5
        # PRESEASON2016       6
        # SEASON2016          7
        # PRESEASON2017       8


    @property
    def Versions(self):
        url = "https://"+ self.region + self.root +"static-data/v3/versions?api_key=" + self.__apiKey
        print(url)
        response = requests.get(url)
        return response.json()

    def RateLimited(maxPerSecond):
        minInterval = 1.0 / float(maxPerSecond)
        def decorate(func):
            lastTimeCalled = [0.0]
            def rateLimitedFunction(*args, **kargs):
                elapsed = time.clock() - lastTimeCalled[0]
                leftToWait = minInterval - elapsed
                if leftToWait > 0:
                    time.sleep(leftToWait)
                ret = func(*args, **kargs)
                lastTimeCalled[0] = time.clock()
                return ret
            return rateLimitedFunction
        return decorate

    @RateLimited(0.833)
    def sendRequest(self, url):
        for attempt in range(1,10):
            try:
                response = requests.get(url)
                try:
                    if response.json()['status']['status_code'] == 400:
                        raise BadRequestException
                    elif response.json()['status']['status_code'] == 401:
                        raise UnauthorizedException
                    elif response.json()['status']['status_code'] == 403:
                        raise ForbiddenException
                    elif response.json()['status']['status_code'] == 404:
                        raise MatchNotFoundException
                    elif response.json()['status']['status_code'] == 429:
                        raise RateLimitException
                    elif response.json()['status']['status_code'] == 500:
                        raise InternalServerErrorException
                    elif response.json()['status']['status_code'] == 503:
                        raise ServiceException
                except KeyError:
                    break
                except (RateLimitException, InternalServerErrorException, ServiceException):
                    time.sleep(10)
                    pass
            except ConnectionError:
                seconds = attempt * 30
                print("Connection Error! Retry in "+ seconds +" seconds!")
                time.sleep(seconds)
        return response.json()

    def requestChampionData(self, region:str, version = None):
        if version is None: version = self.currentVersion
        #https://euw1.api.riotgames.com/lol/static-data/v3/champions?locale=en_GB&tags=all&dataById=true&api_key=RGAPI-3f33e490-afd7-450b-81d3-94abd670cc58
        url = "https://"+region+self.root+"static-data/v3/champions?locale=en_GB&tags=all&dataById=true&version="+version+"&api_key=" + self.__apiKey
        print(url)
        return self.sendRequest(url)

    def requestItemData(self, region:str, version = None):
        if version is None: version = self.currentVersion
        #https://euw1.api.riotgames.com/lol/static-data/v3/items?api_key=d0e89ead-8392-4faf-b6f8-d10c24b2cef7
        #https://ru.api.riotgames.com/lol/static-data/v3/items?locale=en_GB&version=7.15.1&tags=all&api_key=RGAPI-2b8652c3-163a-4077-8fa7-846d1c11990e
        url = "https://"+region+self.root+"static-data/v3/items?locale=en_GB&tags=all&version="+version+"&api_key=" + self.__apiKey
        print(url)
        return self.sendRequest(url)

    def requestSummonerData(self, region, summonerName):
        # https: // euw1.api.riotgames.com / lol / summoner / v3 / summoners / by - name / G2 % 20Ups3t?api_key = d0e89ead - 8392 - 4
        # faf - b6f8 - d10c24b2cef7
        summonerName = summonerName.replace(" ","%20")
        url = "https://" + region + self.root + "summoner" + \
              "/v3/summoners/by-name/" + summonerName + "?api_key=" + self.__apiKey
        print(url)
        return self.sendRequest(url)

    def requestRankedData(self, region, _id):
        url = "https://" + region + self.root + region + \
              "/v2.5/league/by-summoner/" + _id + "/entry?api_key=" + self.__apiKey
        print(url)
        return self.sendRequest(url)

    def requestMatchData(self, region: str, matchId):
        url = "https://" + region + self.root + \
            "match/v3/matches/" + str(matchId) + "?api_key=" + self.__apiKey
        print(url)
        return self.sendRequest(url)

    def requestMatchTimelineData(self, region, matchId):
        url = "https://" + region + self.root + \
            "match/v3/timelines/by-match/" + str(matchId) + "?api_key=" + self.__apiKey
        print(url)
        return self.sendRequest(url)

    def requestMatchList(self, region, _id, queue:int = None, season = None, beginTime = None, endTime = None):
        beginIndex = 0
        endIndex = 100
        seasonInfo:str = ""
        begin:str = ""
        end:str = ""
        queueInfo:str = ""
        matchlist = []
        #Season Param
        if season is not None:
            if type(season) is list:
                for s in season:
                    seasonInfo += "season=" + str(s) + "&"
            if type(season) is int: seasonInfo += "season=" + str(season) + "&"
        #Time Parameters

        if beginTime is not None:
            begin = "beginTime=" + str(int(time.mktime(time.strptime(beginTime, "%d.%m.%Y"))) * 1000) + "&"
        if endTime is not None:
            end = "endTime=" + str(int(time.mktime(time.strptime(endTime, "%d.%m.%Y")))*1000) + "&"
        #Queue Param
        if queue is not None:
            queueInfo = "queue=" + str(queue) + "&"
        #URL
        url = "https://" + region + self.root + "match" + \
              "/v3/matchlists/by-account/" + str(_id) + "?" + seasonInfo + queueInfo + begin + end + "beginIndex="+str(beginIndex)+"&endIndex="+str(endIndex)+ "&api_key=" + self.__apiKey
        print(url)
        response =  self.sendRequest(url)
        matchlist = response['matches']
        numgames = response["totalGames"]
        pprint("beginIndex: " + str(beginIndex) + " endIndex: " + str(endIndex) + "  totalGames: "+ str(numgames))
        if numgames < endIndex:
            return matchlist
        else :
            while endIndex < numgames:
                beginIndex += 100
                endIndex += 100
                url = "https://" + region + self.root + "match" + \
                      "/v3/matchlists/by-account/" + str(
                    _id) + "?" + seasonInfo + queueInfo + begin + end + "beginIndex=" + str(
                    beginIndex) + "&endIndex=" + str(endIndex) + "&api_key=" + self.__apiKey
                response = self.sendRequest(url)
                matchlist += response['matches']
                numgames = response['totalGames']
                pprint("beginIndex: " + str(beginIndex) + " endIndex: " + str(endIndex) + "  totalGames: " + str(numgames))
        return matchlist
