import datetime
import time
from typing import Optional as _Optional


class Patch:

    def __init__(self, name: str, patch: str, start: datetime.date, end: _Optional[datetime.date]):
        self.name = name
        self.patch = patch
        self.start = start
        self.end = end

PATCH_LIST = [ # changed the dates to circumvent different patch deployment dates throughout all regions as much as possible
    #Patch(name="Season 6", patch="6.22", start=datetime.date(2016, 11, 10), end=datetime.date(2016, 11, 21)), # start=datetime.date(2016, 11, 10), end=datetime.date(2016, 11, 22)
    Patch(name="Season 6", patch="6.23", start=datetime.date(2016, 11, 23), end=datetime.date(2016, 12, 6)), # start=datetime.date(2016, 11, 22), end=datetime.date(2016, 12, 7)
    Patch(name="Season 6", patch="6.24", start=datetime.date(2016, 12, 8), end=datetime.date(2017, 1, 10)), # start=datetime.date(2016, 12, 7), end=datetime.date(2017, 1, 11)
    Patch(name="Season 7", patch="7.1", start=datetime.date(2017, 1, 12), end=datetime.date(2017, 1, 24)), # start=datetime.date(2017, 1, 11), end=datetime.date(2017, 1, 25)
    Patch(name="Season 7", patch="7.2", start=datetime.date(2017, 1, 26), end=datetime.date(2017, 2, 7)), # start=datetime.date(2017, 1, 25), end=datetime.date(2017, 2, 8)
    Patch(name="Season 7", patch="7.3", start=datetime.date(2017, 2, 9), end=datetime.date(2017, 2, 22)), # start=datetime.date(2017, 2, 8), end=datetime.date(2017, 2, 23)
    Patch(name="Season 7", patch="7.4", start=datetime.date(2017, 2, 24), end=datetime.date(2017, 3, 7)), # start=datetime.date(2017, 2, 23), end=datetime.date(2017, 3, 8)
    Patch(name="Season 7", patch="7.5", start=datetime.date(2017, 3, 9), end=datetime.date(2017, 3, 21)), # start=datetime.date(2017, 3, 8), end=datetime.date(2017, 3, 22)
    Patch(name="Season 7", patch="7.6", start=datetime.date(2017, 3, 23), end=datetime.date(2017, 4, 4)), #start=datetime.date(2017, 3, 22), end=datetime.date(2017, 4, 5))
    Patch(name="Season 7", patch="7.7", start=datetime.date(2017, 4, 6), end=datetime.date(2017, 4, 18)), # start=datetime.date(2017, 4, 5), end=datetime.date(2017, 4, 19)
    Patch(name="Season 7", patch="7.8", start=datetime.date(2017, 4, 20), end=datetime.date(2017, 5, 2)), # start=datetime.date(2017, 4, 19), end=datetime.date(2017, 5, 3)
    Patch(name="Season 7", patch="7.9", start=datetime.date(2017, 5, 4), end=datetime.date(2017, 5, 16)), #start=datetime.date(2017, 5, 3), end=datetime.date(2017, 5, 17)
    Patch(name="Season 7", patch="7.10", start=datetime.date(2017, 5, 18), end=datetime.date(2017, 5, 30)), # start=datetime.date(2017, 5, 17), end=datetime.date(2017, 6, 1)
    Patch(name="Season 7", patch="7.11", start=datetime.date(2017, 6, 2), end=datetime.date(2017, 6, 13)), # start=datetime.date(2017, 6, 1), end=datetime.date(2017, 6, 14)
    Patch(name="Season 7", patch="7.12", start=datetime.date(2017, 6, 15), end=datetime.date(2017, 6, 27)), # start=datetime.date(2017, 6, 14), end=datetime.date(2017, 6, 28)
    Patch(name="Season 7", patch="7.13", start=datetime.date(2017, 6, 29), end=datetime.date(2017, 7, 11)),# start=datetime.date(2017, 6, 28), end=datetime.date(2017, 7, 12)
    Patch(name="Season 7", patch="7.14", start=datetime.date(2017, 7, 13), end=datetime.date(2017, 7, 25)),# start=datetime.date(2017, 7, 12), end=datetime.date(2017, 7, 26)
    Patch(name="Season 7", patch="7.15", start=datetime.date(2017, 7, 27), end=datetime.date(2017, 8, 8)), # start=datetime.date(2017, 7, 26), end=datetime.date(2017, 8, 9)
    Patch(name="Season 7", patch="7.16", start=datetime.date(2017, 8, 10), end=datetime.date(2017, 8, 22)),# start=datetime.date(2017, 8, 9), end=datetime.date(2017, 8, 23)
    Patch(name="Season 7", patch="7.17", start=datetime.date(2017, 8, 24), end=datetime.date(2017, 9, 12)), #start=datetime.date(2017, 8, 23), end=datetime.date(2017, 9, 13)
    Patch(name="Season 7", patch="7.18", start=datetime.date(2017, 9, 14), end=datetime.date(2017, 9, 26)), # start=datetime.date(2017, 9, 13), end=datetime.date(2017, 9, 27)
    Patch(name="Season 7", patch="7.19", start=datetime.date(2017, 9, 28), end=datetime.date(2017, 10, 11)),
    Patch(name="Season 7", patch="7.20", start=datetime.date(2017, 10, 13), end=datetime.date(2017, 10, 24)),
    Patch(name="Season 7", patch="7.21", start=datetime.date(2017, 10, 26), end=datetime.date(2017, 11, 7))
]


def getAllPatches():
    return [p.name for p in PATCH_LIST]


def getPatchFromTimestamp(timestamp: int):
    for patch in PATCH_LIST:
        start = time.mktime(patch.start.timetuple()) * 1000
        end = time.mktime(patch.end.timetuple()) * 1000 if patch.end is not None else float('inf')
        if start <= timestamp < end:
            return patch.patch
