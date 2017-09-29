from data._Constants import *
from data.Piplines import *


class CollectionCreator:

    @staticmethod
    def adcCounterPerPatchPerRagion():
        db.matchDetails2000.aggregate(MatchDetails2000.adcCounterPerPatchPerRagion, allowDiskUse=True)

    @staticmethod
    def adcCounterTotal():
        db.matchDetails2000.aggregate(MatchDetails2000.adcCountTotal, allowDiskUse=True)


