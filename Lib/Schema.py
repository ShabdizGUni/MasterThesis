from sqlalchemy import *
from sqlalchemy import orm
from sqlalchemy.ext.declarative import declarative_base

global MatchDetail, Team, TeamBan, Participant
Base = declarative_base()



class MatchDetail(Base,object):
    __tablename__ = 'MatchDetails'

    gameId = Column(BigInteger, primary_key=True)
    seasonId = Column(Integer)
    queueId = Column(Integer)
    participantIdentities = orm.relationship('ParticipantIdentity',
                                                        cascade='all, delete-orphan', passive_deletes=True)
    gameVersion = Column(String(30))
    gameMode = Column(String(30))
    mapId = Column(Integer)
    gameType = Column(String(30))
    teams = orm.relationship('Team', cascade="all, delete-orphan", passive_deletes=True)
    participants = orm.relationship('Participant', cascade="all, delete-orphan", passive_deletes=True)
    gameCreation = Column(BigInteger)
    gameDuration = Column(Integer)
    timeline = orm.relationship('MatchTimeline', uselist=False, cascade="all, delete-orphan", passive_deletes=True)

    def __init__(self, dict):
        self.gameId = str(dict.get("gameId", 0))
        self.seasonId = dict.get("seasonId", 0)
        self.queueId = dict.get("queueId", 0)
        self.gameVersion = dict.get("gameVersion", "")
        self.gameMode = dict.get("gameMode", "")
        self.mapId = dict.get("mapId", 0)
        self.gameType = dict.get("gameType", "")
        self.gameCreation = str(dict.get("gameCreation", 0))
        self.gameDuration = dict.get("gameDuration", 0)
        self.participantIdentities = [(ParticipantIdentity(p) if not isinstance(p, ParticipantIdentity) else p) for p in dict.get("participantIdentities",[]) if p]
        self.teams = [(Team(t) if not isinstance(t, Team) else t) for t in dict.get("teams", []) if t]
        self.participants = [(Participant(p) if not isinstance(p, Participant) else p) for p in dict.get("participants", []) if p]

    def addTimeline(self, t):
        self.timeline = t


class Team(Base):
    __tablename__ = 'Teams'

    _id = Column(BigInteger, primary_key=True)
    gameId = Column(BigInteger,
                    ForeignKey('MatchDetails.gameId', ondelete='CASCADE'))
    teamId = Column(Integer)
    bans = orm.relationship('TeamBan', cascade="all, delete-orphan", passive_deletes=True)
    baronKills = Column(Integer)
    dominionVictoryScore = Column(Integer)
    dragonKills = Column(Integer)
    firstBaron = Column(Boolean)
    firstBlood = Column(Boolean)
    firstDragon = Column(Boolean)
    firstInhibitor = Column(Boolean)
    firstTower = Column(Boolean)
    inhibitorKills = Column(Integer)
    towerKills = Column(Integer)
    vilemawKills = Column(Integer)
    winner = Column(Boolean)

    def __init__(self, dict:dict):
        self.baronKills = dict.get("baronKills", 0)
        self.teamId = dict.get("teamId",0)
        self.dominionVictoryScore = dict.get("dominionVictoryScore",0)
        self.dragonKills = dict.get("dragonKills",0)
        self.firstBaron = dict.get("firstBaron", False)
        self.firstBlood = dict.get("firstBlood", False)
        self.firstDragon = dict.get("firstDragon", False)
        self.firstInhibitor = dict.get("firstInhibitor", False)
        self.firstTower = dict.get("firstTower", False)
        self.inhibitorKills = dict.get("inhibitorKills", 0)
        self.towerKills = dict.get("towerKills", 0)
        self.vilemawKills = dict.get("vilemawKills", 0)
        self.winner = dict.get("winner", False)
        self.bans = [(TeamBan(t) if not isinstance(t, TeamBan) else t) for t in dict.get("bans", []) if t]


class Participant(Base):
    __tablename__ = 'Participants'

    _id = Column(BigInteger, primary_key=True)
    _gameId = Column(BigInteger, ForeignKey('MatchDetails.gameId'))
    participantId = Column(Integer)
    teamId = Column(Integer)
    spell1Id = Column(Integer)
    spell2Id = Column(Integer)
    highestAchievedSeasonTier = Column(String(30))
    championId = Column(Integer)
    stats = orm.relationship('Stats', uselist=False, cascade="all, delete-orphan",  passive_deletes=True)
    runes = orm.relationship('Rune', cascade="all, delete-orphan", passive_deletes=True)
    masteries = orm.relationship('Mastery', cascade="all, delete-orphan", passive_deletes=True)
    timeline = orm.relationship('Timeline', uselist=False, cascade="all, delete-orphan", passive_deletes=True)

    def __init__(self, dict:dict):
        self.participantId = dict.get("participantId", 0)
        self.teamId = dict.get("teamId", 0)
        self.spell1Id = dict.get("spell1Id", 0)
        self.spell2Id = dict.get("spell2Id", 0)
        self.highestAchievedSeasonTier = dict.get("highestAchievedSeasonTier", "")
        self.championId = dict.get("championId", 0)
        statsVal = dict.get("stats", None)
        runesVal = dict.get("runes", [])
        masteriesVal = dict.get("masteries", [])
        timelineVal = dict.get("timeline", None)
        self.stats = Stats(statsVal) if not isinstance(statsVal, Stats) else statsVal
        self.runes = [Rune(r) if not isinstance(r, Rune) else r for r in runesVal]
        self.masteries = [Mastery(m) if not isinstance(m, Mastery) else m for m in masteriesVal]
        self.timeline = Timeline(timelineVal) if not isinstance(timelineVal, Timeline) else timelineVal


class ParticipantIdentity(Base):
    __tablename__  = 'ParticipantIdentities'

    _id = Column(BigInteger, primary_key=True)
    _gameId = Column(BigInteger, ForeignKey('MatchDetails.gameId', ondelete='CASCADE'))
    participantId = Column(Integer)
    player = orm.relationship('Player', uselist=False, cascade='all, delete-orphan', passive_deletes=True)

    def __init__(self, dict:dict):
        self.participantId = dict.get("participantId", 0)
        playerVal = dict.get("player", None)
        self.player = Player(playerVal) if not isinstance(playerVal, Player) else playerVal

class Player(Base):
    __tablename__ = 'Players'

    _id = Column(BigInteger, primary_key=True)
    _participantIdentityId = Column(BigInteger, ForeignKey('ParticipantIdentities._id', ondelete='CASCADE'))
    currentPlatformId = Column(String(30))
    summonerName = Column(String(30))
    matchHistoryUri = Column(String(30))
    platformId = Column(String(30))
    currentAccountId = Column(BigInteger)
    profileIcon = Column(Integer)
    summonerId = Column(BigInteger)
    accountId = Column(BigInteger)

    def __init__(self, dict:dict):
        self.currentPlatformId = dict.get('currentPlatformId', 0)
        self.summonerName = dict.get('summonerName',"")
        self.matchHistoryUri = dict.get('machtHistoryUri', "")
        self.platformId = dict.get('platformId', "")
        self.currentAccountId = dict.get('currentAccountId', 0)
        self.profileIcon = dict.get('profileIcon', 0)
        self.summonerId = dict.get('summonerId', 0)
        self.accountId = dict.get('accountId', 0)

class TeamBan(Base):
    __tablename__ = 'TeamBans'

    _id = Column(BigInteger, primary_key=True)
    _teamId = Column(BigInteger, ForeignKey('Teams._id', ondelete="CASCADE"))
    pickTurn = Column(Integer)
    championId = Column(Integer)

    def __init__(self, dict:dict):
        self.pickTurn = dict.get("pickTurn",0)
        self.championId = dict.get("championId", 0)

class Stats(Base):
    __tablename__ = "Stats"

    _id = Column(BigInteger, primary_key=True)
    _participant_id = Column(BigInteger, ForeignKey('Participants._id', ondelete='CASCADE'))

    participantId = Column(Integer)
    win = Column(Boolean)
    item0 = Column(Integer)
    item1 = Column(Integer)
    item2 = Column(Integer)
    item3 = Column(Integer)
    item4 = Column(Integer)
    item5 = Column(Integer)
    item6 = Column(Integer)
    champLevel = Column(Integer)

    kills = Column(Integer)
    deaths = Column(Integer)
    assists = Column(Integer)
    largestMultikill = Column(Integer)
    killingSprees = Column(Integer)
    largestKillingSpree = Column(Integer)
    doubleKills = Column(Integer)
    tripleKills = Column(Integer)
    quadraKills = Column(Integer)
    pentaKills = Column(Integer)
    unrealKills = Column(Integer)

    turretKills = Column(Integer)
    inhibitorKills = Column(Integer)

    firstBloodKill = Column(Boolean)
    firstBloodAssist = Column(Boolean)
    firstTowerKill = Column(Boolean)
    firstTowerAssist = Column(Boolean)
    firstInhibitorKill = Column(Boolean)
    firstInhibitorAssist = Column(Boolean)

    totalDamageDealtToChampions = Column(BigInteger)
    physicalDamageDealtToChampions = Column(BigInteger)
    magicDamageDealtToChampions = Column(BigInteger)
    trueDamageDealtToChampions = Column(BigInteger)
    totalDamageDealt = Column(BigInteger)
    physicalDamageDealt = Column(BigInteger)
    magicDamageDealt = Column(BigInteger)
    trueDamageDealt = Column(BigInteger)
    largestCriticalStrike = Column(Integer)
    totalTimeCrowdControlDealt = Column(Integer)
    timeCCingOthers = Column(BigInteger)

    damageDealtToObjectives = Column(BigInteger)
    damageDealtToTurrets = Column(BigInteger)

    totalHeal = Column(BigInteger)
    totalUnitsHealed = Column(Integer)
    totalDamageTaken = Column(BigInteger)
    physicalDamageTaken = Column(BigInteger)
    magicalDamageTaken = Column(BigInteger)
    trueDamageTaken = Column(BigInteger)
    longestTimeSpentLiving = Column(Integer)
    damageSelfMitigated = Column(BigInteger)

    wardsPlaced = Column(Integer)
    wardsKilled = Column(Integer)
    sightWardsBoughtInGame = Column(Integer)
    visionWardsBoughtInGame = Column(Integer)
    visionScore = Column(Integer)

    goldEarned = Column(Integer)
    goldSpent = Column(Integer)
    totalMinionsKilled = Column(Integer)
    neutralMinionsKilled = Column(Integer)
    neutralMinionsKilledTeamJungle = Column(Integer)
    neutralMinionsKilledEnemyJungle = Column(Integer)

    totalPlayerScore = Column(Integer)
    altarsCaptured = Column(Integer)
    nodeNeutralizeAssist = Column(Integer)
    nodeCaptureAssist = Column(Integer)
    objectivePlayerScore = Column(Integer)
    combatPlayerScore = Column(Integer)
    altarsNeutralized = Column(Integer)
    nodeCapture = Column(Integer)
    totalScoreRank = Column(Integer)

    def __init__(self, dict:dict):
        self.participantId = dict.get("participantId", 0)
        self.win = dict.get("win", False)
        self.item0 = dict.get("item0", 0)
        self.item1 = dict.get("item1", 0)
        self.item2 = dict.get("item2", 0)
        self.item3 = dict.get("item3", 0)
        self.item4 = dict.get("item4", 0)
        self.item5 = dict.get("item5", 0)
        self.item6 = dict.get("item6", 0)
        self.champLevel = dict.get("champLevel", 0)

        self.kills = dict.get("kills", 0)
        self.deaths = dict.get("deaths", 0)
        self.assists = dict.get("assists", 0)
        self.largestMultikill = dict.get("largestMultikill", 0)
        self.killingSprees = dict.get("killingSprees", 0)
        self.largestKillingSpree = dict.get("largestKillingSpree", 0)
        self.doubleKills = dict.get("doubleKills", 0)
        self.tripleKills = dict.get("tripleKills", 0)
        self.quadraKills = dict.get("quadraKills", 0)
        self.pentaKills = dict.get("pentaKills", 0)
        self.unrealKills = dict.get("unrealKills", 0)

        self.turretKills = dict.get("turretKills",0)
        self.inhibitorKills = dict.get("inhibitorKills",0)

        self.firstBloodKill = dict.get("firstBloodKill", False)
        self.firstBloodAssist = dict.get("firstBloodAssist", False)
        self.firstTowerKill = dict.get("firstTowerKill", False)
        self.firstTowerAssist = dict.get("firstTowerAssist", False)
        self.firstInhibitorKill = dict.get("firstInhibitorKill", False)
        self.firstInhibitorAssist = dict.get("firstInhibitorAssist", False)

        self.totalDamageDealtToChampions = dict.get("totalDamageDealtToChampions", 0)
        self.physicalDamageDealtToChampions = dict.get("physicalDamageDealtToChampions",0)
        self.magicDamageDealtToChampions = dict.get("magicDamageDealtToChampions",0)
        self.trueDamageDealtToChampions = dict.get("trueDamageDealtToChampions",0)
        self.totalDamageDealt = dict.get("totalDamageDealt",0)
        self.physicalDamageDealt = dict.get("physicalDamageDealt",0)
        self.magicDamageDealt = dict.get("magicDamageDealt",0)
        self.trueDamageDealt = dict.get("trueDamageDealt",0)
        self.largestCriticalStrike = dict.get("largestCriticalStrike",0)
        self.totalTimeCrowdControlDealt = dict.get("totalTimeCrowdControlDealt",0)
        self.timeCCingOthers = dict.get("timeCCingOthers",0)

        self.damageDealtToObjectives = dict.get("damageDealtToObjectives",0)
        self.damageDealtToTurrets = dict.get("damageDealtToTurrets",0)

        self.totalHeal = dict.get("totalHeal",0)
        self.totalUnitsHealed = dict.get("totalUnitsHealed",0)
        self.totalDamageTaken = dict.get("totalDamageTaken",0)
        self.physicalDamageTaken = dict.get("physicalDamageTaken",0)
        self.magicalDamageTaken = dict.get("magicalDamageTaken",0)
        self.trueDamageTaken = dict.get("trueDamageTaken",0)
        self.longestTimeSpentLiving = dict.get("longestTimeSpentLiving",0)
        self.damageSelfMitigated = dict.get("damageSelfMitigated",0)

        self.wardsPlaced = dict.get("wardsPlaced",0)
        self.wardsKilled = dict.get("wardsKilled",0)
        self.sightWardsBoughtInGame = dict.get("sightWardsBoughtInGame",0)
        self.visionWardsBoughtInGame = dict.get("visionWardsBoughtInGame",0)
        self.visionScore = dict.get("visionScore",0)

        self.goldEarned = dict.get("goldEarned",0)
        self.goldSpent = dict.get("goldSpent",0)
        self.totalMinionsKilled = dict.get("totalMinionsKilled",0)
        self.neutralMinionsKilled = dict.get("neutralMinionsKilled",0)
        self.neutralMinionsKilledTeamJungle = dict.get("neutralMinionsKilledTeamJungle",0)
        self.neutralMinionsKilledEnemyJungle = dict.get("neutralMinionsKilledEnemyJungle",0)

        self.totalPlayerScore = dict.get("totalPlayerScore",0)
        self.altarsCaptured = dict.get("altarsCaptured",0)
        self.nodeNeutralizeAssist = dict.get("nodeNeutralizeAssist",0)
        self.nodeCaptureAssist = dict.get("nodeCaptureAssist",0)
        self.objectivePlayerScore = dict.get("objectivePlayerScore",0)
        self.combatPlayerScore = dict.get("combatPlayerScore",0)
        self.altarsNeutralized = dict.get("altarsNeutralized",0)
        self.nodeCapture = dict.get("nodeCapture",0)
        self.totalScoreRank = dict.get("totalScoreRank",0)
        
class Mastery(Base):
    __tablename__ = "Masteries"

    _id = Column(BigInteger, primary_key=True)
    _participant_id = Column(BigInteger, ForeignKey('Participants._id', ondelete='CASCADE'))
    masteryId = Column(Integer)
    rank = Column(Integer)

    def __init__(self, dict:dict):
        self.masteryId = dict.get("masteryId", 0)
        self.rank = dict.get("rank", 0)

class Rune(Base):
    __tablename__ = "Runes"

    _id = Column(BigInteger, primary_key=True)
    _participant_id = Column(BigInteger, ForeignKey('Participants._id', ondelete='CASCADE'))
    runeId = Column(Integer)
    rank = Column(Integer)

    def __init__(self, dict:dict):
        self.runeId = dict.get("runeId", 0)
        self.rank = dict.get("rank", 0)

class Timeline(Base):
    __tablename__ = "Timelines"

    _id = Column(BigInteger, primary_key=True)
    _participant_id = Column(BigInteger, ForeignKey('Participants._id', ondelete='CASCADE'))
    participantId = Column(Integer)
    role = Column(String(20))
    lane = Column(String(20))
    csDiffPerMinDeltas = orm.relationship('TimelineDelta', uselist=False, cascade='all, delete-orphan', passive_deletes=True,
                                          primaryjoin="and_(Timeline._id==TimelineDelta._timeline_id, TimelineDelta._type=='csDiffPerMinDeltas')")
    goldPerMinDeltas = orm.relationship('TimelineDelta', uselist=False, cascade='all, delete-orphan', passive_deletes=True,
                                          primaryjoin="and_(Timeline._id==TimelineDelta._timeline_id, TimelineDelta._type=='goldPerMinDeltas')")
    xpDiffPerMinDeltas = orm.relationship('TimelineDelta', uselist=False, cascade='all, delete-orphan', passive_deletes=True,
                                          primaryjoin="and_(Timeline._id==TimelineDelta._timeline_id, TimelineDelta._type=='xpDiffPerMinDeltas')")
    creepsPerMinDeltas = orm.relationship('TimelineDelta', uselist=False, cascade='all, delete-orphan', passive_deletes=True,
                                          primaryjoin="and_(Timeline._id==TimelineDelta._timeline_id, TimelineDelta._type=='creepsPerMinDeltas')")
    xpPerMinDeltas = orm.relationship('TimelineDelta', uselist=False, cascade='all, delete-orphan', passive_deletes=True,
                                          primaryjoin="and_(Timeline._id==TimelineDelta._timeline_id, TimelineDelta._type=='xpPerMinDeltas')")
    damageTakenDiffPerMinDeltas = orm.relationship('TimelineDelta', uselist=False, cascade='all, delete-orphan', passive_deletes=True,
                                          primaryjoin="and_(Timeline._id==TimelineDelta._timeline_id, TimelineDelta._type=='damageTakenDiffPerMinDeltas')")
    damageTakenPerMinDeltas = orm.relationship('TimelineDelta', uselist=False, cascade='all, delete-orphan', passive_deletes=True,
                                          primaryjoin="and_(Timeline._id==TimelineDelta._timeline_id, TimelineDelta._type=='damageTakenPerMinDeltas')")

    def __init__(self, dict:dict):
        self.participantId = dict.get("participantId", 0)
        self.role = dict.get("role", "")
        self.lane = dict.get("lane", "")
        val = dict.get("csDiffPerMinDeltas", None)
        self.csDiffPerMinDeltas = TimelineDelta(val, "csDiffPerMinDeltas") if not isinstance(val, TimelineDelta) else val
        val = dict.get("goldPerMinDeltas", None)
        self.goldPerMinDeltas = TimelineDelta(val, "goldPerMinDeltas") if not isinstance(val, TimelineDelta) else val
        val = dict.get("xpDiffPerMinDeltas", None)
        self.xpDiffPerMinDeltas = TimelineDelta(val, "xpDiffPerMinDeltas") if not isinstance(val, TimelineDelta) else val
        val = dict.get("creepsPerMinDeltas", None)
        self.creepsPerMinDeltas = TimelineDelta(val, "creepsPerMinDeltas") if not isinstance(val, TimelineDelta) else val
        val = dict.get("xpPerMinDeltas", None)
        self.xpPerMinDeltas = TimelineDelta(val, "xpPerMinDeltas") if not isinstance(val, TimelineDelta) else val
        val = dict.get("damageTakenDiffPerMinDeltas", None)
        self.damageTakenDiffPerMinDeltas = TimelineDelta(val, "damageTakenDiffPerMinDeltas") if not isinstance(val, TimelineDelta) else val
        val = dict.get("damageTakenPerMinDeltas", None)
        self.damageTakenPerMinDeltas = TimelineDelta(val, "damageTakenPerMinDeltas") if not isinstance(val, TimelineDelta) else val


class TimelineDelta(Base):
    __tablename__ = "TimelineDeltas"

    _id = Column(BigInteger, primary_key=True)
    _timeline_id = Column(BigInteger, ForeignKey('Timelines._id', ondelete='CASCADE'))
    _type = Column(String(30))
    zeroToTen = Column(Float)
    tenToTwenty = Column(Float)
    twentyToThirty = Column(Float)

    def __init__(self, dict:dict, _type:str = None):
        self.zeroToTen = dict.get("0-10", 0.0)
        self.tenToTwenty = dict.get("10-20", 0.0)
        self.twentyToThirty = dict.get("20-30", 0.0)
        self._type = _type



class MatchTimeline(Base):
    __tablename__ = "MatchTimelines"

    _id = Column(BigInteger, primary_key=True)
    gameId = Column(BigInteger, ForeignKey('MatchDetails.gameId', ondelete='CASCADE'))
    frameInterval = Column(BigInteger)
    frames = orm.relationship('MatchFrame', cascade='all, delete-orphan', passive_deletes=True)

    def __init__(self, dict:dict, gameId:int):
        self.gameId = gameId
        self.frameInterval = dict.get("frameInterval",0)
        self.frames = [MatchFrame(f) if not isinstance(f, MatchFrame) else f for f in dict.get("frames", [])]


class MatchFrame(Base):
    __tablename__ = "MatchFrames"

    _id = Column(BigInteger, primary_key=True)
    _timeline_id = Column(BigInteger, ForeignKey('MatchTimelines._id', ondelete='CASCADE'))
    timestamp = Column(BigInteger)
    participantFrames = orm.relationship('ParticipantFrame', cascade='all, delete-orphan', passive_deletes=True)
    events = orm.relationship('Event', cascade='all, delete-orphan', passive_deletes=True)

    def __init__(self, dict:dict):
        self.timestamp = dict.get("timestamp", 0)
        self.participantFrames = [ParticipantFrame(v) if not isinstance(v, ParticipantFrame) else v for k,v in dict.get('participantFrames',{}).items()]
        self.events = [Event(e) if not isinstance(e, Event) else e for e in dict.get("events", [])]

class ParticipantFrame(Base):
    __tablename__ = "ParticipantFrames"

    _id = Column(BigInteger, primary_key=True)
    _frame_id = Column(BigInteger, ForeignKey('MatchFrames._id', ondelete='CASCADE') )
    participantId = Column(Integer)
    level = Column(Integer)
    currentGold = Column(Integer)
    x = Column(Integer)
    y = Column(Integer)
    minionsKilled = Column(Integer)
    jungleMinionsKilled = Column(Integer)
    totalGold = Column(Integer)
    dominionScore = Column(Integer)
    teamScore = Column(Integer)

    def __init__(self, dic:dict):
        self.participantId = dic.get("participantId", 0)
        self.level = dic.get("level", 0)
        self.currentGold = dic.get("currentGold", 0)
        val = dic.get("position", None)
        self.x = val.get("x", 0) if isinstance(val, dict) else val
        self.y = val.get("y", 0) if isinstance(val, dict) else val
        self.minionsKilled = dic.get("minionsKilled", 0)
        self.jungleMinionsKilled = dic.get("jungleMinionsKilled",0)
        self.totalGold = dic.get("totalGold",0)
        self.dominionScore = dic.get("dominionScore", 0)
        self.teamScore = dic.get("teamScore", 0)



class Event(Base):
    __tablename__ = "Events"

    _id = Column(BigInteger, primary_key=True)
    _frame_id = Column(BigInteger, ForeignKey('MatchFrames._id', ondelete='CASCADE'))
    timestamp = Column(BigInteger)
    eventType = Column(String(30))
    _type = Column(String(30))
    teamId = Column(Integer)
    participantId = Column(Integer)
    'kills'
    killerId = Column(Integer)
    assistant_1 = Column(Integer)
    assistant_2 = Column(Integer)
    assistant_3 = Column(Integer)
    assistant_4 = Column(Integer)
    vicitimId = Column(Integer)
    'ward placed'
    wardType = Column(String(30))
    towerType = Column(String(30))
    'objecives'
    monsterType = Column(String(30))
    monsterSubType = Column(String(30))
    buildingType = Column(String(30))
    laneType = Column(String(30))
    'levelup'
    levelUpType = Column(String(30))
    skillSlot = Column(Integer)
    'item purchased/destroyed/undo'
    afterId = Column(Integer)
    beforeId = Column(Integer)
    itemId = Column(String(30))
    creatorId = Column(Integer)
    'ascention'
    ascendedType = Column(String(30))
    pointCaptured = Column(String(30))
    x = Column(Integer)
    y = Column(Integer)

    def __init__(self, dic:dict):
        self.timestamp = dic.get("timestamp")
        self.eventType = dic.get("eventType")
        self._type = dic.get("type")
        self.teamId = dic.get("teamId")
        self.participantId = dic.get("participantId")
        'kills'
        self.killerId = dic.get("killerId")
        val = dic.get("assistingParticipantIds", [])
        self.assistant_1 = val[0] if len(val) > 0 else None
        self.assistant_2 = val[1] if len(val) > 1 else None
        self.assistant_3 = val[2] if len(val) > 2 else None
        self.assistant_4 = val[3] if len(val) > 3 else None
        self.vicitimId = dic.get("vicitimId")
        'ward placed'
        self.wardType = dic.get("wardType")
        self.towerType = dic.get("towerType")
        'objecives'
        self.monsterType = dic.get("monsterType")
        self.monsterSubType = dic.get("monsterSubType")
        self.buildingType = dic.get("buildingType")
        self.laneType = dic.get("laneType")
        'levelup'
        self.levelUpType = dic.get("levelUpType")
        self.skillSlot = dic.get("skillSlot")
        'item purchased/destroyed/undo'
        self.afterId = dic.get("afterId")
        self.beforeId = dic.get("beforeId")
        self.itemId = dic.get("itemId")
        self.creatorId = dic.get("creatorId")
        'ascention'
        self.ascendedType = dic.get("ascendedType")
        self.pointCaptured = dic.get("pointCaptured")
        val = dic.get("position", None)
        self.x = val.get("x", 0) if isinstance(val, dict) else val
        self.y = dic.get("y",0) if isinstance(val, dict) else val