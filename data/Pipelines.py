def playerstats(platformId, patch):
    return [
        {"$match": {
            "platformId": platformId,
            "patch": patch
        }
        },
        {"$unwind": "$participants"},
        {"$unwind": "$participantIdentities"},
        {"$unwind": "$teams"},
        {"$redact":
             {"$cond": {"if": {"$eq": ["$participants.participantId", "$participantIdentities.participantId"]},
                        "then": "$$DESCEND",
                        "else": "$$PRUNE"
                        }}
         },
        {"$redact":
             {"$cond": {"if": {"$eq": ["$participants.teamId", "$teams.teamId"]},
                        "then": "$$DESCEND",
                        "else": "$$PRUNE"
                        }}
         },
        {"$project": {
            "_id": {"$concat": [{"$substr": ["$gameId", 0, -1]}, "$platformId",
                                {"$substr": ["$participants.participantId", 0, -1]}]},
            "gameId": "$gameId",
            "platformId": "$platformId",
            "gameVersion": "$gameVersion",
            "patch": "$patch",
            "gameCreation": "$gameCreation",
            "gameDuration": "$gameDuration",
            "participantId": "$participants.participantId",
            "teamId": "$participants.teamId",
            "summonerId": "$participantIdentities.player.summonerId",
            "championId": "$participants.championId",
            "lane": "$participants.timeline.lane",
            "role": "$participants.timeline.role",
            "win": "$participants.stats.win",
            "item0": "$participants.stats.item0",
            "item1": "$participants.stats.item1",
            "item2": "$participants.stats.item2",
            "item3": "$participants.stats.item3",
            "item4": "$participants.stats.item4",
            "item5": "$participants.stats.item5",
            "item6": "$participants.stats.item6",
            "kills": "$participants.stats.kills",
            "deaths": "$participants.stats.deaths",
            "assists": "$participants.stats.assists",
            "largestMultikill": "$participants.stats.largestMultiKill",
            "killingSprees": "$participants.stats.killingSprees",
            "doubleKills": "$participants.stats.doubleKills",
            "tripleKills": "$participants.stats.tripleKills",
            "quadraKills": "$participants.stats.quadraKills",
            "pentaKills": "$participants.stats.pentaKills",
            "unrealKills": "$participants.stats.unrealKills",
            "turretKills": "$participants.stats.turretKills",
            "inhibitorKills": "$participants.stats.inhibitorKills",
            "firstBloodKill": "$participants.stats.firstBloodKill",
            "firstBloodAssist": "$participants.stats.firstBloodAssist",
            "firstTowerKill": "$participants.stats.firstTowerKill",
            "firstTowerAssist": "$participants.stats.firstTowerAssist",
            "firstInhibitorKill": "$participants.stats.firstInhibitorKill",
            "firstInhibitorAssist": "$participants.stats.firstInhibitorAssist",
            "totalDamageDealtToChampions": "$participants.stats.totalDamageDealtToChampions",
            "physicalDamageDealtToChampions": "$participants.stats.physicalDamageDealtToChampions",
            "magicDamageDealtToChampions": "$participants.stats.magicDamageDealtToChampions",
            "trueDamageDealtToChampions": "$participants.stats.trueDamageDealtToChampions",
            "totalDamageDealt": "$participants.stats.totalDamageDealt",
            "physicalDamageDealt": "$participants.stats.physicalDamageDealt",
            "magicDamageDealt": "$participants.stats.magicDamageDealt",
            "trueDamageDealt": "$participants.stats.trueDamageDealt",
            "largestCriticalStrike": "$participants.stats.largestCriticalStrike",
            "totalTimeCrowdControlDealt": "$participants.stats.totalTimeCrowdControlDealt",
            "timeCCingOthers": "$participants.stats.timeCCingOthers",
            "damageDealtToObjectives": "$participants.stats.damageDealtToObjectives",
            "damageDealtToTurrets": "$participants.stats.damageDealtToTurrets",
            "totalHeal": "$participants.stats.totalHeal",
            "totalUnitsHealed": "$participants.stats.totalUnitsHealed",
            "totalDamageTaken": "$participants.stats.totalDamageTaken",
            "physicalDamageTaken": "$participants.stats.physicalDamageTaken",
            "magicalDamageTaken": "$participants.stats.magicalDamageTaken",
            "trueDamageTaken": "$participants.stats.trueDamageTaken",
            "longestTimeSpentLiving": "$participants.stats.longestTimeSpentLiving",
            "damageSelfMitigated": "$participants.stats.damageSelfMitigated",
            "wardsPlaced": "$participants.stats.wardsPlaced",
            "wardsKilled": "$participants.stats.wardsKilled",
            "sightWardsBoughtInGame": "$participants.stats.sightWardsBoughtInGame",
            "visionWardsBoughtInGame": "$participants.stats.visionWardsBoughtInGame",
            "visionScore": "$participants.stats.visionScore",
            "goldEarned": "$participants.stats.goldEarned",
            "goldSpent": "$participants.stats.goldSpent",
            "totalMinionsKilled": "$participants.stats.totalMinionsKilled",
            "neutralMinionsKilled": "$participants.stats.neutralMinionsKilled",
            "neutralMinionsKilledTeamJungle": "$participants.stats.neutralMinionsKilledTeamJungle",
            "neutralMinionsKilledEnemyJungle": "$participants.stats.neutralMinionsKilledEnemyJungle",
            "creepsPerMinDeltas": "$participants.timeline.creepsPerMinDeltas",
            "xpPerMinDeltas": "$participants.timeline.xpPerMinDeltas",
            "goldPerMinDeltas": "$participants.timeline.goldPerMinDeltas",
            "csDiffPerMinDeltas": "$participants.timeline.csDiffPerMinDeltas",
            "xpDiffPerMinDeltas": "$participants.timeline.xpDiffPerMinDeltas",
            "damageTakenPerMinDeltas": "$participants.timeline.damageTakenPerMinDeltas",
            "damageTakenDiffPerMinDeltas": "$participants.timeline.damageTakenDiffPerMinDeltas",
            "teamFirstBlood": "$teams.firstBlood",
            "teamFirstTower": "$teams.firstTower",
            "teamFirstInhibitor": "$teams.firstInhibitor",
            "teamFirstBaron": "$teams.firstBaron",
            "teamFirstDragon": "$teams.firstDragon",
            "teamFirstRiftHerald": "$teams.firstRiftHerald",
            "teamTowerKills": "$teams.towerKills",
            "teamInhibitorKills": "$teams.inhibitorKills",
            "teamBaronKills": "$teams.baronKills",
            "teamDragonKills": "$teams.dragonKills",
            "teamRiftHeraldKills": "$teams.RiftHeraldKills",
            "bans": "$teams.bans"
        }},
        {"$addFields": {
            "ban1": {
                "$let": {
                    "vars": {
                        "ban": {
                            "$arrayElemAt": ["$bans", 0]
                        }
                    },
                    "in": "$$ban.championId"
                }
            },
            "ban2": {
                "$let": {
                    "vars": {
                        "ban": {
                            "$arrayElemAt": ["$bans", 1]
                        }
                    },
                    "in": "$$ban.championId"
                }
            },
            "ban3": {
                "$let": {
                    "vars": {
                        "ban": {
                            "$arrayElemAt": ["$bans", 2]
                        }
                    },
                    "in": "$$ban.championId"
                }
            },
            "ban4 ": {
                "$let": {
                    "vars": {
                        "ban": {
                            "$arrayElemAt": ["$bans", 3]
                        }
                    },
                    "in": "$$ban.championId"
                }
            },
            "ban5": {
                "$let": {
                    "vars": {
                        "ban": {
                            "$arrayElemAt": ["$bans", 4]
                        }
                    },
                    "in": "$$ban.championId"
                }
            }
        }
        }
    ]

def proplayerstats(patch):
    return [
        {"$match": {
            "patch": patch
        }
        },
        {"$unwind": "$participants"},
        {"$unwind": "$participantIdentities"},
        {"$unwind": "$teams"},
        {"$redact": {
            "$cond": {
                "if": {
                    "$eq": ["$participants.participantId", "$participantIdentities.participantId"]},
                "then": "$$DESCEND",
                "else": "$$PRUNE"
            }
        }
        },
        {"$redact": {
            "$cond": {
                "if": {
                    "$eq": ["$participants.teamId", "$teams.teamId"]},
                "then": "$$DESCEND",
                "else": "$$PRUNE"
            }
        }
        },
        {"$project": {
            "_id": {"$concat": [{"$substr": ["$gameId", 0, -1]}, "$platformId",
                                {"$substr": ["$participants.participantId", 0, -1]}]},
            "gameId": "$gameId",
            "platformId": "$platformId",
            "gameVersion": "$gameVersion",
            "patch": "$patch",
            "gameCreation": "$gameCreation",
            "gameDuration": "$gameDuration",
            "participantId": "$participants.participantId",
            "teamId": "$participants.teamId",
            "summonerId": "$participantIdentities.player.summonerName",
            "championId": "$participants.championId",
            "lane": "$participants.timeline.lane",
            "role": "$participants.timeline.role",
            "win": "$participants.stats.win",
            "item0": "$participants.stats.item0",
            "item1": "$participants.stats.item1",
            "item2": "$participants.stats.item2",
            "item3": "$participants.stats.item3",
            "item4": "$participants.stats.item4",
            "item5": "$participants.stats.item5",
            "item6": "$participants.stats.item6",
            "kills": "$participants.stats.kills",
            "deaths": "$participants.stats.deaths",
            "assists": "$participants.stats.assists",
            "largestMultikill": "$participants.stats.largestMultiKill",
            "killingSprees": "$participants.stats.killingSprees",
            "doubleKills": "$participants.stats.doubleKills",
            "tripleKills": "$participants.stats.tripleKills",
            "quadraKills": "$participants.stats.quadraKills",
            "pentaKills": "$participants.stats.pentaKills",
            "unrealKills": "$participants.stats.unrealKills",
            "turretKills": "$participants.stats.turretKills",
            "inhibitorKills": "$participants.stats.inhibitorKills",
            "firstBloodKill": "$participants.stats.firstBloodKill",
            "firstBloodAssist": "$participants.stats.firstBloodAssist",
            "firstTowerKill": "$participants.stats.firstTowerKill",
            "firstTowerAssist": "$participants.stats.firstTowerAssist",
            "firstInhibitorKill": "$participants.stats.firstInhibitorKill",
            "firstInhibitorAssist": "$participants.stats.firstInhibitorAssist",
            "totalDamageDealtToChampions": "$participants.stats.totalDamageDealtToChampions",
            "physicalDamageDealtToChampions": "$participants.stats.physicalDamageDealtToChampions",
            "magicDamageDealtToChampions": "$participants.stats.magicDamageDealtToChampions",
            "trueDamageDealtToChampions": "$participants.stats.trueDamageDealtToChampions",
            "totalDamageDealt": "$participants.stats.totalDamageDealt",
            "physicalDamageDealt": "$participants.stats.physicalDamageDealt",
            "magicDamageDealt": "$participants.stats.magicDamageDealt",
            "trueDamageDealt": "$participants.stats.trueDamageDealt",
            "largestCriticalStrike": "$participants.stats.largestCriticalStrike",
            "totalTimeCrowdControlDealt": "$participants.stats.totalTimeCrowdControlDealt",
            "timeCCingOthers": "$participants.stats.timeCCingOthers",
            "damageDealtToObjectives": "$participants.stats.damageDealtToObjectives",
            "damageDealtToTurrets": "$participants.stats.damageDealtToTurrets",
            "totalHeal": "$participants.stats.totalHeal",
            "totalUnitsHealed": "$participants.stats.totalUnitsHealed",
            "totalDamageTaken": "$participants.stats.totalDamageTaken",
            "physicalDamageTaken": "$participants.stats.physicalDamageTaken",
            "magicalDamageTaken": "$participants.stats.magicalDamageTaken",
            "trueDamageTaken": "$participants.stats.trueDamageTaken",
            "longestTimeSpentLiving": "$participants.stats.longestTimeSpentLiving",
            "damageSelfMitigated": "$participants.stats.damageSelfMitigated",
            "wardsPlaced": "$participants.stats.wardsPlaced",
            "wardsKilled": "$participants.stats.wardsKilled",
            "sightWardsBoughtInGame": "$participants.stats.sightWardsBoughtInGame",
            "visionWardsBoughtInGame": "$participants.stats.visionWardsBoughtInGame",
            "visionScore": "$participants.stats.visionScore",
            "goldEarned": "$participants.stats.goldEarned",
            "goldSpent": "$participants.stats.goldSpent",
            "totalMinionsKilled": "$participants.stats.totalMinionsKilled",
            "neutralMinionsKilled": "$participants.stats.neutralMinionsKilled",
            "neutralMinionsKilledTeamJungle": "$participants.stats.neutralMinionsKilledTeamJungle",
            "neutralMinionsKilledEnemyJungle": "$participants.stats.neutralMinionsKilledEnemyJungle",
            "creepsPerMinDeltas": "$participants.timeline.creepsPerMinDeltas",
            "xpPerMinDeltas": "$participants.timeline.xpPerMinDeltas",
            "goldPerMinDeltas": "$participants.timeline.goldPerMinDeltas",
            "csDiffPerMinDeltas": "$participants.timeline.csDiffPerMinDeltas",
            "xpDiffPerMinDeltas": "$participants.timeline.xpDiffPerMinDeltas",
            "damageTakenPerMinDeltas": "$participants.timeline.damageTakenPerMinDeltas",
            "damageTakenDiffPerMinDeltas": "$participants.timeline.damageTakenDiffPerMinDeltas",
            "teamFirstBlood": "$teams.firstBlood",
            "teamFirstTower": "$teams.firstTower",
            "teamFirstInhibitor": "$teams.firstInhibitor",
            "teamFirstBaron": "$teams.firstBaron",
            "teamFirstDragon": "$teams.firstDragon",
            "teamFirstRiftHerald": "$teams.firstRiftHerald",
            "teamTowerKills": "$teams.towerKills",
            "teamInhibitorKills": "$teams.inhibitorKills",
            "teamBaronKills": "$teams.baronKills",
            "teamDragonKills": "$teams.dragonKills",
            "teamRiftHeraldKills": "$teams.RiftHeraldKills",
            "bans": "$teams.bans"
        }},
        {"$addFields": {
            "ban1": {
                "$let": {
                    "vars": {
                        "ban": {
                            "$arrayElemAt": ["$bans", 0]
                        }
                    },
                    "in": "$$ban.championId"
                }
            },
            "ban2": {
                "$let": {
                    "vars": {
                        "ban": {
                            "$arrayElemAt": ["$bans", 1]
                        }
                    },
                    "in": "$$ban.championId"
                }
            },
            "ban3": {
                "$let": {
                    "vars": {
                        "ban": {
                            "$arrayElemAt": ["$bans", 2]
                        }
                    },
                    "in": "$$ban.championId"
                }
            },
            "ban4" : {
                "$let": {
                    "vars": {
                        "ban": {
                            "$arrayElemAt": ["$bans", 3]
                        }
                    },
                    "in": "$$ban.championId"
                }
            },
            "ban5": {
                "$let": {
                    "vars": {
                        "ban": {
                            "$arrayElemAt": ["$bans", 4]
                        }
                    },
                    "in": "$$ban.championId"
                }
            }
        }
        }
    ]

def itemsets_adc():
    return \
    [{"$match": {
        "gameDuration": {"$gte": 900}  # game Duration >= 900 seconds = 15 Minutes
        }
    },
    {"$unwind": "$participants"},
    {"$match": {
        "participants.timeline.lane": "BOTTOM",
        "participants.timeline.role" : "DUO_CARRY"
    }
    },
    {"$unwind": "$timeline.frames"},
    {"$unwind": "$timeline.frames.events"},
    {"$match": {
        "timeline.frames.events.type": "ITEM_PURCHASED"
    }
    },
    {"$redact": {
        "$cond": {
            "if": {
                "$eq": ["$participants.participantId", "$timeline.frames.events.participantId"]
            },
            "then": "$$KEEP",
            "else": "$$PRUNE"
        }
    }
    },
    {"$group": {
        "_id": {
            "gameId": "$gameId",
            "platformId": "$platformId",
            "gameVersion": "$gameVersion",
            "championId": "$participants.championId"
        },
        "items": {
            "$push": "$timeline.frames.events.itemId"
        }
    }},
    {"$project": {
        "_id": 0,
        "gameId": "$_id.gameId",
        "platformId": "$_id.platformId",
        "gameVersion": "$_id.gameVersion",
        "championId": "$_id.championId",
        "items": "$items"
    }
    }]

def itemsets(championId, export:False):
    pipline = \
        [{"$match": {
            "gameDuration": {"$gte": 900 } #game Duration >= 900 seconds = 15 Minutes
            }
        },
        {"$unwind": "$participants"},
        {"$match": {
            "participants.championId": championId
            }
        },
        {"$unwind": "$timeline.frames"},
        {"$unwind": "$timeline.frames.events"},
        {"$match": {
            "timeline.frames.events.type": "ITEM_PURCHASED"
        }
        },
        {"$redact": {
            "$cond": {
                "if": {
                    "$eq": ["$participants.participantId", "$timeline.frames.events.participantId"]
                    },
                "then": "$$KEEP",
                "else": "$$PRUNE"
                }
            }
        },
        {"$group": {
            "_id": {
                "gameId": "$gameId",
                "platformId": "$platformId",
                "gameVersion" :"$gameVersion",
                "championId": "$participants.championId"
            },
            "items": {
                "$push": "$timeline.frames.events.itemId"
            }
        }},
        {"$project": {
            "_id": 0,
            "gameId": "$_id.gameId",
            "platformId": "$_id.platformId",
            "gameVersion": "$_id.gameVersion",
            "championId": "$_id.championId",
            "items": "$items"
            }
        }]
    if export: pipline.append({"$out": "aggregationTest"})
    return pipline

