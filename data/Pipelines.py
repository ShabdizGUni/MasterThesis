def playerstats(platformId, patch, testset:int=None):
    pipeline = [
        {"$match": {
            "platformId": platformId,
            "patch" : patch
            }
        },
        {"$unwind": "$participants"},
        {"$unwind": "$participantIdentities"},
        {"$addFields": {
            "bans" : {
                "$concatArrays": [
                    {
                        "$let": {
                            "vars": {"t1": {"$arrayElemAt": ["$teams", 0] } },
                            "in": "$$t1.bans"
                            }
                    },
                    {
                        "$let": {
                            "vars": {"t2": {"$arrayElemAt": ["$teams", 1] } },
                            "in": "$$t2.bans"

                        }
                    }
                ]
            }
        }
        },
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
            "tier": "$participants.highestAchievedSeasonTier",
            "bans": "$bans",
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
            "doublekills": "$participants.stats.doublekills",
            "triplekills": "$participants.stats.triplekills",
            "quadrakills": "$participants.stats.quadrakills",
            "pentakills": "$participants.stats.pentakills",
            "unrealkills": "$participants.stats.unrealkills",
            "turretkills": "$participants.stats.turretkills",
            "inhibitorkills": "$participants.stats.inhibitorkills",
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
            "teamTowerkills": "$teams.towerkills",
            "teamInhibitorkills": "$teams.inhibitorkills",
            "teamBaronkills": "$teams.baronkills",
            "teamDragonkills": "$teams.dragonkills",
            "teamRiftHeraldkills": "$teams.RiftHeraldkills"
        }}
    ]
    if testset: pipeline.insert(0,{"$limit": testset})
    return pipeline

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
            "doublekills": "$participants.stats.doublekills",
            "triplekills": "$participants.stats.triplekills",
            "quadrakills": "$participants.stats.quadrakills",
            "pentakills": "$participants.stats.pentakills",
            "unrealkills": "$participants.stats.unrealkills",
            "turretkills": "$participants.stats.turretkills",
            "inhibitorkills": "$participants.stats.inhibitorkills",
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
            "teamTowerkills": "$teams.towerkills",
            "teamInhibitorkills": "$teams.inhibitorkills",
            "teamBaronkills": "$teams.baronkills",
            "teamDragonkills": "$teams.dragonkills",
            "teamRiftHeraldkills": "$teams.RiftHeraldkills",
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
    if test: pipeline[0] = {"$limit": 2000}
    return pipeline

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

def get_summoners(region:str, tiers:list):
    return \
    [
    {"$match": {
        "platformId": region
        }
    },
    {"$unwind": "$participants"},
    {"$unwind": "$participantIdentities"},
    {"$redact":
        {"$cond":{ "if": { "$eq": [ "$participants.participantId", "$participantIdentities.participantId" ] },
              "then": "$$DESCEND",
              "else": "$$PRUNE"
                  }}
    },
    {"$match": {
        "participants.highestAchievedSeasonTier": {
            "$in": tiers
            },
        "participantIdentities.player.currentPlatformId" : region
        }
    },
    {"$group": {
        "_id" : 0,
        "summonerIds": {"$addToSet": "$participantIdentities.player.currentAccountId"}
        }
    }
    ]

def get_events(championId:int, sample_size:int):
    return \
    [
    {"$match": {
        "participants.championId": championId
        }
    },
    {"$limit": sample_size},
    {"$unwind": "$participants"},
    {"$match": {
        "participants.championId": championId
        }
    },
    {"$unwind": "$timeline.frames"},
    {"$unwind": "$timeline.frames.events"},
    {"$match": {
        "$or": [
            {"timeline.frames.events.type": "ITEM_PURCHASED"},
            {"timeline.frames.events.type": "ITEM_UNDO"},
            {"timeline.frames.events.type": "ITEM_DESTROYED"},
            {"timeline.frames.events.type": "ITEM_SOLD"},
            {"timeline.frames.events.type": "CHAMPION_KILL"},
            {"timeline.frames.events.type": "SKILL_LEVEL_UP"},
            {"timeline.frames.events.type": "BUILDING_KILL"}]
    }
    },
    {"$redact":
    {"$cond": {
    "if": {
            "$or": [
                {"$eq": ["$participants.participantId", "$timeline.frames.events.participantId"]},
                {"$eq": ["$participants.participantId", "$timeline.frames.events.killerId"]},
                {"$eq": ["$participants.participantId", "$timeline.frames.events.victimId"]},
                {"$in": [
                    "$participants.participantId",
                    {"$ifNull": ["$timeline.frames.events.assistingParticipantIds", {"$literal": []}]}
                ]
                }
            ]
        },
        "then": "$$KEEP",
    "else": "$$PRUNE"
    }}
    },
    {"$project": {
        "_id": 0,
        "gameId": "$gameId",
        "platformId": "$platformId",
        "patch": "$patch",
        "participantId": "$participants.participantId",
        "frame_number": {" $trunc": { "$divide": ["$timeline.frames.timestamp", 60000]}},
    "killerId": "$timeline.frames.events.killerId",
    "victimId": "$timeline.frames.events.victimId",
    "assistingParticipantIds": "$timeline.frames.events.assistingParticipantIds",
    "championId": "$participants.championId",
    "type": "$timeline.frames.events.type",
    "itemId": "$timeline.frames.events.itemId",
    "beforeId": "$timeline.frames.events.beforeId",
    "afterid": "$timeline.frames.events.afterId",
    "skillslot": "$timeline.frames.events.skillslot",
    "timestamp": "$timeline.frames.events.timestamp"
    }},
    { "$sort": {
        "gameId": 1,
        "frame_number": 1,
        "timestamp": 1
    }
    },
    { "$out": "jhin_events_test"}
    ]

def get_frames(championId:int, sample_size:int):
    return \
    [
    {"$match": {
        "participants.championId": 110
        }
    },
    {"$limit": 20000},
    {"$unwind": "$participants"},
    {"$unwind": "$timeline.frames"},
    {"$addFields": {
        "timeline.frames.participantFrames.frame_number": { "$trunc": { "$divide": ["$timeline.frames.timestamp", 60000]}}
        }
    },
    {"$addFields": {
        "frames": {
            "$let": {
                "vars": {
                    "frame": {
                        "$arrayElemAt": [
                            {"$filter": {
                                "input": { "$objectToArray": "$timeline.frames.participantFrames"},
                                "as": "frame",
                                "cond": {
                                    "$eq": ["$$frame.v.participantId", "$participants.participantId"]
                                }
                            }
                            },
                            0]
                    }
                },
                "in": "$$frame.v"
            }
        }
    }
    },
    { "$project": {
        "_id": 0,
        "gameId": "$gameId",
        "platformId": "$platformId",
        "participantId": "$participants.participantId",
        "currentGold": "$frames.currentGold",
        "totalGold": "$frames.totalGold",
        "level": "$frames.level",
        "xp": "$frames.xp",
        "minionsKilled": "$frames.minionsKilled",
        "jungleMinionsKilled": "$frames.jungleMinionsKilled",
        "frameNo": { "$add": ["$timeline.frames.participantFrames.frame_number", 1]}
        }
    }
    ]

def append_specialisation():
    return \
    [
    {"$sort": {"platformId": 1, "gameId": 1, "timestamp": 1}},
    {"$group": {"_id": {"gameId": "$gameId", "platformId": "$platformId"}, "data": {"$push": "$$ROOT"}}},
    {"$addFields": {"data": {
    "$reduce": {
                 "input": "$data",
                 "initialValue": {
                     "data": [],
                     "kills": 0, "deaths": 0, "assists": 0,
                     "q_rank": 0, "w_rank": 0, "e_rank": 0, "r_rank": 0,
                     "dorans_blade": 0, "dorans_ring": 0, "dorans_shield": 0,
                     "boots": 0, "bers_greaves": 0, "mobi_boots": 0, "boots_swiftn": 0, "ninja_tabi": 0, "boots_luci": 0,
                     "merc_treads": 0,
                     "long_sword": 0, "dagger": 0, "brawl_gloves": 0, "cloak_oa": 0,
                     "ruby_crystal": 0, "saph_crystal": 0,
                     "blasting_wand": 0, "pickaxe": 0, "bf_sword": 0,
                     "serrated_dirk": 0, "cf_warhammer": 0, "vamp_scepter": 0, "cutlass": 0,
                     "recurve_bow": 0, "zeal": 0, "kircheis_shard": 0, "sheen": 0, "phage": 0,
                     "p_dancer": 0, "s_shivv": 0, "runaans_h": 0, "rapid_fc": 0, "tri_force": 0,
                     "ie": 0, "er": 0, "gb": 0, "duskblade": 0, "eon": 0, "muramana": 0, "ibg": 0,
                     "wits_end": 0, "g_rageblade": 0, "guard_angel": 0, "last_whisper": 0, "lord_dom": 0, "mortal_rem": 0
                 },
             "in": {
    "kills": {
        "$sum": [
            "$$value.kills",
            {"$cond": [
                {"$and": [
                    {"$eq": ["$$this.killerId", "$$this.participantId"]},
                    {"$eq": ["$$this.type", "CHAMPION_KILL"]}
                ]},
                1,
                0
            ]}
        ]
    },
    "deaths": {
    "$sum": [
        "$$value.deaths",
        {"$cond": [
        {"$and": [
        {"$eq": ["$$this.victimId", "$$this.participantId"]},
    {"$eq": ["$$this.type", "CHAMPION_KILL"]}
    ]},
    1,
    0
    ]}
    ]
    },
    "assists": {
    "$sum": [
        "$$value.assists",
        {"$cond": [
        {"$and": [
        {"$eq": ["$$this.type", "CHAMPION_KILL"]},
    {"$in": [
        "$$this.participantId",
        {"$ifNull": ["$$this.assistingParticipantIds", {"$literal": []}]}
    ]
    }
    ]},
    1,
    0
    ]}
    ]
    },
    "q_rank": {
    "$sum": [
        "$$value.q_rank",
        {"$cond": [
        {"$and": [
        {"$eq": ["$$this.type", "SKILL_LEVEL_UP"]},
    {"$eq": ["$$this.skillslot", 1]},
    ]},
    1,
    0
    ]}
    ]
    },
    "w_rank": {
    "$sum": [
        "$$value.w_rank",
        {"$cond": [
        {"$and": [
        {"$eq": ["$$this.type", "SKILL_LEVEL_UP"]},
    {"$eq": ["$$this.skillslot", 2]},
    ]},
    1,
    0
    ]}
    ]
    },
    "e_rank": {
    "$sum": [
        "$$value.e_rank",
        {"$cond": [
        {"$and": [
        {"$eq": ["$$this.type", "SKILL_LEVEL_UP"]},
    {"$eq": ["$$this.skillslot", 3]},
    ]},
    1,
    0
    ]}
    ]
    },
    "r_rank": {
    "$sum": [
        "$$value.r_rank",
        {"$cond": [
        {"$and": [{"$eq": ["$$this.type", "SKILL_LEVEL_UP"]}, {"$eq": ["$$this.skillslot", 4]}, ]}, 1, 0]}
    ]
    },
    "dorans_blade": {
    "$sum": [
        "$$value.dorans_blade",
        { "$switch": {
        "branches": [
            {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_PURCHASED"]}, { "$eq": ["$$this.itemId", 1055]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.afterId", 1055]}]}
    ]},
    "then": 1
    },
    {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_DESTROYED"]}, { "$eq": ["$$this.itemId", 1055]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.beforeId", 1055]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_SOLD"]}, { "$eq": ["$$this.itemId", 1055]}]}
    ]},
    "then": -1
    }],
    "default": 0
    }}
    ]
    },
    "dorans_ring": {
    "$sum": [
        "$$value.dorans_ring",
        { "$switch": {
        "branches": [
            {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_PURCHASED"]}, { "$eq": ["$$this.itemId", 1056]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.afterId", 1056]}]}
    ]},
    "then": 1
    },
    {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_DESTROYED"]}, { "$eq": ["$$this.itemId", 1056]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.beforeId", 1056]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_SOLD"]}, { "$eq": ["$$this.itemId", 1056]}]}
    ]},
    "then": -1
    }],
    "default": 0
    }}
    ]
    },
    "dorans_shield": {
    "$sum": [
        "$$value.dorans_shield",
        { "$switch": {
        "branches": [
            {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_PURCHASED"]}, { "$eq": ["$$this.itemId", 1054]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.afterId", 1054]}]}
    ]},
    "then": 1
    },
    {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_DESTROYED"]}, { "$eq": ["$$this.itemId", 1054]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.beforeId", 1054]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_SOLD"]}, { "$eq": ["$$this.itemId", 1054]}]}
    ]},
    "then": -1
    }],
    "default": 0
    }}
    ]
    },
    "boots": {
    "$sum": [
        "$$value.boots",
        { "$switch": {
        "branches": [
            {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_PURCHASED"]}, { "$eq": ["$$this.itemId", 1001]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.afterId", 1001]}]}
    ]},
    "then": 1
    },
    {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_DESTROYED"]}, { "$eq": ["$$this.itemId", 1001]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.beforeId", 1001]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_SOLD"]}, { "$eq": ["$$this.itemId", 1001]}]}
    ]},
    "then": -1
    }],
    "default": 0
    }}
    ]
    },
    "bers_greaves": {
    "$sum": [
        "$$value.boots",
        { "$switch": {
        "branches": [
            {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_PURCHASED"]}, { "$eq": ["$$this.itemId", 3006]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.afterId", 3006]}]}
    ]},
    "then": 1
    },
    {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_DESTROYED"]}, { "$eq": ["$$this.itemId", 3006]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.beforeId", 3006]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_SOLD"]}, { "$eq": ["$$this.itemId", 3006]}]}
    ]},
    "then": -1
    }],
    "default": 0
    }}
    ]
    },
    "boots_swiftn": {
    "$sum": [
        "$$value.boots_swiftn",
        { "$switch": {
        "branches": [
            {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_PURCHASED"]}, { "$eq": ["$$this.itemId", 3009]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.afterId", 3009]}]}
    ]},
    "then": 1
    },
    {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_DESTROYED"]}, { "$eq": ["$$this.itemId", 3009]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.beforeId", 3009]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_SOLD"]}, { "$eq": ["$$this.itemId", 3009]}]}
    ]},
    "then": -1
    }],
    "default": 0
    }}
    ]
    },
    "mobi_boots": {
    "$sum": [
        "$$value.mobi_boots",
        { "$switch": {
        "branches": [
            {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_PURCHASED"]}, { "$eq": ["$$this.itemId", 3117]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.afterId", 3117]}]}
    ]},
    "then": 1
    },
    {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_DESTROYED"]}, { "$eq": ["$$this.itemId", 3117]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.beforeId", 3117]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_SOLD"]}, { "$eq": ["$$this.itemId", 3117]}]}
    ]},
    "then": -1
    }],
    "default": 0
    }}
    ]
    },
    "ninja_tabi": {
    "$sum": [
        "$$value.ninja_tabi",
        { "$switch": {
        "branches": [
            {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_PURCHASED"]}, { "$eq": ["$$this.itemId", 3047]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.afterId", 3047]}]}
    ]},
    "then": 1
    },
    {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_DESTROYED"]}, { "$eq": ["$$this.itemId", 3047]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.beforeId", 3047]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_SOLD"]}, { "$eq": ["$$this.itemId", 3047]}]}
    ]},
    "then": -1
    }],
    "default": 0
    }}
    ]
    },
    "boots_luci": {
    "$sum": [
        "$$value.boots_luci",
        { "$switch": {
        "branches": [
            {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_PURCHASED"]}, { "$eq": ["$$this.itemId", 3158]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.afterId", 3158]}]}
    ]},
    "then": 1
    },
    {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_DESTROYED"]}, { "$eq": ["$$this.itemId", 3158]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.beforeId", 3158]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_SOLD"]}, { "$eq": ["$$this.itemId", 3158]}]}
    ]},
    "then": -1
    }],
    "default": 0
    }}
    ]
    },
    "merc_treads": {
    "$sum": [
        "$$value.boots_luci",
        { "$switch": {
        "branches": [
            {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_PURCHASED"]}, { "$eq": ["$$this.itemId", 3111]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.afterId", 3111]}]}
    ]},
    "then": 1
    },
    {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_DESTROYED"]}, { "$eq": ["$$this.itemId", 3111]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.beforeId", 3111]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_SOLD"]}, { "$eq": ["$$this.itemId", 3111]}]}
    ]},
    "then": -1
    }],
    "default": 0
    }}
    ]
    },
    "long_sword": {
    "$sum": [
        "$$value.long_sword",
        { "$switch": {
        "branches": [
            {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_PURCHASED"]}, { "$eq": ["$$this.itemId", 1036]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.afterId", 1036]}]}
    ]},
    "then": 1
    },
    {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_DESTROYED"]}, { "$eq": ["$$this.itemId", 1036]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.beforeId", 1036]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_SOLD"]}, { "$eq": ["$$this.itemId", 1036]}]}
    ]},
    "then": -1
    }],
    "default": 0
    }}
    ]
    },
    "dagger": {
    "$sum": [
        "$$value.dagger",
        { "$switch": {
        "branches": [
            {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_PURCHASED"]}, { "$eq": ["$$this.itemId", 1042]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.afterId", 1042]}]}
    ]},
    "then": 1
    },
    {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_DESTROYED"]}, { "$eq": ["$$this.itemId", 1042]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.beforeId", 1042]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_SOLD"]}, { "$eq": ["$$this.itemId", 1042]}]}
    ]},
    "then": -1
    }],
    "default": 0
    }}
    ]
    },
    "brawl_gloves": {
    "$sum": [
        "$$value.brawl_gloves",
        { "$switch": {
        "branches": [
            {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_PURCHASED"]}, { "$eq": ["$$this.itemId", 1051]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.afterId", 1051]}]}
    ]},
    "then": 1
    },
    {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_DESTROYED"]}, { "$eq": ["$$this.itemId", 1051]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.beforeId", 1051]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_SOLD"]}, { "$eq": ["$$this.itemId", 1051]}]}
    ]},
    "then": -1
    }],
    "default": 0
    }}
    ]
    },
    "cloak_oa": {
    "$sum": [
        "$$value.cloak_oa",
        { "$switch": {
        "branches": [
            {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_PURCHASED"]}, { "$eq": ["$$this.itemId", 1018]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.afterId", 1018]}]}
    ]},
    "then": 1
    },
    {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_DESTROYED"]}, { "$eq": ["$$this.itemId", 1018]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.beforeId", 1018]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_SOLD"]}, { "$eq": ["$$this.itemId", 1018]}]}
    ]},
    "then": -1
    }],
    "default": 0
    }}
    ]
    },
    "ruby_crytal": {
    "$sum": [
        "$$value.ruby_crytal",
        { "$switch": {
        "branches": [
            {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_PURCHASED"]}, { "$eq": ["$$this.itemId", 1028]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.afterId", 1028]}]}
    ]},
    "then": 1
    },
    {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_DESTROYED"]}, { "$eq": ["$$this.itemId", 1028]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.beforeId", 1028]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_SOLD"]}, { "$eq": ["$$this.itemId", 1028]}]}
    ]},
    "then": -1
    }],
    "default": 0
    }}
    ]
    },
    "saph_crytal": {
    "$sum": [
        "$$value.saph_crytal",
        { "$switch": {
        "branches": [
            {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_PURCHASED"]}, { "$eq": ["$$this.itemId", 1027]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.afterId", 1027]}]}
    ]},
    "then": 1
    },
    {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_DESTROYED"]}, { "$eq": ["$$this.itemId", 1027]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.beforeId", 1027]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_SOLD"]}, { "$eq": ["$$this.itemId", 1027]}]}
    ]},
    "then": -1
    }],
    "default": 0
    }}
    ]
    },
    "blasting_wand": {
    "$sum": [
        "$$value.blasting_wand",
        { "$switch": {
        "branches": [
            {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_PURCHASED"]}, { "$eq": ["$$this.itemId", 1026]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.afterId", 1026]}]}
    ]},
    "then": 1
    },
    {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_DESTROYED"]}, { "$eq": ["$$this.itemId", 1026]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.beforeId", 1026]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_SOLD"]}, { "$eq": ["$$this.itemId", 1026]}]}
    ]},
    "then": -1
    }],
    "default": 0
    }}
    ]
    },
    "pickaxe": {
    "$sum": [
        "$$value.pickaxe",
        { "$switch": {
        "branches": [
            {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_PURCHASED"]}, { "$eq": ["$$this.itemId", 1037]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.afterId", 1037]}]}
    ]},
    "then": 1
    },
    {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_DESTROYED"]}, { "$eq": ["$$this.itemId", 1037]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.beforeId", 1037]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_SOLD"]}, { "$eq": ["$$this.itemId", 1037]}]}
    ]},
    "then": -1
    }],
    "default": 0
    }}
    ]
    },
    "bf_sword": {
    "$sum": [
        "$$value.bf_sword",
        { "$switch": {
        "branches": [
            {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_PURCHASED"]}, { "$eq": ["$$this.itemId", 1038]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.afterId", 1038]}]}
    ]},
    "then": 1
    },
    {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_DESTROYED"]}, { "$eq": ["$$this.itemId", 1038]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.beforeId", 1038]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_SOLD"]}, { "$eq": ["$$this.itemId", 1038]}]}
    ]},
    "then": -1
    }],
    "default": 0
    }}
    ]
    },
    "serrated_dirk": {
    "$sum": [
        "$$value.serrated_dirk",
        { "$switch": {
        "branches": [
            {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_PURCHASED"]}, { "$eq": ["$$this.itemId", 3134]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.afterId", 3134]}]}
    ]},
    "then": 1
    },
    {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_DESTROYED"]}, { "$eq": ["$$this.itemId", 3134]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.beforeId", 3134]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_SOLD"]}, { "$eq": ["$$this.itemId", 3134]}]}
    ]},
    "then": -1
    }],
    "default": 0
    }}
    ]
    },
    "cf_warhammer": {
    "$sum": [
        "$$value.cf_warhammer",
        { "$switch": {
        "branches": [
            {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_PURCHASED"]}, { "$eq": ["$$this.itemId", 3133]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.afterId", 3133]}]}
    ]},
    "then": 1
    },
    {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_DESTROYED"]}, { "$eq": ["$$this.itemId", 3133]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.beforeId", 3133]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_SOLD"]}, { "$eq": ["$$this.itemId", 3133]}]}
    ]},
    "then": -1
    }],
    "default": 0
    }}
    ]
    },
    "vamp_scepter": {
    "$sum": [
        "$$value.vamp_scepter",
        { "$switch": {
        "branches": [
            {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_PURCHASED"]}, { "$eq": ["$$this.itemId", 1053]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.afterId", 1053]}]}
    ]},
    "then": 1
    },
    {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_DESTROYED"]}, { "$eq": ["$$this.itemId", 1053]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.beforeId", 1053]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_SOLD"]}, { "$eq": ["$$this.itemId", 1053]}]}
    ]},
    "then": -1
    }],
    "default": 0
    }}
    ]
    },
    "cutlass": {
    "$sum": [
        "$$value.cutlass",
        { "$switch": {
        "branches": [
            {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_PURCHASED"]}, { "$eq": ["$$this.itemId", 3144]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.afterId", 3144]}]}
    ]},
    "then": 1
    },
    {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_DESTROYED"]}, { "$eq": ["$$this.itemId", 3144]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.beforeId", 3144]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_SOLD"]}, { "$eq": ["$$this.itemId", 3144]}]}
    ]},
    "then": -1
    }],
    "default": 0
    }}
    ]
    },
    "recurve_bow": {
    "$sum": [
        "$$value.recurve_bow",
        { "$switch": {
        "branches": [
            {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_PURCHASED"]}, { "$eq": ["$$this.itemId", 1043]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.afterId", 1043]}]}
    ]},
    "then": 1
    },
    {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_DESTROYED"]}, { "$eq": ["$$this.itemId", 1043]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.beforeId", 1043]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_SOLD"]}, { "$eq": ["$$this.itemId", 1043]}]}
    ]},
    "then": -1
    }],
    "default": 0
    }}
    ]
    },
    "zeal": {
    "$sum": [
        "$$value.zeal",
        { "$switch": {
        "branches": [
            {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_PURCHASED"]}, { "$eq": ["$$this.itemId", 3086]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.afterId", 3086]}]}
    ]},
    "then": 1
    },
    {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_DESTROYED"]}, { "$eq": ["$$this.itemId", 3086]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.beforeId", 3086]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_SOLD"]}, { "$eq": ["$$this.itemId", 3086]}]}
    ]},
    "then": -1
    }],
    "default": 0
    }}
    ]
    },
    "kircheis_shard": {
    "$sum": [
        "$$value.kircheis_shard",
        { "$switch": {
        "branches": [
            {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_PURCHASED"]}, { "$eq": ["$$this.itemId", 2015]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.afterId", 2015]}]}
    ]},
    "then": 1
    },
    {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_DESTROYED"]}, { "$eq": ["$$this.itemId", 2015]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.beforeId", 2015]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_SOLD"]}, { "$eq": ["$$this.itemId", 2015]}]}
    ]},
    "then": -1
    }],
    "default": 0
    }}
    ]
    },
    "sheen": {
    "$sum": [
        "$$value.sheen",
        { "$switch": {
        "branches": [
            {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_PURCHASED"]}, { "$eq": ["$$this.itemId", 3057]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.afterId", 3057]}]}
    ]},
    "then": 1
    },
    {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_DESTROYED"]}, { "$eq": ["$$this.itemId", 3057]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.beforeId", 3057]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_SOLD"]}, { "$eq": ["$$this.itemId", 3057]}]}
    ]},
    "then": -1
    }],
    "default": 0
    }}
    ]
    },
    "phage": {
    "$sum": [
        "$$value.phage",
        { "$switch": {
        "branches": [
            {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_PURCHASED"]}, { "$eq": ["$$this.itemId", 3044]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.afterId", 3044]}]}
    ]},
    "then": 1
    },
    {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_DESTROYED"]}, { "$eq": ["$$this.itemId", 3044]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.beforeId", 3044]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_SOLD"]}, { "$eq": ["$$this.itemId", 3044]}]}
    ]},
    "then": -1
    }],
    "default": 0
    }}
    ]
    },
    "p_dancer": {
    "$sum": [
        "$$value.p_dancer",
        { "$switch": {
        "branches": [
            {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_PURCHASED"]}, { "$eq": ["$$this.itemId", 3046]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.afterId", 3046]}]}
    ]},
    "then": 1
    },
    {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_DESTROYED"]}, { "$eq": ["$$this.itemId", 3046]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.beforeId", 3046]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_SOLD"]}, { "$eq": ["$$this.itemId", 3046]}]}
    ]},
    "then": -1
    }],
    "default": 0
    }}
    ]
    },
    "s_shivv": {
    "$sum": [
        "$$value.s_shivv",
        { "$switch": {
        "branches": [
            {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_PURCHASED"]}, { "$eq": ["$$this.itemId", 3087]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.afterId", 3087]}]}
    ]},
    "then": 1
    },
    {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_DESTROYED"]}, { "$eq": ["$$this.itemId", 3087]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.beforeId", 3087]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_SOLD"]}, { "$eq": ["$$this.itemId", 3087]}]}
    ]},
    "then": -1
    }],
    "default": 0
    }}
    ]
    },
    "runaans_h": {
    "$sum": [
        "$$value.runaans_h",
        { "$switch": {
        "branches": [
            {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_PURCHASED"]}, { "$eq": ["$$this.itemId", 3085]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.afterId", 3085]}]}
    ]},
    "then": 1
    },
    {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_DESTROYED"]}, { "$eq": ["$$this.itemId", 3085]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.beforeId", 3085]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_SOLD"]}, { "$eq": ["$$this.itemId", 3085]}]}
    ]},
    "then": -1
    }],
    "default": 0
    }}
    ]
    },
    "rapid_fc": {
    "$sum": [
        "$$value.rapid_fc",
        { "$switch": {
        "branches": [
            {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_PURCHASED"]}, { "$eq": ["$$this.itemId", 3094]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.afterId", 3094]}]}
    ]},
    "then": 1
    },
    {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_DESTROYED"]}, { "$eq": ["$$this.itemId", 3094]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.beforeId", 3094]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_SOLD"]}, { "$eq": ["$$this.itemId", 3094]}]}
    ]},
    "then": -1
    }],
    "default": 0
    }}
    ]
    },
    "tri_force": {
    "$sum": [
        "$$value.tri_force",
        { "$switch": {
        "branches": [
            {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_PURCHASED"]}, { "$eq": ["$$this.itemId", 3078]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.afterId", 3078]}]}
    ]},
    "then": 1
    },
    {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_DESTROYED"]}, { "$eq": ["$$this.itemId", 3078]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.beforeId", 3078]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_SOLD"]}, { "$eq": ["$$this.itemId", 3078]}]}
    ]},
    "then": -1
    }],
    "default": 0
    }}
    ]
    },
    "ie": {
    "$sum": [
        "$$value.ie",
        { "$switch": {
        "branches": [
            {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_PURCHASED"]}, { "$eq": ["$$this.itemId", 3031]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.afterId", 3031]}]}
    ]},
    "then": 1
    },
    {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_DESTROYED"]}, { "$eq": ["$$this.itemId", 3031]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.beforeId", 3031]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_SOLD"]}, { "$eq": ["$$this.itemId", 3031]}]}
    ]},
    "then": -1
    }],
    "default": 0
    }}
    ]
    },
    "er": {
    "$sum": [
        "$$value.er",
        { "$switch": {
        "branches": [
            {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_PURCHASED"]}, { "$eq": ["$$this.itemId", 3058]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.afterId", 3058]}]}
    ]},
    "then": 1
    },
    {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_DESTROYED"]}, { "$eq": ["$$this.itemId", 3058]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.beforeId", 3058]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_SOLD"]}, { "$eq": ["$$this.itemId", 3058]}]}
    ]},
    "then": -1
    }],
    "default": 0
    }}
    ]
    },
    "gb": {
    "$sum": [
        "$$value.gb",
        { "$switch": {
        "branches": [
            {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_PURCHASED"]}, { "$eq": ["$$this.itemId", 3142]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.afterId", 3142]}]}
    ]},
    "then": 1
    },
    {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_DESTROYED"]}, { "$eq": ["$$this.itemId", 3142]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.beforeId", 3142]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_SOLD"]}, { "$eq": ["$$this.itemId", 3142]}]}
    ]},
    "then": -1
    }],
    "default": 0
    }}
    ]
    },
    "duskblade": {
    "$sum": [
        "$$value.duskblade",
        { "$switch": {
        "branches": [
            {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_PURCHASED"]}, { "$eq": ["$$this.itemId", 3147]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.afterId", 3147]}]}
    ]},
    "then": 1
    },
    {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_DESTROYED"]}, { "$eq": ["$$this.itemId", 3147]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.beforeId", 3147]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_SOLD"]}, { "$eq": ["$$this.itemId", 3147]}]}
    ]},
    "then": -1
    }],
    "default": 0
    }}
    ]
    },
    "eon": {
    "$sum": [
        "$$value.eon",
        { "$switch": {
        "branches": [
            {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_PURCHASED"]}, { "$eq": ["$$this.itemId", 3814]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.afterId", 3814]}]}
    ]},
    "then": 1
    },
    {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_DESTROYED"]}, { "$eq": ["$$this.itemId", 3814]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.beforeId", 3814]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_SOLD"]}, { "$eq": ["$$this.itemId", 3814]}]}
    ]},
    "then": -1
    }],
    "default": 0
    }}
    ]
    },
    "ibg": {
    "$sum": [
        "$$value.ibg",
        { "$switch": {
        "branches": [
            {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_PURCHASED"]}, { "$eq": ["$$this.itemId", 3025]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.afterId", 3025]}]}
    ]},
    "then": 1
    },
    {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_DESTROYED"]}, { "$eq": ["$$this.itemId", 3025]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.beforeId", 3025]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_SOLD"]}, { "$eq": ["$$this.itemId", 3025]}]}
    ]},
    "then": -1
    }],
    "default": 0
    }}
    ]
    },
    "ibg": {
    "$sum": [
        "$$value.ibg",
        { "$switch": {
        "branches": [
            {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_PURCHASED"]}, { "$eq": ["$$this.itemId", 3025]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.afterId", 3025]}]}
    ]},
    "then": 1
    },
    {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_DESTROYED"]}, { "$eq": ["$$this.itemId", 3025]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.beforeId", 3025]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_SOLD"]}, { "$eq": ["$$this.itemId", 3025]}]}
    ]},
    "then": -1
    }],
    "default": 0
    }}
    ]
    },
    "wits_end": {
    "$sum": [
        "$$value.wits_end",
        { "$switch": {
        "branches": [
            {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_PURCHASED"]}, { "$eq": ["$$this.itemId", 3091]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.afterId", 3091]}]}
    ]},
    "then": 1
    },
    {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_DESTROYED"]}, { "$eq": ["$$this.itemId", 3091]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.beforeId", 3091]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_SOLD"]}, { "$eq": ["$$this.itemId", 3091]}]}
    ]},
    "then": -1
    }],
    "default": 0
    }}
    ]
    },
    "g_rageblade": {
    "$sum": [
        "$$value.g_rageblade",
        { "$switch": {
        "branches": [
            {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_PURCHASED"]}, { "$eq": ["$$this.itemId", 3124]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.afterId", 3124]}]}
    ]},
    "then": 1
    },
    {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_DESTROYED"]}, { "$eq": ["$$this.itemId", 3124]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.beforeId", 3124]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_SOLD"]}, { "$eq": ["$$this.itemId", 3124]}]}
    ]},
    "then": -1
    }],
    "default": 0
    }}
    ]
    },
    "guard_angel": {
    "$sum": [
        "$$value.guard_angel",
        { "$switch": {
        "branches": [
            {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_PURCHASED"]}, { "$eq": ["$$this.itemId", 3026]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.afterId", 3026]}]}
    ]},
    "then": 1
    },
    {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_DESTROYED"]}, { "$eq": ["$$this.itemId", 3026]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.beforeId", 3026]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_SOLD"]}, { "$eq": ["$$this.itemId", 3026]}]}
    ]},
    "then": -1
    }],
    "default": 0
    }}
    ]
    },
    "last_whisper": {
    "$sum": [
        "$$value.last_whisper",
        { "$switch": {
        "branches": [
            {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_PURCHASED"]}, { "$eq": ["$$this.itemId", 3035]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.afterId", 3035]}]}
    ]},
    "then": 1
    },
    {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_DESTROYED"]}, { "$eq": ["$$this.itemId", 3035]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.beforeId", 3035]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_SOLD"]}, { "$eq": ["$$this.itemId", 3035]}]}
    ]},
    "then": -1
    }],
    "default": 0
    }}
    ]
    },
    "lord_dom": {
    "$sum": [
        "$$value.lord_dom",
        { "$switch": {
        "branches": [
            {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_PURCHASED"]}, { "$eq": ["$$this.itemId", 3036]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.afterId", 3036]}]}
    ]},
    "then": 1
    },
    {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_DESTROYED"]}, { "$eq": ["$$this.itemId", 3036]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.beforeId", 3036]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_SOLD"]}, { "$eq": ["$$this.itemId", 3036]}]}
    ]},
    "then": -1
    }],
    "default": 0
    }}
    ]
    },
    "mortal_rem": {
    "$sum": [
        "$$value.mortal_rem",
        { "$switch": {
        "branches": [
            {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_PURCHASED"]}, { "$eq": ["$$this.itemId", 3033]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.afterId", 3033]}]}
    ]},
    "then": 1
    },
    {"case": { "$or": [
        { "$and": [{ "$eq": ["$$this.type", "ITEM_DESTROYED"]}, { "$eq": ["$$this.itemId", 3033]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_UNDO"]}, { "$eq": ["$$this.beforeId", 3033]}]},
    { "$and": [{ "$eq": ["$$this.type", "ITEM_SOLD"]}, { "$eq": ["$$this.itemId", 3033]}]}
    ]},
    "then": -1
    }],
    "default": 0
    }}
    ]
    },
    "data": { "$concatArrays": [
        "$$value.data",
        {"$cond": [
        { "$or": [{"$eq": ["$$this.type", "ITEM_PURCHASED"]}, {"$eq": ["$$this.type", "ITEM_SOLD"]}]},
    [
        {
            "_id": "$$this._id",
            "gameId": "$$this.gameId",
            "platformId": "$$this.platformId",
            "patch": "$$this.patch",
            "frameNo": "$$this.frameNo",
            "participantId": "$$this.participantId",
            "type": "$$this.type",
            "itemId": "$$this.itemId",
            "timestamp": "$$this.timestamp",
            "kills": "$$value.kills",
            "deaths": "$$value.deaths",
            "assists": "$$value.assists",
            "q_rank": "$$value.q_rank",
            "w_rank": "$$value.w_rank",
            "e_rank": "$$value.e_rank",
            "r_rank": "$$value.r_rank",
            "dorans_blade": "$$value.dorans_blade",
            "dorans_ring": "$$value.dorans_ring",
            "dorans_shield": "$$value.dorans_shield",
            "boots": "$$value.boots",
            "bers_grieves": "$$value.bers_grieves",
            "mobi_boots": "$$value.mobi_boots",
            "boots_swiftn": "$$value.boots_swiftn",
            "ninja_tabi": "$$value.ninja_tabi",
            "boots_luci": "$$value.boots_luci",
            "merc_treads": "$$value.merc_treads",
            "long_sword": "$$value.long_sword",
            "dagger": "$$value.dagger",
            "brawl_gloves": "$$value.brawl_gloves",
            "cloak_oa": "$$value.cloak_oa",
            "ruby_crystal": "$$value.ruby_crystal",
            "saph_crystal": "$$value.saph_crystal",
            "blasting_wand": "$$value.blasting_wand",
            "pickaxe": "$$value.pickaxe",
            "serrated_dirk": "$$value.serrated_dirk",
            "cf_warhammer": "$$value.cf_warhammer",
            "vamp_scepter": "$$value.vamp_scepter",
            "cutlass": "$$value.cutlass",
            "recurve_bow": "$$value.recurve_bow",
            "kircheis_shard": "$$value.kircheis_shard",
            "sheen": "$$value.sheen",
            "phage": "$$value.phage",
            "p_dancer": "$$value.p_dancer",
            "s_shivv": "$$value.s_shivv",
            "runaans_h": "$$value.runaans_h",
            "rapid_fc": "$$value.rapid_fc",
            "tri_force": "$$value.tri_force",
            "ie": "$$value.ie",
            "er": "$$value.er",
            "gb": "$$value.gb",
            "duskblade": "$$value.duskblade",
            "eon": "$$value.eon",
            "ibg": "$$value.ibg",
            "wits_end": "$$value.wits_end",
            "g_rageblade": "$$value.g_rageblade",
            "guard_angel": "$$value.guard_angel",
            "last_whisper": "$$value.last_whisper",
            "lord_dom": "$$value.lord_dom",
            "mortal_rem": "$$value.mortal_rem"
        }
    ],
    []
    ]}
    ]}
    }
    }}
    }},
    {"$unwind": "$data.data"},
    {"$replaceRoot": {"newRoot": "$data.data"}},
    {"$out": "jhin_training_set"}
    ]