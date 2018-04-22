columns_blank = ["gameId", "platformId", "patch", "frameNo", "participantId", "side",
                 "championId", "type", "itemId", "timestamp"]

inventory = ["q_rank",
             "w_rank",
             "e_rank",
             "r_rank",
             "dorans_blade",
             "dorans_ring",
             "dorans_shield",
             "boots",
             "bers_grieves",
             "mobi_boots",
             "boots_swiftn",
             "ninja_tabi",
             "boots_luci",
             "merc_treads",
             "long_sword",
             "dagger",
             "brawl_gloves",
             "cloak_oa",
             "ruby_crystal",
             "saph_crystal",
             "blasting_wand",
             "pickaxe",
             "serrated_dirk",
             "cf_warhammer",
             "vamp_scepter",
             "cutlass",
             "recurve_bow",
             "kircheis_shard",
             "sheen",
             "phage",
             "p_dancer",
             "s_shivv",
             "runaans_h",
             "rapid_fc",
             "tri_force",
             "ie",
             "er",
             "gb",
             "duskblade",
             "eon",
             "ibg",
             "wits_end",
             "g_rageblade",
             "guard_angel",
             "last_whisper",
             "lord_dom",
             "mortal_rem"
             ]


columns_pre_game = columns_blank + ["tier",
                                    "side",
                                    "masteryId"]

columns_in_game = columns_pre_game + inventory

columns_performance = columns_in_game + ["kills",
                                         "deaths",
                                         "assists",
                                         "turretKills",
                                         "turretAssists",
                                         "xp",
                                         "minionsKilled",
                                         "jungleMinionsKilled",
                                         "totalGold",
                                         "availGold"]

columns_team_enemy = columns_performance + ["teamTurrets",
                                            "enemyTurrets",
                                            "teamMidT",
                                            "teamBotT",
                                            "teamTopT",
                                            "enemyMidT",
                                            "enemyBotT",
                                            "enemyTopT",
                                            "teamInh",
                                            "teamMidInh",
                                            "teamTopInh",
                                            "teamBotInh",
                                            "enemyInh",
                                            "enemyMidInh",
                                            "enemyBotInh",
                                            "enemyTopInh",
                                            "teamHeralds",
                                            "enemyHeralds",
                                            "teamBarons",
                                            "teamDragons",
                                            "teamInfernal",
                                            "teamMountain",
                                            "teamOcean",
                                            "teamCloud",
                                            "teamElder,"
                                            "enemyInfernal",
                                            "enemyMountain",
                                            "enemyOcean",
                                            "enemyCloud",
                                            "enemyElder"]


itemStatSet = [
    ("Infinity Edge", 'ie', [
        "goldTotal",
        "goldBase",
        "goldSell",
        "FlatCritChanceMod",
        "FlatPhysicalDamageMod",
        "e1"]),
    ("Essence Reaver", 'er', [
        "goldTotal",
        "goldBase",
        "goldSell",
        "FlatCritChanceMod",
        "FlatPhysicalDamageMod",
        "e1",
        "e2",
        "e3",
        "e4"
    ]),
    ("Blade of the Ruined King", 'bork', [
        "goldTotal",
        "goldBase",
        "goldSell",
        "FlatPhysicalDamageMod",
        "PercentAttackSpeedMod",
        "PercentLifeStealMod",
    ]),
    ("Edge of Night", 'eon', [
        "goldTotal",
        "goldBase",
        "goldSell",
        "FlatHPPoolMod",
        "FlatPhysicalDamageMod",
        "FlatSpellBlockMod",
        "e1",  # Lethality
        "e2",  # Active Duration (adjusted manually!)
        "e3",  # Movementspeed out of Combat
        "e4",  # Active Cooldown
    ]),
    ("The Black Cleaver", 'bc', [
        "goldTotal",
        "goldBase",
        "goldSell",
        "FlatHPPoolMod",
        "e2",
        "e5"
    ]),
    # original data needs to be adjusted: bonus dmg -> bonus dmg less but now scaling
    # therefore: Bonus Dmg before 7.14: 55 or 75  200% Lethality:
    # 55 or 75 + 15 * 2 = 85 or 105
    # maximum amount of bonus damage: 20 leth (ghostblade) +  15 leth (edge of night) + 10 leth (serrated dirk)
    # = > 45 lethality * 200% = 90 maximum bonus damage possible
    ("Duskblade of Draktharr", 'db', [
        "goldTotal",
        "goldBase",
        "goldSell",
        "FlatPhysicalDamageMod",
        "e1",  # Lethality
        "e2",  # Out of Combat Movementspeed
        "e5",  # min active damage
        "e6",  # max active damage
        "e7",  # slow percentage
        "e8",  # slow duration
        "e10",  # ranged min damage
        "e12"  # ranged max damage
    ]),
    ("Zeal", 'z', [
        "goldTotal",
        "goldBase",
        "goldSell"
    ]),
    ("Rapid FireCannon", 'rfc', [
        "goldTotal",
        "goldBase",
        "goldSell",
        "e6",  # energized attack rate
        "e3",  # min bonus damage
        "e4"  # max bonus damage
    ]),
    ("Statikk Shiv", 'ss', [
        "goldTotal",
        "goldBase",
        "goldSell",
        "e5",  # min damage charge
        "e6",  # max damage charge
        "e9",  # min damage to minions (ADJUSTED)
        "e10"  # max damage to minions (ADJUSTED)
    ]),
    ("Runaan\\'s Hurricane", 'rh', [
        "goldTotal",
        "goldBase",
        "goldSell",
        "e2",  # AD Ratio
        "e4",  # Bonus on-hit Damage
    ]),
    ("Phantom Dancer", 'pd', [
        "goldTotal",
        "goldBase",
        "goldSell"
    ])
]

# note: needed to adjust some values manually because
# of inconsistencies in the static data endpoint
# see: https://discussion.developer.riotgames.com/questions/5178/edge-of-night-erronous-entry.html
# and: https://discussion.developer.riotgames.com/questions/1596/known-issue-static-data-errorsmissing-data.html
