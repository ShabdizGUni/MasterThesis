columns_blank_item = ["gameId", "platformId", "patch", "frameNo", "participantId",
                      "side", "type", "itemId", "timestamp"]

inventory = [
    "dorans_blade",
    "dorans_ring",
    "dorans_shield",
    "boots",
    "bers_greaves",
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
    "kindlegem",
    "nullmag_mantle",
    "negatron",
    "qss",
    "cloth",
    "chainvest",
    "blasting_wand",
    "pickaxe",
    "bf_sword",
    "serrated_dirk",
    "cf_warhammer",
    "vamp_scepter",
    "cutlass",
    "recurve_bow",
    "zeal",
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
    "muramana",
    "ibg",
    "wits_end",
    "g_rageblade",
    "guard_angel",
    "hexdrinker",
    "maw",
    "merc_scimitar",
    "bt",
    "dd",
    "last_whisper",
    "giant_slayer",
    "exec_calling",
    "black_cleaver",
    "lord_dom",
    "mortal_rem"
]

columns_pre_game = columns_blank_item + ["tier",
                                         "masteryId",
                                         "championId"]

columns_in_game = columns_pre_game + ["q_rank",
                                      "w_rank",
                                      "e_rank",
                                      "r_rank"]

columns_inventory = columns_in_game + inventory

columns_performance = columns_inventory + ["kills",
                                           "deaths",
                                           "assists",
                                           "turretKills",
                                           "turretAssists",
                                           "xp",
                                           "minionsKilled",
                                           "jungleMinionsKilled",
                                           "totalGold",
                                           "availGold"]

columns_teams = columns_performance + ["teamTurrets",
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
                                       "teamElder",
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

item_columns = [
    # Infinity Edge
    "ie_goldTotal",
    "ie_goldBase",
    "ie_goldSell",
    "ie_FlatCritChanceMod",
    "ie_FlatPhysicalDamageMod",
    "ie_e1",
    # Essence Reaver
    "er_goldTotal",
    "er_goldBase",
    "er_goldSell",
    "er_FlatCritChanceMod",
    "er_FlatPhysicalDamageMod",
    "er_e1",
    "er_e2",
    "er_e3",
    "er_e4",
    # BORK
    "bork_goldTotal",
    "bork_goldBase",
    "bork_goldSell",
    "bork_FlatPhysicalDamageMod",
    "bork_PercentAttackSpeedMod",
    "bork_PercentLifeStealMod",
    # Edge of Night
    "eon_goldTotal",
    "eon_goldBase",
    "eon_goldSell",
    "eon_FlatHPPoolMod",
    "eon_FlatPhysicalDamageMod",
    "eon_FlatSpellBlockMod",
    "eon_e1",
    "eon_e2",
    "eon_e3",
    "eon_e4",
    # Black Cleaver
    "bc_goldTotal",
    "bc_goldBase",
    "bc_goldSell",
    "bc_FlatHPPoolMod",
    "bc_e2",
    "bc_e5",
    # Duskblade
    "db_goldTotal",
    "db_goldBase",
    "db_goldSell",
    "db_FlatPhysicalDamageMod",
    "db_e1",
    "db_e2",
    "db_e5",
    "db_e6",
    "db_e7",
    "db_e8",
    "db_e10",
    "db_e12",
    # Zeal
    "z_goldTotal",
    "z_goldBase",
    "z_goldSell",
    # Rapid FireCannon
    "rfc_goldTotal",
    "rfc_goldBase",
    "rfc_goldSell",
    "rfc_e6",
    "rfc_e3",
    "rfc_e4",
    # Statikk Shiv
    "ss_goldTotal",
    "ss_goldBase",
    "ss_goldSell",
    "ss_e5",
    "ss_e6",
    "ss_e9",
    "ss_e10",
    # Runaan's
    "rh_goldTotal",
    "rh_goldBase",
    "rh_goldSell",
    "rh_e2",
    "rh_e4",
    # Phantom Dancer
    "pd_goldTotal",
    "pd_goldBase",
    "pd_goldSell",
]
