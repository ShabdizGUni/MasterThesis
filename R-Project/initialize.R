list.of.packages <- c("readr", 
                      "ggplot2",
                      "data.table",
                      "RMySQL",
                      "DBI",
                      "lubridate",
                      "scales",
                      "ggthemes",
                      "gridExtra",
                      "RColorBrewer",
                      "ggiraph",
                      "dplyr",
                      "tibble"
                      )
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

#library(readr)
library(ggplot2)
library(data.table)
library(RMySQL)
library(DBI)
library(lubridate)
library(ggradar)
library(ggthemes)
library(gridExtra)
require(scales)
library(RColorBrewer)
library(ggiraphExtra)
require(dplyr)
require(tibble)
library(rgl)

# especially important for weekday labels
Sys.setlocale(category = "LC_ALL", locale = "english")

coord_radar <- function (theta = "x", start = 0, direction = 1) 
{
  theta <- match.arg(theta, c("x", "y"))
  r <- if (theta == "x") 
    "y"
  else "x"
  ggproto("CordRadar", CoordPolar, theta = theta, r = r, start = start, 
          direction = sign(direction),
          is_linear = function(coord) TRUE)
}

# color palettes:
cbPalette <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

connection <-
  dbConnect(RMySQL::MySQL(), user = "root", password = "Milch4321", "leaguestats")

#Soloqueue
#playerstats <- data.table(dbReadTable(connection, "playerdetails"))
#teamstats <- data.table(dbReadTable(connection, "teamdetails"))
# playerstats <- 
#   merge(
#     playerstats,
#     teamstats,
#     by.x = c("gameId", "platformId", "teamId"),
#     by.y = c("gameId", "platformId", "teamId")
#   )

champPerformancePatch <- data.table(dbReadTable(connection, "champ_kpi_patch"))
champPerformancePatchRegion <- data.table(dbReadTable(connection, "champ_kpi_patch_region"))
bans <- data.table(dbReadTable(connection, "bans"))
#champPerformancePatchByWin <- data.table(dbReadTable(connection, "champPerformancePatchWin"))

#ProGames
proplayerstats <- data.table(dbReadTable(connection, "proplayerdetails"))
proteamstats <- data.table(dbReadTable(connection, "proteamdetails"))
playerstats <-
  merge(
    proplayerstats,
    proteamstats,
    by.x = c("gameId", "platformId", "teamId"),
    by.y = c("gameId", "platformId", "teamId")
  )

champPerformancePatchPro <- data.table(dbReadTable(connection, "champPerformancePatchPro"))
champPerformancePatchRegionPro <- data.table(dbReadTable(connection, "champPerformancePatchRegionPro"))
champPerformancePatchByWinPro <- data.table(dbReadTable(connection, "champPerformancePatchWinPro"))

tiers_stats <- dbGetQuery(connection, "select tier, patch, count(*) as count from playerdetails group by tier, patch")

patchOrder = c("6.23",
               "6.24",
               "7.1",
               "7.2",
               "7.3",
               "7.4",
               "7.5",
               "7.6",
               "7.7",
               "7.8",
               "7.9",
               "7.10",
               "7.11",
               "7.12",
               "7.13",
               "7.14",
               "7.15",
               "7.16",
               "7.17",
               "7.18")

tiers_order <- factor(c("CHALLENGER", "MASTER", "DIAMOND", "PLATINUM", "GOLD", "SILVER", "BRONZE", "UNRANKED"),
                      levels=c("CHALLENGER", "MASTER", "DIAMOND", "PLATINUM", "GOLD", "SILVER", "BRONZE", "UNRANKED"))

tier_palette <- c("CHALLENGER"="orange", 
                  "MASTER"="#966F33", 
                  "DIAMOND"="#cbe3f0",
                  "PLATINUM"="#A0BFB4",
                  "GOLD"="#e6c200",
                  "SILVER"="#c0c0c0",
                  "BRONZE"="#cd7f32",
                  "UNRANKED"="black")

xpSteps = c(280,380,480,580,680,780,880,980,1080,1180,1280,1380,1480,1580,1680,1780,1880)

## Items
itemLookUp <- data.table(dbReadTable(connection, "itemkeys"))
setkey(itemLookUp, "id")

item <- itemLookUp$name
names(item) <- itemLookUp$id
itemLookUp[id == 3004]$name = "Manamune/Muramana"
itemLookUp[id == 3042]$name = "Manamune/Muramana"
itemLookUp[id == 3155]$name = "Hexdrinker/Maw"
itemLookUp[id == 3156]$name = "Hexdrinker/Maw"



## Champions
champLookUp <- data.table(dbReadTable(connection, "championkeys"))
setkey(champLookUp, "championId")

champ <- champLookUp$name
names(champ) <- champLookUp$championId


### AD CARRIES
adc <- data.table(dbGetQuery(conn = connection, "SELECT * FROM playerdetails WHERE lane = 'BOTTOM' AND role = 'DUO_CARRY' AND gameDuration >= 900"))
#adc.Pro <- proplayerstats[lane == "BOTTOM" & role == "DUO_CARRY"] 

banlist <- merge(rbind(
                bans[,list(gameId, platformId,patch, ban=ban0)], 
                bans[,list(gameId, platformId,patch, ban=ban1)],
                bans[,list(gameId, platformId,patch, ban=ban2)],
                bans[,list(gameId, platformId,patch, ban=ban3)],
                bans[,list(gameId, platformId,patch, ban=ban4)]
              ),
              champLookUp,
              by.x="ban", by.y="championId"
        )

#bans <- bans[,list(gamesBanned = .N), by = list(gameId, patch, platformId,name)]
bans2 <- banlist[, banned := list(list(sort(unique(name)))) , by=list(gameId, platformId, patch)]

banned2 <- bans2[,list(ban1 = banned[[1]][1], 
                       ban2 = banned[[1]][2], 
                       ban3 = banned[[1]][3], 
                       ban4 = banned[[1]][4], 
                       ban5 = banned[[1]][5], 
                       ban6 = banned[[1]][6], 
                       ban7 = banned[[1]][7], 
                       ban8 = banned[[1]][8], 
                       ban9 = banned[[1]][9], 
                       ban10 = banned[[1]][10]), by = list(gameId,patch,platformId)]

banlist <- rbind(
            banned2[,list(gameId, platformId,patch, name=ban1)],
            banned2[,list(gameId, platformId,patch, name=ban2)],
            banned2[,list(gameId, platformId,patch, name=ban3)],
            banned2[,list(gameId, platformId,patch, name=ban4)],
            banned2[,list(gameId, platformId,patch, name=ban5)],
            banned2[,list(gameId, platformId,patch, name=ban6)],
            banned2[,list(gameId, platformId,patch, name=ban7)],
            banned2[,list(gameId, platformId,patch, name=ban8)],
            banned2[,list(gameId, platformId,patch, name=ban9)],
            banned2[,list(gameId, platformId,patch, name=ban10)]
          )



relchamps.adc <- adc[,list(gamesPlayed = .N), by = championId][order(by = gamesPlayed, decreasing = T)][,.SD[1:16]][,championId]

adc.relevant <- adc[championId %in% relchamps.adc]
adc.performance <- champPerformancePatchRegion[lane=="BOTTOM" & role == "DUO_CARRY" & championId %in% unique(adc.relevant$championId)]
adc.performance <- merge(
                    adc.performance,
                    banlist[, list(gamesBanned=.N), by= list(name, platformId, patch)],
                    by=c("patch", "platformId", "name")
)
adc.performance.patch <- champPerformancePatch[lane == "BOTTOM" & role == "DUO_CARRY" & championId %in% unique(adc.relevant$championId)]
#adc.performance.patch.win <- champPerformancePatchByWin[lane == "BOTTOM" & role == "DUO_CARRY" & championId %in% unique(adc.relevant$championId)]

adc.relevant.patchTier <- merge(
  adc.relevant[, list(gamesPlayed = .N), by= list(championId, tier, patch)],
  tiers_stats[c("tier","patch","count")],
  by=c("tier","patch")
)
adc.relevant.patchTier$playRate  <- adc.relevant.patchTier$gamesPlayed/adc.relevant.patchTier$count
adc.relevant.patchTier <- adc.relevant.patchTier %>%
  merge(champLookUp, by="championId")
adc.relevant.patchTier$order <- factor(adc.relevant.patchTier$tier, levels=tiers_order)

adc.pro.performance <- champPerformancePatchRegionPro[lane=="BOTTOM" & role == "DUO_CARRY" & championId %in% unique(adc.relevant$championId)]
adc.pro.performance.patch <- champPerformancePatchPro[lane == "BOTTOM" & role == "DUO_CARRY" & championId %in% unique(adc.relevant$championId)]
adc.pro.performance.patch.win <- champPerformancePatchByWinPro[lane == "BOTTOM" & role == "DUO_CARRY" & championId %in% unique(adc.relevant$championId)]


adc.distribution <- merge(
  data.table(adc[,list(gamesPlayed = .N), by = championId][order(by = gamesPlayed, decreasing = T)]),
  champLookUp,
  by= "championId"
)
adc.distribution.platform <- merge(
  data.table(adc[,list(gamesPlayed = .N), by = list(championId,platformId)][order(by = gamesPlayed, decreasing = T)]),
  champLookUp,
  by= "championId"
)
adc.distribution.patch <- merge(
  data.table(adc[,list(gamesPlayed = .N), by = list(championId,patch)][order(by = gamesPlayed, decreasing = T)]),
  champLookUp,
  by= "championId"
)
setnames(adc.distribution,"name", "names")

items.adc = rbind(
  adc[championId %in% relchamps.adc,list(championId, platformId, patch, item=item0)], 
  adc[championId %in% relchamps.adc,list(championId, platformId, patch, item=item1)],
  adc[championId %in% relchamps.adc,list(championId, platformId, patch, item=item2)],
  adc[championId %in% relchamps.adc,list(championId, platformId, patch, item=item3)],
  adc[championId %in% relchamps.adc,list(championId, platformId, patch, item=item4)],
  adc[championId %in% relchamps.adc,list(championId, platformId, patch, item=item5)],
  adc[championId %in% relchamps.adc,list(championId, platformId, patch, item=item6)]
)
items.adc <- merge(
  items.adc,
  itemLookUp,
  by.x = "item",
  by.y = "id",
  all.x = T
)
items.adc <- merge(
  items.adc,
  champLookUp,
  by = "championId"
)
setnames(items.adc, c("championId","item", "platformId","patch","itemName", "championName"))
items.adc <- subset(items.adc, !is.na(itemName))


relItems.ADC = c("Infinity Edge", # crit
                 "Essence Reaver",
                 "Manamune/Muramana", # mana
                 "Iceborn Gauntlet",
                 "Trinity Force",  # multiplier
                 "Runaan's Hurricane",
                 "Rapid Firecannon",
                 "Statikk Shiv",
                 "Phantom Dancer",
                 "The Bloodthirster", # defense
                 "Death's Dance",
                 "Hexdrinker/Maw",
                 "Blade of the Ruined King", #  on hit
                 "Guinsoo's Rageblade",
                 "Wit's End",
                 "The Black Cleaver", # leathality/armor pen
                 "Edge of Night",
                 "Duskblade of Draktharr",
                 "Youmuu's Ghostblade")

items.palette = c("#FFD700", #IE
                  "#CFB53B", #ER
                  "#007FFF", #Manamune
                  "#89CFF0", #Iceborn
                  "#a5ffb3", #Tri Force
                  "#70d880", #Runaans
                  "#4cb25c", #Rapid F
                  "#2e963e", #Statikk
                  "#156d23", # phantom
                  "#e85a66", # bt
                  "#ce2b39", # deaths dance
                  "#9b121e",  # maw
                  "#CDADED", # bork
                  "#DF73FF", # guinsoo
                  "#9966CC", # wits end
                  "#D9D6CF", # Black Cleaver
                  "#C9C0BB", # edge of night
                  "#848482", # duskblade
                  "#555555" # yomouus
                  )

## support
sup <- data.table(dbGetQuery(conn = connection, "SELECT * FROM playerdetails WHERE lane = 'BOTTOM' AND role = 'DUO_SUPPORT' AND gameDuration >= 900"))
relchamps.sup <- sup[patch %in% c("7.14","7.15","7.16","7.17","7.18")][,list(gamesPlayed = .N), by = championId][order(by = gamesPlayed, decreasing = T)][,.SD[1:20]][,championId]
sup.relevant <- sup[championId %in% relchamps.sup]
sup.performance <- champPerformancePatchRegion[lane=="BOTTOM" & role == "DUO_SUPPORT"]
sup.performance <- merge(
  sup.performance,
  banlist[, list(gamesBanned=.N), by= list(name, platformId, patch)],
  by=c("patch", "platformId", "name")
)
sup.performance.patch <- champPerformancePatch[lane == "BOTTOM" & role == "DUO_SUPPORT" & championId %in% unique(sup.relevant$championId)]
#sup.performance.patch.win <- champPerformancePatchByWin[lane == "BOTTOM" & role == "DUO_CARRY" & championId %in% unique(adc.relevant$championId)]

sup.distribution <- merge(
  data.table(sup[,list(gamesPlayed = .N), by = championId][order(by = gamesPlayed, decreasing = T)]),
  champLookUp,
  by= "championId"
)
sup.distribution.platform <- merge(
  data.table(sup[,list(gamesPlayed = .N), by = list(championId,platformId)][order(by = gamesPlayed, decreasing = T)]),
  champLookUp,
  by= "championId"
)
sup.distribution.patch <- merge(
  data.table(sup[,list(gamesPlayed = .N), by = list(championId,patch)][order(by = gamesPlayed, decreasing = T)]),
  champLookUp,
  by= "championId"
)
setnames(sup.distribution,"name", "names")

items.sup = rbind(
  sup[championId %in% relchamps.sup,list(gameId, championId, platformId, patch, item=item0)], 
  sup[championId %in% relchamps.sup,list(gameId, championId, platformId, patch, item=item1)],
  sup[championId %in% relchamps.sup,list(gameId, championId, platformId, patch, item=item2)],
  sup[championId %in% relchamps.sup,list(gameId, championId, platformId, patch, item=item3)],
  sup[championId %in% relchamps.sup,list(gameId, championId, platformId, patch, item=item4)],
  sup[championId %in% relchamps.sup,list(gameId, championId, platformId, patch, item=item5)],
  sup[championId %in% relchamps.sup,list(gameId, championId, platformId, patch, item=item6)]
)
items.sup <- merge(
  items.sup,
  itemLookUp,
  by.x = "item",
  by.y = "id",
  all.x = T,
  allow.cartesian = T
)
items.sup <- merge(
  items.sup,
  champLookUp,
  by = "championId"
)
setnames(items.sup, c("championId","item", "gameId","platformId","patch","itemName", "championName"))
items.sup <- subset(items.sup, !is.na(itemName))


# botlane adc+sup:
# botlane = data.table(
#   merge(adc.relevant,
#         sup.relevant,
#         by = c("gameId", "platformId", "teamId")
#   )
# ) %>%
#   select(
#     gameId,
#     platformId,
#     teamId = teamId,
#     patch = patch.x,
#     gameDuration = gameDuration.x,
#     ad.Id = championId.x,
#     sup.Id = championId.y,
#     win = win.x,
#     ad.DamageToChampions= totalDamageDealtToChampions.x,
#     sup.DamageToChampions = totalDamageDealtToChampions.y,
#     ad.kills = kills.x,
#     ad.assists = assists.x,
#     ad.deaths = deaths.x,
#     sup.kills = kills.y,
#     sup.assists = assists.y,
#     sup.deaths = deaths.y
#   ) %>%
#   mutate(
#     ad.kda = (ad.kills+ad.assists)/ifelse(ad.deaths ==0,1,ad.deaths)
#   ) %>%
#   inner_join(select(champLookUp, championId, ad=name), by=c("ad.Id" = "championId"))  %>%
#   inner_join(select(champLookUp, championId, sup=name), by=c("sup.Id" = "championId"))


# 
# ### TOP Laner
# #top <- playerstats[lane == "TOP"]
# top <- data.table(dbGetQuery(conn = connection, "SELECT * FROM playerdetails WHERE lane = 'TOP'"))
# top.distribution <- merge(
#                       data.table(top[,list(gamesPlayed = .N), by = championId][order(by = gamesPlayed, decreasing = T)]),
#                       champLookUp,
#                       by= "championId"
#                       )
# top.distribution.platform <- merge(
#   data.table(top[,list(gamesPlayed = .N), by = list(championId,platformId)][order(by = gamesPlayed, decreasing = T)]),
#   champLookUp,
#   by= "championId"
# )
# top.distribution.patch <- merge(
#   data.table(top[,list(gamesPlayed = .N), by = list(championId,patch)][order(by = gamesPlayed, decreasing = T)]),
#   champLookUp,
#   by= "championId"
# )
# setnames(top.distribution,"name", "names")
# 
# relchamps.top <- top[,list(gamesPlayed = .N), by = championId][order(by = gamesPlayed, decreasing = T)][,.SD[1:20]][,championId]
# top.relevant <- top[championId %in% relchamps.top]
# top.performance <- champPerformancePatchRegion[lane=="TOP" & role == "SOLO" & championId %in% unique(top.relevant$championId)]
# 
# 
# 
# ### Jungler
# jungle <- playerstats[lane == "JUNGLE"]
# relchamps.jungle <- jungle[,list(gamesPlayed = .N), by = championId][order(by = gamesPlayed, decreasing = T)][,.SD[1:20]][,championId]
# jungle.relevant <- jungle[championId %in% relchamps.jungle]
# jungle.performance <- champPerformancePatchRegion[lane=="JUNGLE" & championId %in% unique(jungle.relevant$championId)]
# 
# 
# jungle.distribution <- merge(
#   data.table(jungle[,list(gamesPlayed = .N), by = championId][order(by = gamesPlayed, decreasing = T)]),
#   champLookUp,
#   by= "championId"
# )
# jungle.distribution.platform <- merge(
#   data.table(jungle[,list(gamesPlayed = .N), by = list(championId,platformId)][order(by = gamesPlayed, decreasing = T)]),
#   champLookUp,
#   by= "championId"
# )
# jungle.distribution.patch <- merge(
#   data.table(jungle[,list(gamesPlayed = .N), by = list(championId,patch)][order(by = gamesPlayed, decreasing = T)]),
#   champLookUp,
#   by= "championId"
# )
# setnames(jungle.distribution,"name", "names")
# 
# 
# ### Mid laner
# mid <- playerstats[lane == "MIDDLE"]
# relchamps.mid <- mid[,list(gamesPlayed = .N), by = championId][order(by = gamesPlayed, decreasing = T)][,.SD[1:20]][,championId]
# mid.relevant <- mid[championId %in% relchamps.mid]
# mid.performance <- champPerformancePatchRegion[lane=="MIDDLE" & role == "SOLO" & championId %in% unique(mid.relevant$championId)]
# 
# mid.distribution <- merge(
#   data.table(mid[,list(gamesPlayed = .N), by = championId][order(by = gamesPlayed, decreasing = T)]),
#   champLookUp,
#   by= "championId"
# )
# mid.distribution.platform <- merge(
#   data.table(mid[,list(gamesPlayed = .N), by = list(championId,platformId)][order(by = gamesPlayed, decreasing = T)]),
#   champLookUp,
#   by= "championId"
# )
# mid.distribution.patch <- merge(
#   data.table(mid[,list(gamesPlayed = .N), by = list(championId,patch)][order(by = gamesPlayed, decreasing = T)]),
#   champLookUp,
#   by= "championId"
# )
# setnames(mid.distribution,"name", "names")
# 
