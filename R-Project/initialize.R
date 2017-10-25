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
#require(tibble)
library(rgl)

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


connection <-
  dbConnect(RMySQL::MySQL(), user = "root", password = "Milch4321", "leaguestats")

#Soloqueue
playerstats <- data.table(dbReadTable(connection, "playerdetails"))
teamstats <- data.table(dbReadTable(connection, "teamdetails"))
playerstats <- 
  merge(
    playerstats,
    teamstats,
    by.x = c("gameId", "platformId", "teamId"),
    by.y = c("gameId", "platformId", "teamId")
  )

champPerformancePatch <- data.table(dbReadTable(connection, "champPerformancePatch"))
champPerformancePatchRegion <- data.table(dbReadTable(connection, "champPerformancePatchRegion"))
champPerformancePatchByWin <- data.table(dbReadTable(connection, "champPerformancePatchWin"))

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

xpSteps = c(280,380,480,580,680,780,880,980,1080,1180,1280,1380,1480,1580,1680,1780,1880)

## Items
itemLookUp <- data.table(dbReadTable(connection, "itemkeys"))
setkey(itemLookUp, "id")

item <- itemLookUp$name
names(item) <- itemLookUp$champId


## Champions
champLookUp <- data.table(dbReadTable(connection, "championkeys"))
setkey(champLookUp, "championId")

champ <- champLookUp$name
names(champ) <- champLookUp$champId


### TOP Laner
top <- playerstats[lane == "TOP"]
top.distribution <- merge(
                      data.table(top[,list(gamesPlayed = .N), by = championId][order(by = gamesPlayed, decreasing = T)]),
                      champLookUp,
                      by= "championId"
                      )
top.distribution.platform <- merge(
  data.table(top[,list(gamesPlayed = .N), by = list(championId,platformId)][order(by = gamesPlayed, decreasing = T)]),
  champLookUp,
  by= "championId"
)
top.distribution.patch <- merge(
  data.table(top[,list(gamesPlayed = .N), by = list(championId,patch)][order(by = gamesPlayed, decreasing = T)]),
  champLookUp,
  by= "championId"
)
setnames(top.distribution,"name", "names")

relchamps.top <- top[,list(gamesPlayed = .N), by = championId][order(by = gamesPlayed, decreasing = T)][,.SD[1:20]][,championId]
top.relevant <- top[championId %in% relchamps.top]
top.performance <- champPerformancePatchRegion[lane=="TOP" & role == "SOLO" & championId %in% unique(top.relevant$championId)]



### Jungler
jungle <- playerstats[lane == "JUNGLE"]
relchamps.jungle <- jungle[,list(gamesPlayed = .N), by = championId][order(by = gamesPlayed, decreasing = T)][,.SD[1:20]][,championId]
jungle.relevant <- jungle[championId %in% relchamps.jungle]
jungle.performance <- champPerformancePatchRegion[lane=="JUNGLE" & championId %in% unique(jungle.relevant$championId)]


jungle.distribution <- merge(
  data.table(jungle[,list(gamesPlayed = .N), by = championId][order(by = gamesPlayed, decreasing = T)]),
  champLookUp,
  by= "championId"
)
jungle.distribution.platform <- merge(
  data.table(jungle[,list(gamesPlayed = .N), by = list(championId,platformId)][order(by = gamesPlayed, decreasing = T)]),
  champLookUp,
  by= "championId"
)
jungle.distribution.patch <- merge(
  data.table(jungle[,list(gamesPlayed = .N), by = list(championId,patch)][order(by = gamesPlayed, decreasing = T)]),
  champLookUp,
  by= "championId"
)
setnames(jungle.distribution,"name", "names")


### Mid laner
mid <- playerstats[lane == "MIDDLE"]
relchamps.mid <- mid[,list(gamesPlayed = .N), by = championId][order(by = gamesPlayed, decreasing = T)][,.SD[1:20]][,championId]
mid.relevant <- mid[championId %in% relchamps.mid]
mid.performance <- champPerformancePatchRegion[lane=="MIDDLE" & role == "SOLO" & championId %in% unique(mid.relevant$championId)]

mid.distribution <- merge(
  data.table(mid[,list(gamesPlayed = .N), by = championId][order(by = gamesPlayed, decreasing = T)]),
  champLookUp,
  by= "championId"
)
mid.distribution.platform <- merge(
  data.table(mid[,list(gamesPlayed = .N), by = list(championId,platformId)][order(by = gamesPlayed, decreasing = T)]),
  champLookUp,
  by= "championId"
)
mid.distribution.patch <- merge(
  data.table(mid[,list(gamesPlayed = .N), by = list(championId,patch)][order(by = gamesPlayed, decreasing = T)]),
  champLookUp,
  by= "championId"
)
setnames(mid.distribution,"name", "names")



### AD CARRIES
adc <- playerstats[lane == "BOTTOM" & role == "DUO_CARRY"] 
adc.Pro <- proplayerstats[lane == "BOTTOM" & role == "DUO_CARRY"] 
relchamps.adc <- adc[,list(gamesPlayed = .N), by = championId][order(by = gamesPlayed, decreasing = T)][,.SD[1:16]][,championId]

adc.relevant <- adc[championId %in% relchamps.adc]
adc.performance <- champPerformancePatchRegion[lane=="BOTTOM" & role == "DUO_CARRY" & championId %in% unique(adc.relevant$championId)]
adc.performance.patch <- champPerformancePatch[lane == "BOTTOM" & role == "DUO_CARRY" & championId %in% unique(adc.relevant$championId)]
adc.performance.patch.win <- champPerformancePatchByWin[lane == "BOTTOM" & role == "DUO_CARRY" & championId %in% unique(adc.relevant$championId)]

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


relItems.ADC = c("Infinity Edge",
                 "Essence Reaver",
                 "Manamune/Muramana",
                 "Guinsoo's Rageblade",
                 "Blade of the Ruined King",
                 "The Bloodthirster",
                 "Edge of Night",
                 "Hexdrinker",
                 "Wit's End",
                 "The Black Cleaver",
                 "Duskblade of Draktharr",
                 "Youmuu's Ghostblade",
                 "Trinity Force",
                 "Runaan's Hurricane",
                 "Rapid Firecannon",
                 "Statikk Shiv",
                 "Phantom Dancer",
                 "Iceborn Gauntlet")

### support
sup <- playerstats[lane == "BOTTOM" & role == "DUO_SUPPORT"]
relchamps.sup <- sup[,list(gamesPlayed = .N), by = championId][order(by = gamesPlayed, decreasing = T)][,.SD[1:20]][,championId]
sup.relevant <- sup[championId %in% relchamps.sup]
sup.performance <- champPerformancePatchRegion[lane=="BOTTOM" & role == "DUO_SUPPORT" & championId %in% unique(sup.relevant$championId)]

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
  sup[championId %in% relchamps.sup,list(championId, platformId, patch, item=item0)], 
  sup[championId %in% relchamps.sup,list(championId, platformId, patch, item=item1)],
  sup[championId %in% relchamps.sup,list(championId, platformId, patch, item=item2)],
  sup[championId %in% relchamps.sup,list(championId, platformId, patch, item=item3)],
  sup[championId %in% relchamps.sup,list(championId, platformId, patch, item=item4)],
  sup[championId %in% relchamps.sup,list(championId, platformId, patch, item=item5)],
  sup[championId %in% relchamps.sup,list(championId, platformId, patch, item=item6)]
)

#botlane adc+sup: 
botlane = data.table(
  merge(adc.relevant, 
        sup.relevant, 
        by = c("gameId", "platformId", "teamId")
        )
  ) %>%
  select(
      gameId,
      platformId, 
      teamId = teamId, 
      patch = patch.x,
      gameDuration = gameDuration.x,
      ad.Id = championId.x,
      sup.Id = championId.y,
      win = win.x,
      ad.DamageToChampions= totalDamageDealtToChampions.x,
      sup.DamageToChampions = totalDamageDealtToChampions.y,
      ad.kills = kills.x,
      ad.assists = assists.x,
      ad.deaths = deaths.x,
      sup.kills = kills.y,
      sup.assists = assists.y,
      sup.deaths = deaths.y
    ) %>%
  mutate(
    ad.kda = (ad.kills+ad.assists)/ifelse(ad.deaths ==0,1,ad.deaths)
  ) %>%
  inner_join(select(champLookUp, championId, ad=name), by=c("ad.Id" = "championId"))  %>%
  inner_join(select(champLookUp, championId, sup=name), by=c("sup.Id" = "championId"))