source("initialize.R")
############################ ADC ####################################

####DISTRIBUTIONS OF CHAMPIONS
#overall distribution in season 7
p <- ggplot(data=adc.distribution,aes(x=names, y=gamesPlayed/1000)) + 
  geom_bar(stat='identity', fill="#56B4E9") +
  scale_x_discrete(limits=adc.distribution[order(by=gamesPlayed,decreasing = T)][1:30]$names) +
  labs(x = "Champion", y = "#matches") +
  theme_bw() + 
  theme(axis.text.x = element_text(size=8,angle = 45, hjust = 1), 
        axis.text.y = element_text(size=8),
        axis.title = element_text(size=12)) 
p



#Barchart 
p <- ggplot(data = adc.performance, aes(x=name, y=games/10000 *100, fill = name)) + 
  geom_bar(stat="Identity") + 
  facet_wrap( ~ platformId+factor(patch, levels = patchOrder), nrow=4) + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1), axis.text.y = element_text(size=6)) + 
  labs(x = "Champion", y = "Playrate in Percentages") +
  coord_flip() + 
  guides(fill=F)
p+ ggtitle("ADC Picks per Patch and Region")
p

#Linechart
p <- ggplot(data = adc.performance[championId %in% relchamps.adc], aes(x = patch, y=games/10000 * 100, group=platformId, color=platformId)) + 
  geom_line(linetype = 1, size=1) +  
  scale_x_discrete(limits=patchOrder) +
  ylim(c(0,60)) +
  theme_bw() +
  theme(axis.text.x = element_text(size=8, angle=90),
        legend.position = "bottom")+
  labs(x = "Patch", y = "Playrate in Percentage", color="Region") +
  facet_wrap(~ name, ncol = 4) 
p

#Linechart
p <- ggplot(data = adc.performance.patch[championId %in% relchamps.adc], aes(x = patch, y=winrate * 100,group=name)) + 
  geom_line(linetype = 1) + 
  #geom_line(data = adc.performance , aes(y = winrate), linetype = 2) + 
  scale_x_discrete(limits=patchOrder) +
  theme_igray() +
  theme(axis.text.x = element_text(size=5)) +
  labs(x = "Patch", y = "Winrate in Percentage") +
  facet_wrap(~ name, ncol = 4) 
p + ggtitle("ADC Picks per Patch and Region")


#Linechart
p <- ggplot(data = adc.performance[championId %in% relchamps.adc], aes(x = patch, y=gamesBanned/10000 * 100, group=platformId, color=platformId)) + 
  geom_line(linetype = 1) + 
  #geom_line(data = adc.performance , aes(y = winrate), linetype = 2) + 
  scale_x_discrete(limits=patchOrder) +
  theme_igray() +
  theme(axis.text.x = element_text(size=5)) +
  labs(x = "Patch", y = "Playrate in Percentage") +
  facet_wrap(~ name, ncol = 4) 
p + ggtitle("ADC Picks per Patch and Region")

#Linechart
p <- ggplot(data = adc.performance[championId %in% relchamps.adc], aes(x = patch, y=(gamesBanned+games)/10000 * 100, group=platformId, color=platformId)) + 
  geom_line(linetype = 1) + 
  #geom_line(data = adc.performance , aes(y = winrate), linetype = 2) + 
  scale_x_discrete(limits=patchOrder) +
  theme_igray() +
  theme(axis.text.x = element_text(size=5)) +
  labs(x = "Patch", y = "Playrate in Percentage") +
  facet_wrap(~ name, ncol = 4) 
p + ggtitle("ADC Picks per Patch and Region")



p <- ggplot(data = adc.performance[championId %in% relchamps.adc], aes(x = patch, y=games/40000 * 100, group=platformId, fill=platformId)) + 
  geom_area() + 
  #geom_line(data = adc.performance , aes(y = winrate), linetype = 2) + 
  scale_x_discrete(limits=patchOrder) +
  theme_igray() +
  theme(axis.text.x = element_text(size=5)) +
  labs(x = "Patch", y = "Playrate in Percentage") +
  facet_wrap(~ name, ncol = 4) 
p + ggtitle("ADC Picks per Patch and Region")


p <- ggplot(data = adc.relevant.patchTier, aes(x = patch, y=playRate, group=order, color=order)) + 
  geom_line(size=1.2) + 
  scale_x_discrete(limits=patchOrder) +
  theme(axis.text.x = element_text(size=5)) +
  labs(x = "Patch", y = "PlayRate") +
  scale_color_manual(values = tier_palette) + 
  # scale_fill_brewer(palette="Set1") +
  facet_wrap(~ name, ncol = 4) 
p + ggtitle("ADC Picks per Patch and Tier")
p

p <- ggplot(data = adc.relevant.patchTier[tier!="UNRANKED"], aes(x = patch, y=playRate*100, group=order, fill=order)) + 
  geom_area() + 
  scale_x_discrete(limits=patchOrder) +
  theme(axis.text.x = element_text(size=5)) +
  labs(x = "Patch", y = "PlayRate") +
  scale_fill_manual(values = tier_palette) + 
  theme_bw() +
  theme(axis.text.x = element_text(size=8, angle=90),
        legend.position = "bottom")+
  labs(x = "Patch", y = "Playrate in Percentage", color="Region") +
  facet_wrap(~ name, ncol = 4) +
  guides(fill=guide_legend(title="Skill Tier",nrow=1,byrow=TRUE))
p


p <- list()
for(i in 1:length(patchOrder)){
  df= adc.distribution.patch[patch==patchOrder[i]][order(by=gamesPlayed,decreasing = T)][1:20]
  p[[i]] <- ggplot(df, aes(x=name, y=gamesPlayed)) +
    geom_bar(stat = "identity",fill="#56B4E9") +
    scale_x_discrete(limits=df[order(by=gamesPlayed,decreasing = T)]$name) +
    scale_y_continuous(limits=c(0,20000)) +
    theme(axis.text.x = element_text(angle = 90, hjust = 1), axis.text.y = element_text(size=6)) + 
    ggtitle("Patch",paste0(patchOrder[i]))
}
do.call("grid.arrange", c(p, ncol= 5))

adc.set1 <- c("Ashe","Caitlyn", "Draven", "Ezreal", "Kog'Maw", "Lucian", "Jhin", "Tristana", "Vayne", "Varus", "Xayah")
adc.set2 <- c("Kog'Maw", "Lucian", "Jhin")
adc.set3 <- c("Tristana", "Vayne", "Varus", "Xayah")
adc.set4 <- c("Ashe", "Jhin", "Varus")
adc.set <- adc.set1

df <- items.adc[championId %in% relchamps.adc][, list(count=.N), by = c("championId","championName", "patch", "itemName")]
setkeyv(df, c("championName", "patch"))
championCount <- items.adc[,list(championName, patch)][,list(gamesPlayed = .N), by = c("championName", "patch")]
setkeyv(championCount, c("championName", "patch"))
df <- merge(df, championCount, by= c("championName","patch"))
df$perc <- df$count/df$gamesPlayed

df[itemName %in% relItems.ADC] %>% 
  group_by(championName, patch) %>%
  summarise(purchase_per_patch = sum(count)) -> df_2
df <- merge(df, df_2, by=c("championName", "patch"))
df$perc_rel_items = df$count/df$purchase_per_patch

p <- ggplot(data = df[championId %in% relchamps.adc[0:4] & itemName %in% relItems.ADC]) + 
  geom_bar(stat= "identity", aes(x=itemName, y=perc, fill= itemName)) + 
  facet_grid(championName ~ factor(patch, levels = patchOrder), scale="free") + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1, size = 6), 
        axis.text.y = element_text(size=6),
        legend.position="None") + 
  # scale_x_discrete(limits=relItems.ADC) +
  coord_flip() 
p


p <- ggplot(data = df[itemName %in% relItems.ADC & championId %in% relchamps.adc[9:16]]) + 
  geom_bar(stat= "identity", aes(x=factor(1), y=perc_rel_items, fill= factor(itemName, levels=relItems.ADC)), width = 1) + 
  coord_polar(theta = "y", start =0) + 
  facet_grid(championName ~ factor(patch, levels = patchOrder)) + 
  ylab('')+ xlab('') + 
  labs(fill = "Item Name") +
  theme(axis.ticks.y = element_blank(),
        axis.text.y = element_blank(),
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        legend.position = "bottom") +
  scale_fill_manual(values = items.palette) +
  guides(fill=guide_legend(nrow=3,byrow=TRUE))
p


p <- ggplot(data = df, aes(x=itemName, y=perc, fill=championName, group=championName)) + 
  geom_bar(stat= "identity") +  
  facet_grid(championName~factor(patch, levels = patchOrder)) + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1, size = 6), axis.text.y = element_text(size=6)) + 
  scale_x_discrete(limits=relItems.ADC) +
  coord_flip() +
  guides(fill=FALSE) +
  scale_fill_manual(values = c("Varus" = "#9370DB", "Jhin" = "#c0c0c0", "Ashe" = "lightblue"))
p

#nur ne Idee, aber da erkennt leider keiner etwas.
p <- ggplot(data = df[itemName %in% relItems.ADC],aes(x=factor(patch, levels = patchOrder), fill = factor(itemName, levels=relItems.ADC))) + 
  geom_bar(width=0.9, position="fill", color="black") +
  #geom_text(aes(y=Freq , label=Freq), vjust=0) +
  facet_grid(. ~ name) +
  theme_igray() + scale_fill_manual(values = colorRampPalette(brewer.pal(8, "Accent"))(length(relItems.ADC)),
                                    guide = guide_legend(nrow=2)) +
  theme(legend.position="bottom")
p + coord_polar(theta = "y")


##items only
df %>%
  group_by(itemName, patch) %>%
  summarise(count = sum(count)) %>%
  filter(itemName %in% relItems.ADC) -> df2

p <- ggplot(data = df2, aes(x=itemName,y = count, fill=itemName)) +
  geom_bar(stat='Identity') +
  coord_flip() +
  theme(legend.position = "None") +
  facet_wrap(~ factor(patch, levels = patchOrder))
p  

### attempt to illustrate specific traits of adcs
# OVERALL
dfprep = adc.performance.patch %>%
  mutate(DPS = totalDamageToChampions/gameDuration) %>%
  mutate(patch2 = 
           dplyr::case_when(patch %in% c("6.23", "6.24") ~ "Pre-Season",
                            patch %in% c("7.1", "7.2", "7.3") ~ "7.1-7.3",
                            patch %in% c("7.4", "7.5", "7.6", "7.7", "7.8") ~ "7.4-7.8",
                            patch %in% c("7.9", "7.10", "7.11", "7.12", "7.13") ~ "7.9-7.13",
                            patch %in% c("7.14", "7.15", "7.16", "7.17", "7.18") ~ "7.14-7.18",
                            TRUE                     ~ "undefined")) %>%
  group_by(name,patch2) %>%
  summarise(
    games = sum(games),
    #summoners,
    #winrate = mean(winrate),
    #DPS,
    DmgD = mean(totalDamageToChampions),
    k = mean(kills),
    a = mean(assists), 
    d = mean(deaths),
    DmgT = mean(totalDamageTaken),
    cs = mean(csPerGame),
    gold = mean(goldEarned)
  )
dfprep = data.table(dfprep)

df = dfprep %>%
  rownames_to_column( var = "champ" ) %>%
  mutate_each(funs(rescale), -c(champ,name,patch2)) %>%
  melt(id.vars=c('champ','name','patch2'), measure.vars=colnames(dfprep[,-c("name","patch2")])) %>%
  arrange(champ)
df = data.table(df)

df = df %>%
  merge(dfprep[,list(name, patch2, games)],
        by = c("name", "patch2")
        )

adc.set1 = c("Jhin", "Miss Fortune", "Varus", "Lucian", "Twitch", "Tristana")
adc.set2 = c("Miss Fortune", "Sivir", "Tristana", "Twitch", "Twitch", "Varus", "Vayne", "Xayah")
#radar charts: better filter out some champs
radarChart <- function(adc.set) {
  p <- df[name %in% adc.set1] %>%
        ggplot(aes(x=variable, y=value, group=name, color=name, alpha=games)) + 
        geom_polygon() +
        coord_radar() + facet_grid(factor(patch2, levels=c("Pre-Season", "7.1-7.3","7.4-7.8", "7.9-7.13","7.14-7.18"))~name) + 
        scale_x_discrete(limits=unique(df[variable!="games"]$variable)) + 
        theme_bw() +
        theme(legend.position="none", 
              axis.text.x = element_text(size = 8), 
              axis.ticks.x = element_line(size = 8),
              axis.ticks = element_blank(),
              axis.text.y = element_blank(),
              axis.title.x = element_blank(),
              axis.title.y = element_blank())
  return(p)
}
radarChart()



#bar chart perspective
df %>%
  ggplot(aes(x=variable, y=value, group= name, fill = name, alpha=games)) + 
  geom_bar(stat="identity") + 
  geom_line(y = 0.5, linetype  =2, color = "black") +
  facet_grid(factor(patch, levels=patchOrder)~name) +
  coord_flip() +
  theme_igray() + scale_fill_manual(values = colorRampPalette(brewer.pal(8, "Accent"))(length(unique(df$name)))) +
  theme(axis.text.y = element_text(size = 5), legend.position="none")

#line chart perspective
df %>%
  ggplot(aes(y=value,x=factor(patch, levels=patchOrder), group=variable, colour=variable)) + 
  geom_line(stat="identity") + 
  facet_wrap(~name, ncol=4) +
  theme_igray() +
  theme(axis.text.y = element_text(size = 5))


dfprep %>%
  mutate_each(funs(rescale), -c(name, patch)) %>% data.table() -> df_radar

p <- list()
for(j in 1:length(patchOrder))
  p[[j]] <- ggRadar(df_radar[name %in% adc.set4 & patch == patchOrder[j]], aes(color=name), rescale = F, ylim=c(0,1))
do.call("grid.arrange", c(p, nrow = 4 ))

## attempt to scale into 3d and illustrate
adc.performance.patch %>%
  rownames_to_column(var="id") -> adc.cmd 

adc.cmd.numerics = adc.cmd[,-c("name", "championId", "lane", "role", "patch")]
setkey(adc.cmd.numerics, "id")
rownames(adc.cmd.numerics) = adc.cmd.numerics$id
adc.cmd.dist <- dist(adc.cmd.numerics[,-1])
colnames(adc.cmd.dist) = rownames

fit <- cmdscale(adc.cmd.dist, k = 3)
data.table(fit) %>%
  rownames_to_column(var = "id") %>%
  merge(adc.cmd[,c("id","name", "patch")], by="id") -> fit2 
fit2$detailedName = paste0(fit2$name, " ", fit2$patch)
kmeans3 = kmeans(x = fit2[,2:4], centers = 3)
kmeans4 = kmeans(x = fit2[,2:4], centers = 4)
kmeans5 = kmeans(x = fit2[,2:4], centers = 5)
kmeans6 = kmeans(x = fit2[,2:4], centers = 6)

fit2$cluster3 = kmeans3$cluster
fit2$cluster4 = kmeans4$cluster
fit2$cluster5 = kmeans5$cluster
fit2$cluster6 = kmeans6$cluster

plot3d(fit2[,2:4], size = 10, col = fit2$cluster6)
text3d(fit2[,2:4], texts = fit2$detailedName, size=2)

fit2[,c("name", "cluster6")]

## GAME DURATION, DPS AND DAMAGE DONE

# duration of games where champions participated
adc.performance.patch[championId %in% relchamps.adc] %>%
  ggplot(aes(y = gameDuration/60, x = patch, group=name)) + 
  geom_line() + 
  scale_x_discrete(limits=patchOrder) + 
  facet_wrap(~name, ncol = 4)


# damage done to champions
adc.performance.patch[championId %in% relchamps.adc] %>%
  ggplot(aes(y = totalDamageToChampions, x = patch, group=name)) + 
  geom_line() + 
  scale_x_discrete(limits=patchOrder) + 
  facet_wrap(~name, ncol = 4)

adc.performance.patch[championId %in% relchamps.adc] %>%
  ggplot(aes(y = totalDamageToChampions, x = patch, group=name)) + 
  geom_line() + 
  scale_x_discrete(limits=patchOrder) + 
  facet_wrap(~name, ncol = 4)



# dps
adc.performance.patch[championId %in% relchamps.adc] %>%
  ggplot(aes(y = totalDamageToChampions/(gameDuration/60), x = patch, group=name)) + 
  geom_line() + 
  scale_x_discrete(limits=patchOrder) + 
  facet_wrap(~name, ncol = 4)

# dps by win/loss
df = adc.performance.patch.win[championId %in% relchamps.adc]
ggplot(df,aes(y = totalDamageToChampions/(gameDuration/60), x = patch, group=as.factor(win), color = as.factor(win))) + 
  geom_line() + 
  scale_x_discrete(limits=patchOrder) + 
  facet_wrap(~name, ncol = 4)

## gold earned and gold per min

adc.performance.patch[championId %in% relchamps.adc] %>%
  ggplot(aes(y = goldEarned, x = patch, group=name)) + 
  geom_line() + 
  scale_x_discrete(limits=patchOrder) + 
  facet_wrap(~name, ncol = 4)

adc.performance.patch[championId %in% relchamps.adc] %>%
  ggplot(aes(y = goldEarned/(gameDuration/60), x = patch, group=name)) + 
  geom_line() + 
  scale_x_discrete(limits=patchOrder) + 
  facet_wrap(~name, ncol = 4)

# win and loss
df = adc.performance.patch.win[championId %in% relchamps.adc]
ggplot(df,aes(y = goldEarned/(gameDuration/60), x = patch, group=as.factor(win), color = as.factor(win))) + 
  geom_line() + 
  scale_x_discrete(limits=patchOrder) + 
  facet_wrap(~name, ncol = 4)

### cs and xp development

#cs diffs
df = adc.performance.patch[championId %in% relchamps.adc, list(name, patch, csDiffPerMinTen, csDiffPerMinTwenty, csDiffPerMinThirty)]  
melt(df, id.vars=c('name','patch'), measure.vars=colnames(df[,-c("name","patch")])) %>% 
  ggplot(aes(y = value, x = variable, group = name)) + 
  geom_line() + scale_x_discrete(labels=c("10","20","30")) + 
  facet_grid(name~factor(patch, levels = patchOrder)) + 
  ggtitle("Creep Score Differential for each Champion and Patch")


#xp Diffs
df = adc.performance.patch[championId %in% relchamps.adc, list(name, patch, xpDiffPerMinTen, xpDiffPerMinTwenty, xpDiffPerMinThirty)]  
melt(df, id.vars=c('name','patch'), measure.vars=colnames(df[,-c("name","patch")])) %>% 
  ggplot(aes(y = value, x = variable, group = name)) + 
  geom_line() + scale_x_discrete(labels=c("10","20","30")) + 
  facet_grid(name~factor(patch, levels = patchOrder)) + 
  ggtitle("Experience Differential for each Champion and Patch")

# cs deltas
df = adc.performance.patch[championId %in% relchamps.adc, list(name, patch, csPerMinDeltaTen, csPerMinDeltaTwenty, csPerMinDeltaThirty)]  
melt(df, id.vars=c('name','patch'), measure.vars=colnames(df[,-c("name","patch")])) %>% 
  ggplot(aes(y = value, x = variable, group = name)) + 
  geom_line() + scale_x_discrete(labels=c("0-10","10-20","20-30")) + 
  facet_grid(name~factor(patch, levels = patchOrder)) + 
  ggtitle("Creep Score Per Minute Delta for each Champion and Patch")





## first blood

adc.performance.patch[championId %in% relchamps.adc] %>%
  ggplot(aes(y = firstBloodKill + firstBloodAssist, x = patch, group=name)) + 
  geom_line() + 
  scale_x_discrete(limits=patchOrder) + 
  facet_wrap(~name, ncol = 4)


adc.performance.patch.win[championId %in% relchamps.adc] %>%
  ggplot(aes(y = firstBloodKill + firstBloodAssist, x = patch, group=as.factor(win), color=as.factor(win))) + 
  geom_line() + 
  scale_x_discrete(limits=patchOrder) + 
  facet_wrap(~name, ncol = 4)



# first tower

adc.performance.patch[championId %in% relchamps.adc] %>%
  ggplot(aes(y = firstTowerKill + firstTowerAssist, x = patch, group=name)) + 
  geom_line() + 
  scale_x_discrete(limits=patchOrder) + 
  facet_wrap(~name, ncol = 4)


#first inhib

adc.performance.patch[championId %in% relchamps.adc] %>%
  ggplot(aes(y = firstInhibitorKill + firstInhibitorAssist, x = patch, group=name)) + 
  geom_line() + 
  scale_x_discrete(limits=patchOrder) + 
  facet_wrap(~name, ncol = 4)


## winrate by game duration 
df = adc.performance.patch[,list(name, patch, winrate, winrateAt25, winrateAt30, winrateAt35, winrateAt40, winrateAt45, winrateOver45)]
df1 = df %>%
  rownames_to_column(var = "champ") %>%
  melt(id.vars=c('champ', 'name', 'patch'), measure.vars = colnames(df[,-c("name", "patch")]))

df1$variableNew <- c(0,25,30,35,40,45,50)[match(df1$variable, c("winrate", "winrateAt25", "winrateAt30", "winrateAt35", "winrateAt40","winrateAt45", "winrateOver45"))]
winrate_scale = c("winrateAt25", "winrateAt30", "winrateAt35", "winrateAt40","winrateAt45", "winrateOver45")
p = ggplot(data = df1[variableNew!=0], aes(x = variableNew, y=value, color = name)) + 
  geom_point() + 
  geom_smooth(method="auto") + 
  facet_grid(name ~ factor(patch, levels = patchOrder)) + 
  scale_y_continuous(limits=c(0,1)) + 
  scale_x_continuous(limits = c(20,55), breaks = seq(25,50,5), labels=c( "25", "30", "35" ,"40", "45", ">45")) + 
  theme(legend.position="none")
p



############################ SUP ####################################
#overall distribution in season 7
p <- ggplot(data=sup.distribution,aes(x=names, y=gamesPlayed)) + 
  geom_bar(stat='identity') +
  scale_x_discrete(limits=sup.distribution[order(by=gamesPlayed,decreasing = T)]$names) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1), axis.text.y = element_text(size=6))
p

p <- ggplot(data=sup.distribution,aes(x=names, y=gamesPlayed)) + 
  geom_bar(stat='identity', fill="#56B4E9") +
  scale_x_discrete(limits=sup.distribution[order(by=gamesPlayed,decreasing = T)][1:30]$names) +
  labs(x = "Champion", y = "#matches", title="Distribution of Top 30 Supports played in Season 7") +
  theme_bw() + 
  theme(axis.text.x = element_text(size=16,angle = 90, hjust = 1), 
        axis.text.y = element_text(size=6),
        title = element_text(size=40),
        axis.title = element_text(size=24)) 
p



#Barchart 
p <- ggplot(data = sup.performance, aes(x=name, y=games/10000 *100, fill = name)) + 
  geom_bar(stat="Identity") + 
  facet_wrap( ~ platformId+factor(patch, levels = patchOrder), nrow=4) + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1), axis.text.y = element_text(size=6)) + 
  labs(x = "Champion", y = "Playrate in Percentages") +
  coord_flip() + 
  guides(fill=F)
p+ ggtitle("Support Picks per Patch and Region")
p

#Linechart
p <- ggplot(data = sup.performance[championId %in% relchamps.sup], aes(x = patch, y=games/10000 * 100, group=platformId, color=platformId)) + 
  geom_line(linetype = 1) + 
  scale_x_discrete(limits=patchOrder) +
  theme(axis.text.x = element_text(size=5)) +
  labs(x = "Patch", y = "Playrate in Percentage") +
  facet_wrap(~ name, ncol = 4) 
p + ggtitle("Support Picks per Patch and Region")

#Linechart
p <- ggplot(data = sup.performance[name %in% c("Malzahar", "Miss Fortune", "Karma", "Zyra", "Janna", "Lulu")], aes(x = patch, y=games/10000 * 100, group=platformId, color=platformId)) + 
  geom_line(linetype = 1) + 
  scale_x_discrete(limits=patchOrder) +
  theme(axis.text.x = element_text(size=5)) +
  labs(x = "Patch", y = "Playrate in Percentage") +
  facet_wrap(~ name, ncol = 3) 
p + ggtitle("Support Picks per Patch and Region")


#botlane sup and adc combined:

botlane %>%
  filter(sup.Id %in% relchamps.sup) %>%
  group_by(ad,sup,patch) %>%
  summarise(count =n()) %>%
  left_join(botlane %>% group_by(ad, patch) %>% summarise(adCount = n())) %>% 
  mutate(perc = count/adCount) %>%
  ggplot(aes(x = sup, y = perc)) + geom_bar(stat="identity") + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1), axis.text.y = element_text(size=6)) + 
  coord_flip() +
  facet_grid(ad~factor(patch, levels=patchOrder))

###############################################################################
###############################################################################
############################ PRO GAMES #######################################
###############################################################################
###############################################################################

###adc


####DISTRIBUTIONS OF CHAMPIONS

#Barchart 
majorRegions =  c("TRLH1", #naLCS
                  "TRLH3", #euLCS
                  "TRTW", #lms
                  "ESPORTSTMNT06", #lck
                  "ESPORTSTMNT03" #cbLoL and TCL
)
p <- ggplot(data = adc.pro.performance[platformId %in% majorRegions], aes(x=name, y=playrate *100, fill = name)) + 
  geom_bar(stat="Identity") + 
  facet_wrap( ~ platformId+factor(patch, levels = patchOrder), nrow=5) + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1), axis.text.y = element_text(size=6)) + 
  labs(x = "Champion", y = "Playrate in Percentages") +
  coord_flip() + 
  guides(fill=F)
p + ggtitle("ADC Picks per Patch and Region")

#Linechart
p <- ggplot(data = adc.pro.performance[platformId %in% majorRegions], aes(x = patch, y=playrate * 100, group=platformId, color=platformId)) + 
  geom_line(linetype = 1) + 
  #geom_line(data = adc.performance , aes(y = winrate), linetype = 2) + 
  scale_x_discrete(limits=patchOrder) +
  theme(axis.text.x = element_text(size=5)) +
  labs(x = "Patch", y = "Playrate in Percentage") +
  facet_wrap(~ name, ncol = 4) 
p + ggtitle("ADC Picks per Patch and Region")


p <- list()
for(i in 1:length(patchOrder)){
  df= adc.distribution.patch[patch==patchOrder[i]][order(by=gamesPlayed,decreasing = T)][1:20]
  p[[i]] <- ggplot(df, aes(x=name, y=gamesPlayed)) +
    geom_bar(stat = "identity") +
    scale_x_discrete(limits=df[order(by=gamesPlayed,decreasing = T)]$name) +
    scale_y_continuous(limits=c(0,5000)) +
    theme(axis.text.x = element_text(angle = 90, hjust = 1), axis.text.y = element_text(size=6)) + 
    ggtitle("Patch",paste0(patchOrder[i]))
}
do.call("grid.arrange", c(p, ncol= 5))

adc.set1 <- c("Ashe","Caitlyn", "Draven", "Ezreal", "Kog'Maw", "Lucian", "Jhin", "Tristana", "Vayne", "Varus", "Xayah")
adc.set <- adc.set1

df <- items.adc[championName %in% adc.set[1:4]][, list(count=.N), by = c("championName", "patch", "itemName")]
setkeyv(df, c("championName", "patch"))
championCount <- items.adc[championName %in% adc.set,list(championName, patch)][,list(gamesPlayed = .N), by = c("championName", "patch")]
setkeyv(championCount, c("championName", "patch"))
df <- merge(df, championCount, by= c("championName","patch"))
df$perc <- df$count/df$gamesPlayed


p <- ggplot(data = df) + 
  geom_bar(stat= "identity", aes(x=itemName, y=perc)) + 
  facet_grid(championName ~ factor(patch, levels = patchOrder)) + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1, size = 6), axis.text.y = element_text(size=6)) + 
  scale_x_discrete(limits=relItems.ADC) +
  coord_flip() +
  guides(fill=FALSE)
p

#nur ne Idee, aber da erkennt leider keiner etwas.
p <- ggplot(data = df[itemName %in% relItems.ADC],aes(x=factor(patch, levels = patchOrder), fill = factor(itemName))) + 
  geom_bar(width=0.9, position="fill") + 
  facet_grid(. ~ championName) +
  theme_igray() + scale_fill_manual(values = colorRampPalette(brewer.pal(8, "Accent"))(length(relItems.ADC)),
                                    guide = guide_legend(nrow=2)) +
  theme(legend.position="bottom")
p + coord_polar(theta = "y")


### attempt to illustrate specific traits of adcs
# OVERALL
dfprep = adc.performance.patch %>%
  mutate(DPS = totalDamageToChampions/gameDuration) %>%
  select(
    name, patch,
    #games,
    #summoners,
    #winrate,
    DPS,
    DmgDealt = totalDamageToChampions,
    kills,
    assists, 
    deaths,
    DmgTaken = totalDamageTaken,
    cs = csPerGame,
    gold = goldEarned
  )
dfprep = data.table(dfprep)

df = dfprep %>%
  rownames_to_column( var = "champ" ) %>%
  mutate_each(funs(rescale), -c(champ,name,patch)) %>%
  melt(id.vars=c('champ','name','patch'), measure.vars=colnames(dfprep[,-c("name","patch")])) %>%
  arrange(champ)
df = data.table(df)

#radar charts: better filter out some champs
df[name %in% c(adc.set1,adc.set2)] %>%
  ggplot(aes(x=variable, y=value, group=name, color=name)) + 
  geom_polygon()
geom_polygon(fill=NA) + 
  coord_radar() + theme_bw() + facet_grid(name~factor(patch, levels=patchOrder)) + 
  #scale_x_discrete(labels = abbreviate) + 
  theme(axis.text.x = element_text(size = 5), legend.position="none")

#bar chart perspective
df %>%
  ggplot(aes(x=variable, y=value, group= name, fill = name)) + 
  geom_bar(stat="identity") + 
  geom_line(y = 0.5, linetype  =2, color = "black") +
  facet_grid(factor(patch, levels=patchOrder)~name) +
  coord_flip() +
  theme_igray() + scale_fill_manual(values = colorRampPalette(brewer.pal(8, "Accent"))(length(unique(df$name)))) +
  theme(axis.text.y = element_text(size = 5), legend.position="none")

dfprep %>%
  mutate_each(funs(rescale), -c(name, patch)) %>% data.table() -> df_radar

p <- list()
for(j in 1:length(patchOrder))
  p[[j]] <- ggRadar(df_radar[name %in% adc.set & patch == patchOrder[j]], aes(color=name), rescale = F, ylim=c(0,1))
do.call("grid.arrange", c(p, nrow = 4 ))

## attempt to scale into 3d and illustrate
adc.performance.patch %>%
  rownames_to_column(var="id") -> adc.cmd 

adc.cmd.numerics = adc.cmd[,-c("name", "championId", "lane", "role", "patch")]
setkey(adc.cmd.numerics, "id")
rownames(adc.cmd.numerics) = adc.cmd.numerics$id
adc.cmd.dist <- dist(adc.cmd.numerics[,-1])
colnames(adc.cmd.dist) = rownames

fit <- cmdscale(adc.cmd.dist, k = 3)
data.table(fit) %>%
  rownames_to_column(var = "id") %>%
  merge(adc.cmd[,c("id","name", "patch")], by="id") -> fit2 
fit2$detailedName = paste0(fit2$name, " ", fit2$patch)
kmeans3 = kmeans(x = fit2[,2:4], centers = 3)
kmeans4 = kmeans(x = fit2[,2:4], centers = 4)
kmeans5 = kmeans(x = fit2[,2:4], centers = 5)
kmeans6 = kmeans(x = fit2[,2:4], centers = 6)

fit2$cluster3 = kmeans3$cluster
fit2$cluster4 = kmeans4$cluster
fit2$cluster5 = kmeans5$cluster
fit2$cluster6 = kmeans6$cluster

plot3d(fit2[,2:4], size = 10, col = fit2$cluster6)
text3d(fit2[,2:4], texts = fit2$detailedName, size=2)

## GAME DURATION, DPS AND DAMAGE DONE

# duration of games where champions participated
adc.performance.patch[championId %in% relchamps.adc] %>%
  ggplot(aes(y = gameDuration, x = patch, group=name)) + 
  geom_line() + 
  scale_x_discrete(limits=patchOrder) + 
  facet_wrap(~name, ncol = 4)


# damage done to champions
adc.performance.patch[championId %in% relchamps.adc] %>%
  ggplot(aes(y = totalDamageToChampions, x = patch, group=name)) + 
  geom_line() + 
  scale_x_discrete(limits=patchOrder) + 
  facet_wrap(~name, ncol = 4)


# dps
adc.performance.patch[championId %in% relchamps.adc] %>%
  ggplot(aes(y = totalDamageToChampions/(gameDuration/60), x = patch, group=name)) + 
  geom_line() + 
  scale_x_discrete(limits=patchOrder) + 
  facet_wrap(~name, ncol = 4)

# dps by win/loss
df = adc.performance.patch.win[championId %in% relchamps.adc]
ggplot(df,aes(y = totalDamageToChampions/(gameDuration/60), x = patch, group=as.factor(win), color = as.factor(win))) + 
  geom_line() + 
  scale_x_discrete(limits=patchOrder) + 
  facet_wrap(~name, ncol = 4)

## gold earned and gold per min

adc.performance.patch[championId %in% relchamps.adc] %>%
  ggplot(aes(y = goldEarned, x = patch, group=name)) + 
  geom_line() + 
  scale_x_discrete(limits=patchOrder) + 
  facet_wrap(~name, ncol = 4)

adc.performance.patch[championId %in% relchamps.adc] %>%
  ggplot(aes(y = goldEarned/(gameDuration/60), x = patch, group=name)) + 
  geom_line() + 
  scale_x_discrete(limits=patchOrder) + 
  facet_wrap(~name, ncol = 4)

# win and loss
df = adc.performance.patch.win[championId %in% relchamps.adc]
ggplot(df,aes(y = goldEarned/(gameDuration/60), x = patch, group=as.factor(win), color = as.factor(win))) + 
  geom_line() + 
  scale_x_discrete(limits=patchOrder) + 
  facet_wrap(~name, ncol = 4)

### cs and xp development

example_champs_xp <- c("Caitlyn", "Twitch", "Tristana", "Xayah", "Kog'Maw", "Jhin", "Varus", "Vayne", "Kalista", "Jinx")
#cs diffs
df = adc.performance.patch[championId %in% relchamps.adc, list(name, patch, csDiffPerMinTen, csDiffPerMinTwenty, csDiffPerMinThirty)]  
melt(df, id.vars=c('name','patch'), measure.vars=colnames(df[,-c("name","patch")])) %>% 
  ggplot(aes(y = value, x = variable, group = name)) + 
  geom_line() + scale_x_discrete(labels=c("10","20","30")) + 
  facet_grid(name~factor(patch, levels = patchOrder))

df = adc.performance.patch[name %in% example_champs_xp, list(name, patch, csDiffPerMinTen, csDiffPerMinTwenty, csDiffPerMinThirty)]  
melt(df, id.vars=c('name','patch'), measure.vars=colnames(df[,-c("name","patch")])) %>% 
  ggplot(aes(y = value, x = variable, group = name)) + 
  geom_line() + scale_x_discrete(labels=c("10","20","30")) + 
  facet_grid(name~factor(patch, levels = patchOrder))

df %>%
  group_by(name) %>%
  summarise(
    csDiffPerMinTen = mean(csDiffPerMinTen),
    csDiffPerMinTwenty = mean(csDiffPerMinTwenty),
    csDiffPerMinThirty = mean(csDiffPerMinThirty)
  ) %>% data.table -> df2
melt(df2, id.vars=c('name'), measure.vars=colnames(df2[,-c("name")])) %>% 
  ggplot(aes(y = value, x = variable, group = name)) + 
  geom_line() + scale_x_discrete(labels=c("10","20","30")) +
  geom_line() + scale_x_discrete(labels=c("10","20","30")) + 
  ylim(-0.3, 0.3)+
  xlab("Minutes in Match") + ylab("Creep Score Differential to Opponent ADC") + 
  theme_bw() + 
  theme(
    axis.text.x = element_text(size=7),
    strip.text.x = element_text(size=7),
    strip.text.y = element_text(size=7)
  ) +
  facet_wrap(~name, ncol=2)  

#xp Diffs
df = adc.performance.patch[championId %in% relchamps.adc, list(name, patch, xpDiffPerMinTen, xpDiffPerMinTwenty, xpDiffPerMinThirty)]  
melt(df, id.vars=c('name','patch'), measure.vars=colnames(df[,-c("name","patch")])) %>% 
  ggplot(aes(y = value, x = variable, group = name)) + 
  geom_line() + scale_x_discrete(labels=c("10","20","30")) + 
  xlab("Minutes in Match") + ylab("XP Differential to Opponent's ADC") + 
  theme_bw() + 
  theme(
    axis.text.x = element_text(size=7),
    strip.text.x = element_text(size=7),
    strip.text.y = element_text(size=7)
  ) +
  facet_grid(name~factor(patch, levels = patchOrder)) 

df = adc.performance.patch[name %in% example_champs_xp, list(name, patch, xpDiffPerMinTen, xpDiffPerMinTwenty, xpDiffPerMinThirty)] 
melt(df, id.vars=c('name','patch'), measure.vars=colnames(df[,-c("name","patch")])) %>% 
  ggplot(aes(y = value, x = variable, group = name)) + 
  geom_line() + scale_x_discrete(labels=c("10","20","30")) + 
  facet_grid(name~factor(patch, levels = patchOrder))  

df %>%
  group_by(name) %>%
  summarise(
    xpDiffPerMinTen = mean(xpDiffPerMinTen),
    xpDiffPerMinTwenty = mean(xpDiffPerMinTwenty),
    xpDiffPerMinThirty = mean(xpDiffPerMinThirty)
  ) %>% data.table -> df2
melt(df2, id.vars=c('name'), measure.vars=colnames(df2[,-c("name")])) %>% 
  ggplot(aes(y = value, x = variable, group = name)) + 
  geom_line() + scale_x_discrete(labels=c("10","20","30")) +
  geom_line() + scale_x_discrete(labels=c("10","20","30")) + 
  xlab("Minutes in Match") + ylab("XP Differential to Opponent's ADC") + 
  theme_bw() + 
  theme(
    axis.text.x = element_text(size=7),
    strip.text.x = element_text(size=7),
    strip.text.y = element_text(size=7)
  ) +
  facet_wrap(~name, ncol=2)  

# cs deltas
df = adc.performance.patch[championId %in% relchamps.adc, list(name, patch, csPerMinDeltaTen, csPerMinDeltaTwenty, csPerMinDeltaThirty)]  
melt(df, id.vars=c('name','patch'), measure.vars=colnames(df[,-c("name","patch")])) %>% 
  ggplot(aes(y = value, x = variable, group = name)) + 
  geom_line() + scale_x_discrete(labels=c("0-10","10-20","20-30")) + 
  facet_grid(name~factor(patch, levels = patchOrder)) + 
  ggtitle("Creep Score Per Minute Delta for each Champion and Patch")

df = adc.performance.patch[name %in% example_champs_xp, list(name, patch, csPerMinDeltaTen, csPerMinDeltaTwenty, csPerMinDeltaThirty)]  
melt(df, id.vars=c('name','patch'), measure.vars=colnames(df[,-c("name","patch")])) %>% 
  ggplot(aes(y = value, x = variable, group = name)) + 
  geom_line() + scale_x_discrete(labels=c("0-10","10-20","20-30")) + 
  facet_grid(name~factor(patch, levels = patchOrder))

df %>%
  group_by(name) %>%
  summarise(
    csPerMinDeltaTen = mean(csPerMinDeltaTen),
    csPerMinDeltaTwenty = mean(csPerMinDeltaTwenty),
    csPerMinDeltaThirty = mean(csPerMinDeltaThirty)
  ) %>% data.table -> df2
melt(df2, id.vars=c('name'), measure.vars=colnames(df2[,-c("name")])) %>% 
  ggplot(aes(y = value, x = variable, group = name)) + 
  geom_line() + scale_x_discrete(labels=c("10","20","30")) +
  geom_line() + scale_x_discrete(labels=c("10","20","30")) + 
  xlab("Minutes in Match") + ylab("Creep Score per Min") + 
  theme_bw() + 
  theme(
    axis.text.x = element_text(size=7),
    strip.text.x = element_text(size=7),
    strip.text.y = element_text(size=7)
  ) +
  facet_wrap(~name, ncol=2)  

## first blood

adc.performance.patch[championId %in% relchamps.adc] %>%
  ggplot(aes(y = firstBloodKill + firstBloodAssist, x = patch, group=name)) + 
  geom_line() + 
  scale_x_discrete(limits=patchOrder) + 
  facet_wrap(~name, ncol = 4)


adc.performance.patch.win[championId %in% relchamps.adc] %>%
  ggplot(aes(y = firstBloodKill + firstBloodAssist, x = patch, group=as.factor(win), color=as.factor(win))) + 
  geom_line() + 
  scale_x_discrete(limits=patchOrder) + 
  facet_wrap(~name, ncol = 4)



# first tower

adc.performance.patch[championId %in% relchamps.adc] %>%
  ggplot(aes(y = firstTowerKill + firstTowerAssist, x = patch, group=name)) + 
  geom_line() + 
  scale_x_discrete(limits=patchOrder) + 
  facet_wrap(~name, ncol = 4)


#first inhib

adc.performance.patch[championId %in% relchamps.adc] %>%
  ggplot(aes(y = firstInhibitorKill + firstInhibitorAssist, x = patch, group=name)) + 
  geom_line() + 
  scale_x_discrete(limits=patchOrder) + 
  facet_wrap(~name, ncol = 4)


## winrate by game duration 
df = adc.performance.patch[,list(name, patch, winrate, winrateAt25, winrateAt30, winrateAt35, winrateAt40, winrateAt45, winrateOver45)]
df1 = df %>%
  rownames_to_column(var = "champ") %>%
  melt(id.vars=c('champ', 'name', 'patch'), measure.vars = colnames(df[,-c("name", "patch")]))

df1$variableNew <- c(0,25,30,35,40,45,50)[match(df1$variable, c("winrate", "winrateAt25", "winrateAt30", "winrateAt35", "winrateAt40","winrateAt45", "winrateOver45"))]
winrate_scale = c("winrateAt25", "winrateAt30", "winrateAt35", "winrateAt40","winrateAt45", "winrateOver45")
p = ggplot(data = df1[variableNew!=0], aes(x = variableNew, y=value, color = name)) + 
  geom_point() + 
  geom_smooth(method="auto") + 
  facet_grid(name ~ factor(patch, levels = patchOrder)) + 
  scale_y_continuous(limits=c(0,1)) + 
  scale_x_continuous(limits = c(20,55), breaks = seq(25,50,5), labels=c( "25", "30", "35" ,"40", "45", ">45")) + 
  theme(legend.position="none")
p

############################ SUP ####################################
#overall distribution in season 7
p <- ggplot(data=sup.distribution,aes(x=names, y=gamesPlayed)) + 
  geom_bar(stat='identity') +
  scale_x_discrete(limits=sup.distribution[order(by=gamesPlayed,decreasing = T)]$names) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1), axis.text.y = element_text(size=6))
p

p <- ggplot(data=sup.distribution,aes(x=names, y=gamesPlayed)) + 
  geom_bar(stat='identity', fill="#56B4E9") +
  scale_x_discrete(limits=sup.distribution[order(by=gamesPlayed,decreasing = T)][1:30]$names) +
  labs(x = "Champion", y = "#matches", title="Distribution of Top 30 Supports played in Season 7") +
  theme_bw() + 
  theme(axis.text.x = element_text(size=16,angle = 90, hjust = 1), 
        axis.text.y = element_text(size=6),
        title = element_text(size=40),
        axis.title = element_text(size=24)) 
p

matches <- data.table(dbGetQuery(connection, "select patch, count(*)/10 as count from playerdetails group by patch"))
champions <- data.table(dbGetQuery(connection, "select championId, patch, count(*) as gamesPlayed from playerdetails group by championId,patch"))
sup.performance.patch %>%
  merge(
    matches, by ="patch"
  ) %>%
  merge(
    champions, by=c('patch', 'championId')
  ) -> sup.performance.patch
sup.performance.patch$playrate = sup.performance.patch$gamesPlayed/sup.performance.patch$count


#Barchart 
p <- ggplot(data = sup.performance, aes(x=name, y=games, fill = name)) + 
  geom_bar(stat="Identity") + 
  facet_wrap( ~ platformId+factor(patch, levels = patchOrder), nrow=4) + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1), axis.text.y = element_text(size=6)) + 
  labs(x = "Champion", y = "Playrate in Percentages") +
  coord_flip() + 
  guides(fill=F)
p+ ggtitle("Support Picks per Patch and Region")
p

#linechart 1
p <- ggplot(data = sup.performance.patch[championId %in% relchamps.sup], aes(x = patch, y=playrate, group=name)) + 
  geom_line(stat='identity') +  
  scale_x_discrete(limits=patchOrder) +
  ylim(0,0.6) +
  theme_bw() +
  theme(axis.text.x = element_text(size=8, angle=90),
        legend.position = "bottom")+
  labs(x = "Patch", y = "Playrate in Percentage", color="Region") +
  facet_wrap(~name, ncol = 4) 
p

#linechart 2b
p <- ggplot(data = sup.performance.patch[championId %in% relchamps.sup], aes(x = patch, y=playrate*100, group=name)) + 
  geom_line(stat='identity') +  
  scale_x_discrete(limits=patchOrder) +
  ylim(0,60) +
  theme_bw() +
  theme(axis.text.x = element_text(size=8, angle=90),
        legend.position = "bottom")+
  labs(x = "Patch", y = "Playrate in Percentage", color="Region") +
  facet_wrap(~name, ncol = 4) 
p



after_p14 = c("7.14", "7.15", "7.16", "7.17", "7.18") 
relchamps.sup2 <- sup[patch %in% after_p14] %>% group_by(championId) %>% summarise(count = n()) %>% arrange(desc(count)) %>% head(8)

#linechart 2b
p <- ggplot(data = sup.performance.patch[championId %in% relchamps.sup2$championId], aes(x = patch, y=playrate*100, group=name)) + 
  geom_line(stat='identity') +  
  scale_x_discrete(limits=patchOrder) +
  ylim(0,60) +
  theme_bw() +
  theme(axis.text.x = element_text(size=8, angle=90),
        legend.position = "bottom")+
  labs(x = "Patch", y = "Playrate in Percentage", color="Region") +
  facet_wrap(~name, ncol = 2) 
p


df <- items.sup[championId %in% relchamps.sup][, list(count=.N), by = c("championId","championName", "patch", "itemName")]
df$itemName = ifelse(df$itemName == 'Ardent Censer', 'Ardent Censer', 'Other')
df$patch[!(df$patch %in% after_p14)] = "< 7.14" 
setkeyv(df, c("championName", "patch"))
items.sup_p14 <- items.sup
items.sup_p14$patch[!(items.sup_p14$patch %in% after_p14)] = "< 7.14"
championCount <- items.sup_p14[,list(championName, patch)][,list(gamesPlayed = .N), by = c("championName", "patch")]
setkeyv(championCount, c("championName", "patch"))
df <- merge(df, championCount, by= c("championName","patch"))
df$perc <- df$count/df$gamesPlayed

df %>% 
  group_by(championName, patch) %>%
  summarise(purchase_per_patch = sum(count)) -> df_2
df <- merge(df, df_2, by=c("championName", "patch"))
df$perc_rel_items = df$count/df$purchase_per_patch

relsup <- sup[championId %in% relchamps.sup2$championId,list(gameId, championId, platformId, patch, item0, item1, item2, item3, item4, item5, item6)]
relsup$patch[!(relsup$patch %in% after_p14)] = "< 7.14"
relsup %>%
  merge(
    champLookUp,
    by = "championId"
  ) -> relsup
relsup$ardent <- ifelse((relsup$item0 == 3504) | (relsup$item1 == 3504) |( relsup$item2 == 3504) | (relsup$item3 == 3504) |
                          (relsup$item4 == 3504) | (relsup$item5 == 3504) | (relsup$item6 == 3504), 1, 0)

relsup %>%
  group_by(name, patch) %>%
  summarise(
    games = n(),
    ardent = sum(ardent)
  ) -> relsup_grp
relsup_grp$ratio = relsup_grp$ardent/relsup_grp$games

p <- ggplot(data = relsup_grp) + 
  geom_bar(stat= "identity", aes(x=factor(1), y=ratio), width = 1) + 
  coord_polar(theta = "y", start =0) + 
  facet_grid(name ~ factor(patch, levels = c("< 7.14", "7.14", "7.15", "7.16", "7.17", "7.18"))) + 
  ylab('')+ xlab('') + 
  labs(fill = "Item Name") +
  theme_bw() +
  theme(axis.ticks.y = element_blank(),
        axis.text.y = element_blank(),
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        strip.text.x = element_text(size=7),
        strip.text.y = element_text(size=7),
        legend.position = "bottom") +
  guides(fill=guide_legend(nrow=3,byrow=TRUE))
p
