source("initialize.R")


############################ GENERAL STATS ####################################



############################ TOP ####################################

### Distribution in Season 7
p <- ggplot(data=top.distribution,aes(x=names, y=gamesPlayed)) + 
  geom_bar(stat='identity') +
  scale_x_discrete(limits=top.distribution[order(by=gamesPlayed,decreasing = T)]$names) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1), axis.text.y = element_text(size=6))
p


df= top.distribution.patch[patch==patchOrder[16]]
ggplot(df, aes(x=name, gamesPlayed)) +
  geom_bar(stat = "identity") +
  scale_x_discrete(limits=df[order(by=gamesPlayed,decreasing = T)]$name)

p <- list()
for(i in 1:length(patchOrder)){
  df= top.distribution.patch[patch==patchOrder[i]][order(by=gamesPlayed,decreasing = T)][1:20]
  p[[i]] <- ggplot(df, aes(x=name, y=gamesPlayed)) +
    geom_bar(stat = "identity") +
    scale_x_discrete(limits=df[order(by=gamesPlayed,decreasing = T)]$name) +
    scale_y_continuous(limits=c(0,2000)) +
    theme(axis.text.x = element_text(angle = 90, hjust = 1), axis.text.y = element_text(size=6)) + 
    ggtitle("Patch",paste0(patchOrder[i]))
}
do.call("grid.arrange", c(p, ncol= 5))


p <- ggplot(data=top.distribution.patch,aes(x=names, y=gamesPlayed)) + 
  geom_bar(stat='identity') +
  scale_x_discrete(limits=top.distribution[order(by=gamesPlayed,decreasing = T)]$names) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1), axis.text.y = element_text(size=6)) + 
  facet_wrap(.~patch, scales="free")
p

p <- ggplot(data=merge(top[,c("championId", "patch")], champLookUp, "championId") , aes(names)) + 
  geom_bar() +  facet_wrap(~ patch, scales="free")
p

####PER PATCH AND REGION
p <- ggplot(data = top.relevant) + 
  geom_bar(aes(x=as.vector(champ[as.character(top.relevant$championId)]), fill = championId)) + 
  facet_wrap( ~ platformId+factor(patch, levels = patchOrder), nrow=4) + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1), axis.text.y = element_text(size=6)) + 
  coord_flip() + 
  guides(fill=F)
p+ ggtitle("Toplane Picks per Patch and Region")

p <- ggplot(data=top.distribution,aes(x=names, y=gamesPlayed)) + 
  geom_bar(stat='identity') +
  scale_x_discrete(limits=top.distribution[order(by=gamesPlayed,decreasing = T)]$names) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1), axis.text.y = element_text(size=6))
p

#PER CHAMP
p <- ggplot(data = top.relevant, aes(x = patch, group=platformId, color=platformId)) + 
  geom_line(stat = "count") + 
  scale_x_discrete(limits=patchOrder) +
  facet_wrap(~ championId, ncol = 5) + 
  theme_igray() + scale_colour_tableau("colorblind10")
p

p <- ggplot(data = top.relevant[, championId == 39], aes(x = patch, group=platformId, color=platformId)) + 
  geom_line(stat = "count") + 
  geom_line(y = mean(win) ) + 
  scale_x_discrete(limits=patchOrder) +
  facet_wrap(~ championId, ncol = 5) + 
  theme_dark() + 
  theme(plot.background = element_rect("grey"))
p


p <- ggplot(data = top.performance, aes(x = patch, y=winrate, group=platformId, color=platformId)) + 
  geom_line(linetype = 2) + 
  geom_line(data = top.performance , aes(y = games/2000)) + 
  scale_x_discrete(limits=patchOrder) +
  facet_wrap(~ name, ncol = 5) + 
  theme_dark() + 
  theme(plot.background = element_rect("grey"))
p + ggtitle("top Picks per Patch and Region")


########################### JUNGLE ##################################

#overall distribution in season 7
p <- ggplot(data=jungle.distribution,aes(x=names, y=gamesPlayed)) + 
  geom_bar(stat='identity') +
  scale_x_discrete(limits=jungle.distribution[order(by=gamesPlayed,decreasing = T)]$names) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1), axis.text.y = element_text(size=6))
p

#PER PATCH AND REGION
p <- ggplot(data = jungle.relevant) + 
  geom_bar(aes(x=as.vector(champ[as.character(jungle.relevant$championId)]), fill = championId)) + 
  facet_wrap( ~ platformId+factor(patch, levels = patchOrder), nrow=4) + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1), axis.text.y = element_text(size=6)) + 
  coord_flip() + 
  guides(fill=F)
p+ ggtitle("Jungle Picks per Patch and Region")

p <- ggplot(data = jungle.performance, aes(x = patch, y=winrate, group=platformId, color=platformId)) + 
  geom_line(linetype = 2) + 
  geom_line(data = jungle.performance , aes(y = games/2000)) + 
  scale_x_discrete(limits=patchOrder) +
  facet_wrap(~ name, ncol = 5) + 
  theme_dark() + 
  theme(plot.background = element_rect("grey"))
p + ggtitle("jungle Picks per Patch and Region")


########################### MID ##################################
#overall distribution in season 7
p <- ggplot(data=mid.distribution,aes(x=names, y=gamesPlayed)) + 
  geom_bar(stat='identity') +
  scale_x_discrete(limits=mid.distribution[order(by=gamesPlayed,decreasing = T)]$names) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1), axis.text.y = element_text(size=6))
p

#PER PATCH AND REGION
p <- ggplot(data = mid.relevant) + 
  geom_bar(aes(x=as.vector(champ[as.character(mid.relevant$championId)]), fill = championId)) + 
  facet_wrap( ~ platformId+factor(patch, levels = patchOrder), nrow=4) + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1), axis.text.y = element_text(size=6)) + 
  coord_flip() + 
  guides(fill=F)
p+ ggtitle("Jungle Picks per Patch and Region")

p <- ggplot(data = mid.performance, aes(x = patch, y=winrate, group=platformId, color=platformId)) + 
  geom_line(linetype = 2) + 
  geom_line(data = mid.performance , aes(y = games/2000)) + 
  scale_x_discrete(limits=patchOrder) +
  facet_wrap(~ name, ncol = 5) + 
  theme_dark() + 
  theme(plot.background = element_rect("grey"))
p + ggtitle("Mid Lane Picks per Patch and Region")


############################ ADC ####################################

####DISTRIBUTIONS OF CHAMPIONS
#overall distribution in season 7
p <- ggplot(data=adc.distribution,aes(x=names, y=gamesPlayed)) + 
  geom_bar(stat='identity') +
  scale_x_discrete(limits=adc.distribution[order(by=gamesPlayed,decreasing = T)]$names) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1), axis.text.y = element_text(size=6))
p

#Barchart 
p <- ggplot(data = adc.performance, aes(x=name, y=games/2000 *100, fill = name)) + 
  geom_bar(stat="Identity") + 
  facet_wrap( ~ platformId+factor(patch, levels = patchOrder), nrow=4) + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1), axis.text.y = element_text(size=6)) + 
  labs(x = "Champion", y = "Playrate in Percentages") +
  coord_flip() + 
  guides(fill=F)
p+ ggtitle("ADC Picks per Patch and Region")

#Linechart
p <- ggplot(data = adc.performance[championId %in% relchamps.adc], aes(x = patch, y=games/2000 * 100, group=platformId, color=platformId)) + 
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
      games,
      summoners,
      winrate,
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
  geom_line() + 
  geom_line() + 
  scale_x_discrete(limits=patchOrder) + 
  facet_wrap(~name, ncol = 4)


# dps
adc.performance.patch[championId %in% relchamps.adc] %>%
  ggplot(aes(y = totalDamageToChampions/(gameDuration/60), x = patch, group=name)) + 
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
  ggplot(aes(y = goldEarned/gameDuration, x = patch, group=name)) + 
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
  theme(axis.text.x = element_text(angle = 90, hjust = 1), axis.text.y = element_text(size=6))
p

p <- ggplot(data = sup.relevant) + 
  geom_bar(aes(x=as.vector(champ[as.character(sup.relevant$championId)]), fill = as.vector(champ[as.character(sup.relevant$championId)]))) + 
  facet_wrap( ~ platformId+factor(patch, levels = patchOrder), nrow=4) + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1), axis.text.y = element_text(size=6)) +
  coord_flip() + 
  guides(fill=F) 
p+ ggtitle("Support Picks per Patch and Region")

