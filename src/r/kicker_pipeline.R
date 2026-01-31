library(dplyr)

install.packages("nflfastR")
library(nflfastR)
library(nflreadr)

# Aggregate the data using the sum function, grouped by the "team" and "week" columns



Ult <- data.frame()
wk <- 7
year <- 2005
while(year < 2023){ 
  repdf <- data.frame()
while(wk < 21 && year < 2023){
  nflreadr::load_player_stats(seasons = year,stat_type = "kicking") -> K
  filter(K, week < wk, week > wk - 7) -> K
#kf is team based offensive kicker stats
kf <- aggregate(K[7:37], by = list(as.factor(K$team)), sum)
rename(kf, team = Group.1) -> kf
as.factor(kf$team) -> kf$team
if (year < 2016) {
  kf$team <- gsub("LAC", "SD", kf$team)
}
if (year < 2017) {
  kf$team <- gsub("LA", "STL", kf$team)
}
if (year < 2020) {
  kf$team <- gsub("LV", "OAK", kf$team)
}


load_schedules(year) -> gp
gp<- filter(gp, week < wk, week > wk - 7)
data.frame(table(gp$home_team)) -> h
data.frame(table(gp$away_team)) -> a
rename(a, g = Freq, team = Var1) -> a
names(a)[1] <- "team"
cbind(a,h) -> ah
mutate(ah, G = g + Freq) -> ah
select(ah, team, G) -> gamesplayed


library(sqldf)
qk <- "SELECT *
FROM kf 
LEFT JOIN gamesplayed
ON kf.team = gamesplayed.team;"
sqldf(qk) -> try

#remove dup col
select(try, -33) ->try

#add fg per game
try <- mutate(try, attpg = try$fg_att/try$G)

#rename
kickdf <- try




load_schedules(seasons = year) -> sched
filter(sched, week == wk #change
) -> sched
#if (year < 2016) {
 # kickdf$team[kickdf$team == "LAC"] <- "SD"
#}
#if (year < 2017) {
 # kickdf$team[kickdf$team == "LA"] <- "STL"
#}
#if (year < 2020) {
 # kickdf$team[kickdf$team == "LV"] <- "OAK"
#}
qeer <- "SELECT kickdf.*, sched.away_team AS oppteam
FROM kickdf
LEFT JOIN sched
ON kickdf.team = sched.home_team
UNION
SELECT kickdf.*, sched.home_team AS oppteam
FROM kickdf 
LEFT JOIN sched
ON kickdf.team = sched.away_team"
sqldf(qeer) -> atry
atry <- atry[!is.na(atry$oppteam),]
atry -> kickerdf 
remove(atry)

load_pbp(year) -> pbp
filter(pbp, pbp$field_goal_attempt > 0, pbp$week < wk,pbp$week > wk - 7) -> fieg
fieg <- fieg %>% filter(!is.na(field_goal_attempt))
fieg$field_goal_result <- ifelse(fieg$field_goal_result == "made", 1, 0)
ex <- select(fieg, defteam, posteam, field_goal_result, field_goal_attempt)
aggregate(ex[3:4], by = list(as.factor(ex$defteam)), sum) -> att
aggregate(ex[3:4], by = list(as.factor(ex$posteam)), sum) -> oatt
rename(att, team = Group.1) -> att
rename(oatt, team = Group.1) -> oatt
if (year < 2016) {
  att$team <- gsub("LAC", "SD", att$team)
}
if (year < 2017) {
  att$team <- gsub("LA", "STL", att$team)
}
if (year < 2020) {
  att$team <- gsub("LV", "OAK", att$team)
}

if (year < 2016) {
  oatt$team <- gsub("LAC", "SD", oatt$team)
}
if (year < 2017) {
  oatt$team <- gsub("LA", "STL", oatt$team)
}
if (year < 2020) {
  oatt$team <- gsub("LV", "OAK", oatt$team)
}

qk <- "SELECT *
FROM att 
LEFT JOIN gamesplayed
ON att.team = gamesplayed.Team;"
sqldf(qk) -> attdf
select(attdf, -4) -> attdf


attdf$pweek <- wk


qer <- "SELECT kickerdf.*, attdf.field_goal_attempt AS oppfga, attdf.G
FROM kickerdf
LEFT JOIN attdf
ON kickerdf.oppteam = attdf.team;" 
qert <- sqldf(qer)
qert -> kickers
remove(qert)
colnames(kickers)[37] <- "OG"
mutate(kickers, oppfgapg = kickers$oppfga/kickers$OG) -> kickers
mutate(kickers, projfga = (kickers$oppfgapg + kickers$attpg)/2) -> kickers
kickers$pweek = wk
kickers$year = year
repdf <- rbind(repdf, kickers)
wk <- wk + 1
}
  
repdf -> kickers


dLdf <- data.frame()
Ldf <- data.frame()
wk <- 7
#load pbp foradjusted kicker stats
while(wk < 21){
load_pbp(year) -> pbp
filter(pbp, pbp$field_goal_attempt > 0, pbp$week < wk, pbp$week > wk - 7, pbp$down == 4) -> fieg
fieg <- fieg %>% filter(!is.na(field_goal_attempt))
fieg$field_goal_result <- ifelse(fieg$field_goal_result == "made", 1, 0)
ex <- select(fieg, defteam, posteam, field_goal_result, field_goal_attempt)
aggregate(ex[3:4], by = list(as.factor(ex$defteam)), sum) -> att
aggregate(ex[3:4], by = list(as.factor(ex$posteam)), sum) -> oatt
rename(att, team = Group.1) -> att
rename(oatt, team = Group.1) -> oatt
if (year < 2016) {
  att$team <- gsub("LAC", "SD", att$team)
}
if (year < 2017) {
  att$team <- gsub("LA", "STL", att$team)
}
if (year < 2020) {
  att$team <- gsub("LV", "OAK", att$team)
}

if (year < 2016) {
  oatt$team <- gsub("LAC", "SD", oatt$team)
}
if (year < 2017) {
  oatt$team <- gsub("LA", "STL", oatt$team)
}
if (year < 2020) {
  oatt$team <- gsub("LV", "OAK", oatt$team)
}

att -> attdf


attdf$pweek <- wk

oatt -> oattdf
oattdf$pweek <- wk
Ldf <- rbind(Ldf, oattdf)
dLdf <- rbind(dLdf,attdf)
wk <- wk + 1
}
Ldf -> oattdf
dLdf -> attdf

load_schedules(year) ->schedule
as.factor(schedule$home_team) -> schedule$home_team
if (year < 2016) {
  kickers$team <- gsub("LAC", "SD", kickers$team)
}
if (year < 2017) {
  kickers$team <- gsub("LA", "STL", kickers$team)
}
if (year < 2020) {
  kickers$team <- gsub("LV", "OAK", kickers$team)
}
if (year > 2016) {
  kickers$team <- gsub("SD","LAC",  kickers$team)
}
if (year > 2017) {
  kickers$team <- gsub( "STL","LA", kickers$team)
}
if (year > 2020) {
  kickers$team <- gsub("OAK","LV",  kickers$team)
}

q1 <- "SELECT *
  FROM schedule
LEFT JOIN kickers
ON kickers.pweek = schedule.week AND schedule.home_team = kickers.team;"
sqldf(q1) -> step1

as.factor(step1$away_team) -> step1$away_team

q2 <- "SELECT *
  FROM step1
LEFT JOIN kickers
ON kickers.pweek = step1.week AND step1.away_team = kickers.team;"
sqldf(q2) -> step2

col_names <- names(step2)
# Select the column names that you want to modify
cols_to_modify <- col_names[46:86]
# Modify the column names
col_names[46:86] <- paste0("home.", cols_to_modify)
# Assign the modified column names to the data frame
names(step2) <- col_names

col_names <- names(step2)
# Select the column names that you want to modify
cols_to_modify <- col_names[87:127]
# Modify the column names
col_names[87:127] <- paste0("away.", cols_to_modify)
# Assign the modified column names to the data frame
names(step2) <- col_names


q3 <- "SELECT *
  FROM step2
LEFT JOIN attdf
ON attdf.pweek = step2.week AND step2.away_team = attdf.team;"
sqldf(q3) -> step3

q4 <- "SELECT *
  FROM step3
LEFT JOIN attdf
ON attdf.pweek = step3.week AND step3.home_team = attdf.team;"
sqldf(q4) -> step4

col_names <- names(step4)
# Select the column names that you want to modify
cols_to_modify <- col_names[128:131]
# Modify the column names
col_names[128:131] <- paste0("away.adjopp", cols_to_modify)
# Assign the modified column names to the data frame
names(step4) <- col_names

col_names <- names(step4)
# Select the column names that you want to modify
cols_to_modify <- col_names[132:135]
# Modify the column names
col_names[132:135] <- paste0("home.adjopp", cols_to_modify)
# Assign the modified column names to the data frame
names(step4) <- col_names




q5 <- "SELECT *
  FROM step4
LEFT JOIN oattdf
ON oattdf.pweek = step4.week AND step4.away_team = oattdf.team;"
sqldf(q5) -> step5

q6 <- "SELECT *
  FROM step5
LEFT JOIN oattdf
ON oattdf.pweek = step5.week AND step5.home_team = oattdf.team;"
sqldf(q6) -> step6

col_names <- names(step6)
# Select the column names that you want to modify
cols_to_modify <- col_names[136:139]
# Modify the column names
col_names[136:139] <- paste0("away.adj", cols_to_modify)
# Assign the modified column names to the data frame
names(step6) <- col_names

col_names <- names(step6)
# Select the column names that you want to modify
cols_to_modify <- col_names[140:143]
# Modify the column names
col_names[140:143] <- paste0("home.adj", cols_to_modify)
# Assign the modified column names to the data frame
names(step6) <- col_names

step6 -> kickanalysis

kickanalysis <- filter(kickanalysis, week > 6)

nflreadr::load_player_stats(seasons = T, stat_type = "kicking") -> kjr

if (year < 2016) {
  kjr$team <- gsub("LAC", "SD", kjr$team)
}
if (year < 2017) {
  kjr$team <- gsub("LA", "STL", kjr$team)
}
if (year < 2020) {
  kjr$team <- gsub("LV", "OAK", kjr$team)
}


data <- kjr[,7:16]

# Aggregate the data using the sum function, grouped by the "team" and "week" columns
aggregate(data, by = list(team = kjr$team, week = kjr$week, season = kjr$season), sum) ->maybe
actual <- select(maybe, team, week, season , fg_att, fg_made, pat_made)


fq <- "SELECT *
  FROM kickanalysis
LEFT JOIN actual
ON actual.week = kickanalysis.week AND kickanalysis.away_team = actual.team AND kickanalysis.season = actual.season;
"
sqldf(fq) -> may

col_names <- names(may)
# Select the column names that you want to modify
cols_to_modify <- col_names[144:149]
# Modify the column names
col_names[144:149] <- paste0("actualaway.", cols_to_modify)
# Assign the modified column names to the data frame
names(may) <- col_names

ffq <- "SELECT *
  FROM may
LEFT JOIN actual
ON actual.week = may.week AND may.home_team = actual.team AND may.season = actual.season;
"
sqldf(ffq) -> kanal

col_names <- names(kanal)
# Select the column names that you want to modify
cols_to_modify <- col_names[150:155]
# Modify the column names
col_names[150:155] <- paste0("actualhome.", cols_to_modify)
# Assign the modified column names to the data frame
names(kanal) <- col_names

Ult <- rbind(Ult, kanal)
year <- year + 1
wk <- 7 }

#compile gamesplayed dataframe

 year <- 2005
ggpp <- data.frame()
while(year < 2023){
wk <- 7
  while(wk < 21){
load_schedules(year) -> gp
gp<- filter(gp, week < wk, week > wk - 7)
data.frame(table(gp$home_team)) -> h
data.frame(table(gp$away_team)) -> a
rename(a, g = Freq, team = Var1) -> a
names(a)[1] <- "team"
cbind(a,h) -> ah
mutate(ah, G = g + Freq) -> ah
select(ah, team, G) -> gamesplayed
gamesplayed$week <- wk
gamesplayed$season <- year
ggpp <- rbind(ggpp, gamesplayed)
wk <- wk + 1
  }
year <- year + 1
}
  
#merge games played with rest of df

qq <- "SELECT Ult.*, ggpp.G
  FROM Ult
LEFT JOIN ggpp
ON ggpp.week = Ult.week AND ggpp.season = Ult.season AND ggpp.team = Ult.away_team;"
  sqldf(qq) -> stup
colnames(stup)[156] <- "awaygamesplayed"
  
qqq <- "SELECT stup.*, ggpp.G
  FROM stup
LEFT JOIN ggpp
ON ggpp.week = stup.week AND ggpp.season = stup.season AND ggpp.team = stup.home_team;"
  sqldf(qqq) -> wgames
colnames(wgames)[157] <- "homegamesplayed"
wgames -> kickeranalysis

library(tidyr)
#add fantasy outcome
year <- 2005
kfppg <- data.frame()
while(year < 2023){
  load_player_stats(seasons = year, stat_type = "kicking") -> fgmdf
  separate(data = fgmdf, col = fg_made_list, into = c('firstfg', "second", 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth'), sep = "\\;") -> madelist
  for(col in 38:45){
    for(row in 1:nrow(madelist)){
      madelist[row,col] <- substring(madelist[row,col], 1, nchar(madelist[row,col])-1)
    }
  }
  as.numeric(madelist$firstfg) -> madelist$firstfg
  as.numeric(madelist$second)->madelist$second
  as.numeric(madelist$third)->madelist$third
  as.numeric(madelist$fourth)-> madelist$fourth
  as.numeric(madelist$fifth)->madelist$fifth
  as.numeric(madelist$sixth)->madelist$sixth
  as.numeric(madelist$seventh)->madelist$seventh
  as.numeric(madelist$eighth)->madelist$eighth
  for(row in 1:nrow(madelist)){
    for(col in 38:45){
      if(!is.na(madelist[row, col])){
        if(madelist[row, col] == 1 || madelist[row, col] == 2){
          madelist[row, col] <- 3
        }else if(madelist[row, col] == 6){
          madelist[row, col] <- 5
        }
      }
    }
  }
  
  for(col in 38:45){
    madelist[,col][is.na(madelist[,col])] <- 0
  }
  
  mutate(madelist, fppg = pat_made + (1*firstfg+1*second+1*third+1*fourth+ 1*fifth+1*sixth+ 1*seventh + 1*eighth) )->fgmdata
  select(fgmdata, season,week,team,fppg) ->addfp
  if (year < 2016) {
    addfp$team <- gsub("LAC", "SD", addfp$team)
  }
  if (year < 2017) {
    addfp$team <- gsub("LA", "STL", addfp$team)
  }
  if (year < 2020) {
    addfp$team <- gsub("LV", "OAK", addfp$team)
  }
  if (year > 2016) {
    addfp$team <- gsub("SD","LAC",  addfp$team)
  }
  if (year > 2017) {
    addfp$team <- gsub( "STL","LA", addfp$team)
  }
  if (year > 2020) {
    addfp$team <- gsub("OAK","LV",  addfp$team)
  }
  rbind(kfppg, addfp)->kfppg
  year<- year +1}
aggregate(kfppg$fppg, by=list(kfppg$season, kfppg$team, kfppg$week), sum)-> kfppg
names(kfppg)<- c('season', 'team','week','fppg')

sqldf("SELECT k.*, kfppg.fppg AS homefppg
FROM kickeranalysis AS k
LEFT JOIN kfppg
ON kfppg.week = k.week AND kfppg.season =  k.season AND kfppg.team = k.home_team")-> kfkfkf

sqldf("SELECT k.*, kfppg.fppg AS awayfppg
FROM kfkfkf AS k
LEFT JOIN kfppg
ON kfppg.week = k.week AND kfppg.season =  k.season AND kfppg.team = k.away_team")-> kfkf

kfkf->kickeranalysis

#add additional metrics
kickeranalysis <- mutate(kickeranalysis, adjhome.attpg = kickeranalysis$home.adjfield_goal_attempt / kickeranalysis$homegamesplayed, adjaway.attpg = kickeranalysis$away.adjfield_goal_attempt / kickeranalysis$awaygamesplayed)
kickeranalysis <- mutate(kickeranalysis, adjhome.attpgagainst = kickeranalysis$home.adjoppfield_goal_attempt / kickeranalysis$homegamesplayed, adjaway.attpgallowed = kickeranalysis$away.adjoppfield_goal_attempt / kickeranalysis$awaygamesplayed)
kickeranalysis <- mutate(kickeranalysis, homeexpatt = (kickeranalysis$adjhome.attpg + kickeranalysis$adjaway.attpgallowed) / 2, awayexpatt = (kickeranalysis$adjaway.attpg + kickeranalysis$adjhome.attpgagainst)/2)
#add boolean cash column identifying whether the over 1.5 FGM that is offered for most nfl games hit or not
kickeranalysis <- kickeranalysis %>% mutate(homecash = ifelse(actualhome.fg_made > 1.5, 1, -1))
kickeranalysis <- kickeranalysis %>% mutate(awaycash = ifelse(actualaway.fg_made > 1.5, 1, -1))
kickeranalysis <- mutate(kickeranalysis, homexppg = home.pat_made/homegamesplayed, awayxppg = away.pat_made/awaygamesplayed)
kickeranalysis <- mutate(kickeranalysis, homekick_pct = home.fg_made / (home.fg_made + home.fg_missed), awaykick_pct = away.fg_made / (away.fg_made + away.fg_missed))
kickeranalysis <- mutate(kickeranalysis, homeexfppg = homeexpatt * 3.8* homekick_pct + homexppg, awayexfppg = awayexpatt * 3.8*awaykick_pct +awayxppg)
kickeranalysis$awayfppg <- ifelse(is.na(kickeranalysis$awayfppg), 0, kickeranalysis$awayfppg)
kickeranalysis$homefppg <- ifelse(is.na(kickeranalysis$homefppg), 0, kickeranalysis$homefppg)
kickeranalysis <-  mutate(kickeranalysis, adjhomeexpatt = (.35*(kickeranalysis$adjhome.attpg) + .65*(kickeranalysis$adjaway.attpgallowed)), adjawayexpatt = (.35*(kickeranalysis$adjaway.attpg) + .65*(kickeranalysis$adjhome.attpgagainst)))
kickeranalysis<- mutate(kickeranalysis, homeregressfp = 3.96760 + 0.40388 * homexppg + 1.11950 * adjhomeexpatt + 2.24413 * homekick_pct - 0.66712 * adjhomeexpatt * homekick_pct)
kickeranalysis<- mutate(kickeranalysis, awayregressfp = 3.96760 + 0.40388 * awayxppg + 1.11950 * adjawayexpatt + 2.24413 * awaykick_pct - 0.66712 * adjawayexpatt * awaykick_pct)


select(kickeranalysis,174:175,172,173, homefppg,awayfppg, 170,171)->fppg
select(fppg,1, 3,5,7) ->homefant
select(fppg,2,4,6,8)-> awayfant
names(homefant) <- c('exatt',"exfp","fp", "pct")
names(awayfant) <- c('exatt',"exfp","fp", "pct")
rbind(homefant,awayfant)-> fantasyhistory
remove(x)
x<- 5
for(x in 1:nrow(kickeranalysis)){
afantasythreshold <- kickeranalysis$awayexfppg[x] 
afpercentreq <- kickeranalysis$awaykick_pct[x]
hfantasythreshold <- kickeranalysis$homeexfppg[x] 
hfpercentreq <- kickeranalysis$homekick_pct[x]
filter(fantasyhistory, exfp >= afantasythreshold*0.95 & exfp <= afantasythreshold*1.05, pct >= afpercentreq*0.95 & pct <= afpercentreq*1.05) -> afntsythresh#pct >= fpercentreq*0.95 & pct <= fpercentreq*1.05
filter(fantasyhistory, exfp >= hfantasythreshold*0.95 & exfp <= hfantasythreshold*1.05,pct >= hfpercentreq*0.95 & pct <= hfpercentreq*1.05) -> hfntsythresh#pct >= fpercentreq*0.95 & pct <= fpercentreq*1.05
mean(afntsythresh$fp) -> aoutcome
mean(hfntsythresh$fp) -> houtcome

kickeranalysis$homehistoricalfantasy[x] <- houtcome
kickeranalysis$awayhistoricalfantasy[x] <- aoutcome
}


 library(tidyr)
#end of df building




#To view a week of predictions
currentweek <- 20


filter(kickeranalysis, week == currentweek, season == 2022) -> semis
semisx <- select(semis, home_team, away_team, homeexpatt, awayexpatt,awayexfppg,homeexfppg ,roof,homekick_pct, awaykick_pct, homehistoricalfantasy, awayhistoricalfantasy)
select(semisx, home_team, homeexpatt,homeexfppg ,roof, homekick_pct, homehistoricalfantasy)->hhh
select(semisx, away_team, awayexpatt, awayexfppg,roof, awaykick_pct,awayhistoricalfantasy)->aaa
names(hhh) -> names(aaa)
rbind(aaa,hhh)-> wket
names(wket) <- c("team", "ex_att(adj)",'fantasypoints', "roof", 'pct', 'historicalfantasypoints')
wket ->fantasythisweek


#model for currentweek
thisweekhist <- data.frame(matrix(nrow = nrow(wket), ncol = 9))
i <- 1
as.character(wket$team) -> wket$team
for(x in 1:nrow(wket)){
  threshold <- wket[x,2]
  greatorless<- "o"
  percentreq <- wket[x,5]
  team <- wket[x,1]
  select(kickeranalysis, actualaway.fg_att, actualaway.pat_made, actualaway.fg_made, actualhome.fg_att,actualhome.pat_made ,actualhome.fg_made ,awaycash, homecash, home.fg_att, home.fg_made, away.fg_att, away.fg_made, adjawayexpatt, adjhomeexpatt) -> threshresultsa
  select(threshresultsa, 1:3, 7,11,12,13) -> athresh
  select(threshresultsa, 4:6, 8:10,14) -> hthresh
  colnames(athresh) -> atnms
  names(hthresh)<- atnms
  rbind(athresh,hthresh) ->threshresults
  names(threshresults) <- c("fgatt", "xpmade", "fgmade", "cash?", "pastatt","pastmade", "exatt")
  mutate(threshresults, pct = pastmade/pastatt) -> threshresults
  mutate(threshresults, pctadjexatt = pct * exatt) -> threshresults
  threshresults[, c("pastatt", "pastmade","pct")][is.na(threshresults[, c("pastatt", "pastmade", "pct")])] <- 0
  threshresults[is.na(threshresults)] <- 0
  na.omit(threshresults)-> threshresults
  filter(threshresults,pct >= percentreq*0.94 & pct <= percentreq*1.06,exatt >= threshold*0.94 & exatt <= threshold*1.06) -> threshresults
  #,pct < percentreq
  select(kickeranalysis, week,season) -> weeks
  unique(weeks)->weeks
  nrow(threshresults)/ nrow(weeks) ->eventlikelihood
  sum(threshresults$`cash?`)-> unitprofit
  pval <- jarque.bera.test(threshresults$`cash?`)$p.value
  (nrow(threshresults)/2 + unitprofit)/nrow(threshresults) -> hitrate
  library(tseries)
  thisweekhist$threshold[i] <- threshold
  thisweekhist$weeksperoccurence[i] <- eventlikelihood
  thisweekhist$o_u[i] <- greatorless
  thisweekhist$kickpctthresh[i] <- percentreq
  thisweekhist$pvalue[i] <- pval
  thisweekhist$observations[i] <- nrow(threshresults)
  thisweekhist$team[i] <- team
  ifelse(thisweekhist$o_u[i] == 'u',  thisweekhist$profit[i] <- unitprofit * -1, thisweekhist$profit[i] <- unitprofit)
  ifelse(thisweekhist$o_u[i] == 'u',  thisweekhist$hitrate[i] <- 1-hitrate, thisweekhist$hitrate[i] <- hitrate)
  i<- i + 1}
select(thisweekhist, 10:18)-> thisweekhist
thisweekhist$o_u <- ifelse(thisweekhist$hitrate < .5, "u", thisweekhist$o_u)
ifelse(thisweekhist$o_u == 'u',  thisweekhist$profit * -1,  thisweekhist$profit)-> thisweekhist$profit
ifelse(thisweekhist$o_u == 'u',  1-thisweekhist$hitrate,  thisweekhist$hitrate)->thisweekhist$hitrate



#find realistic fantasy kicker prediction (88th percentile outcome at higher end for each week aka 3rd best option)
select(kickeranalysis, season, week, awayexfppg, homeexfppg) ->groupka
select(groupka,season,week, homeexfppg)-> hff
select(groupka, season,week, awayexfppg)-> aff
names(aff)<- c('season','week','averagefntsyoption')
names(hff)<- c('season', 'week','averagefntsyoption')
rbind(hff,aff) -> groupka
groupka %>% 
  group_by(season, week) %>% 
  summarize(homexfppg_90th_percentile = quantile(averagefntsyoption, 0.88, na.rm = T)) %>%
  arrange(season, week) -> averageoption
mean(averageoption$homexfppg_90th_percentile,na.rm = T) ->avgfantasyoptionexfp

#realistic kicker expected attempt threshold to use for over unders(1.5 per week)
select(kickeranalysis, season, week, awayexpatt, homeexpatt) ->groupkatt
select(groupkatt,season,week, homeexpatt)-> hfft
select(groupkatt, season,week, awayexpatt)-> afft
names(afft)<- c('season','week','exatt')
names(hfft)<- c('season', 'week','exatt')
rbind(hfft,afft) -> groupkatt
groupkatt %>% 
  group_by(season, week) %>% 
  summarize(homexatt_90th_percentile = quantile(exatt, 0.95, na.rm = T)) %>%
  arrange(season, week) -> averageattoption
mean(averageattoption$homexatt_90th_percentile,na.rm = T)
#can expect 1-2 2.29+ attempts predicted every week
#can expect 1-2 1.227 or less attempts predicted every week 68% under hit rate
  



newdf <- data.frame(matrix(nrow = 20, ncol = 8))
names(newdf) <- c("threshold",'weeksperoccurence' ,'o_u' ,'kickpctthresh' ,'pvalue' ,'observations', 'profit' ,'hitrate')
i<-1
#threshold analysis, created newdf for different percent and expected attempt hit rates,occurences and profit since 05
threshold <- 1.22
greatorless<- "u"
percentreq <- NA
select(kickeranalysis, actualaway.fg_att, actualaway.pat_made, actualaway.fg_made, actualhome.fg_att,actualhome.pat_made ,actualhome.fg_made ,awaycash, homecash, home.fg_att, home.fg_made, away.fg_att, away.fg_made, awayexpatt, homeexpatt) -> threshresultsa
select(threshresultsa, 1:3, 7,11,12,13) -> athresh
select(threshresultsa, 4:6, 8:10,14) -> hthresh
colnames(athresh) -> atnms
names(hthresh)<- atnms
rbind(athresh,hthresh) ->threshresults
names(threshresults) <- c("fgatt", "xpmade", "fgmade", "cash?", "pastatt","pastmade", "exatt")
mutate(threshresults, pct = pastmade/pastatt) -> threshresults
mutate(threshresults, pctadjexatt = pct * exatt) -> threshresults
threshresults[, c("pastatt", "pastmade","pct")][is.na(threshresults[, c("pastatt", "pastmade", "pct")])] <- 0
threshresults[is.na(threshresults)] <- 0
na.omit(threshresults)-> threshresults
filter(threshresults,exatt >= threshold*0.97 & exatt <= threshold*1.03) -> threshresults
#,pct < percentreq
select(kickeranalysis, week,season) -> weeks
unique(weeks)->weeks
nrow(threshresults)/ nrow(weeks) ->eventlikelihood
sum(threshresults$`cash?`)-> unitprofit
pval <- jarque.bera.test(threshresults$`cash?`)$p.value
(nrow(threshresults)/2 + unitprofit)/nrow(threshresults) -> hitrate
library(tseries)
newdf$threshold[i] <- threshold
newdf$weeksperoccurence[i] <- eventlikelihood
newdf$o_u[i] <- greatorless
newdf$kickpctthresh[i] <- percentreq
newdf$pvalue[i] <- pval
newdf$observations[i] <- nrow(threshresults)
ifelse(newdf$o_u[i] == 'u',  newdf$profit[i] <- unitprofit * -1, newdf$profit[i] <- unitprofit)
ifelse(newdf$o_u[i] == 'u',  newdf$hitrate[i] <- 1-hitrate, newdf$hitrate[i] <- hitrate)
i<- i + 1
write.csv(newdf, "kicker")
rbind(newdf,newdf2)->newdf

summary(threshresults)
#<= 1.3 ex att + <= .9 pct kicks gives 58% under hit rate 2.033observation every 5.8397
#>= 2.7 ex att + >= .9 pct kicks gives 60% over hit rate
#>= 2.6 ex att  gives 59% over hit rate
#<= 1.5 ex att + <= .65 pct kicks gives 60% under hit rate



fnewdf <- data.frame(matrix(nrow = 30, ncol = 7))
names(fnewdf) <- c("fantasythreshold",'weeksperoccurence' ,'kickpctthresh' ,'fpval' ,'observations','advantage', 'outcome')
i<-1

#fantasy kicker threshold analysis

select(kickeranalysis,168:173)->fppg
select(fppg, 1,3,5) ->homefant
select(fppg,2,4,6)-> awayfant
names(homefant) <- c("exfp","fp", "pct")
names(awayfant) <- c("exfp","fp", "pct")
rbind(homefant,awayfant)-> fantasyhistory
fantasythreshold <- 10.25 #88 percentile exppg aka the average best option for each week
fpercentreq <- .95
filter(fantasyhistory, exfp >= fantasythreshold*0.95 & exfp <= fantasythreshold*1.05, pct > fpercentreq) -> fntsythresh#pct >= fpercentreq*0.95 & pct <= fpercentreq*1.05
mean(fntsythresh$fp) -> outcome
kickeranalysis$historicalfantasy 
mean(fantasyhistory$fp) -> avgoutcome
fpval <- jarque.bera.test(fntsythresh$fp)$p.value
fnewdf$fantasythreshold[i] <- fantasythreshold
fnewdf$weeksperoccurence[i] <- nrow(fntsythresh)/nrow(weeks)
fnewdf$kickpctthresh[i] <- fpercentreq
fnewdf$fpval[i] <- fpval
fnewdf$observations[i] <- nrow(fntsythresh)
fnewdf$advantage[i]<- outcome - avgoutcome
fnewdf$outcome[i]<-outcome
i<- i + 1

fnewdf[2,]<- NA
summary(fntsythresh)
summary(fantasyhistory)
sd(fantasyhistory$fp)
#mean fantasy points when exp fp < 9.719 = 7.688 median = 7
#mean fantasy points when exp fp > 9.719 aka 88th percentile option = 8.173 median = 8 (88th percentile used to id realistic kicker option 3rd or 4th best every week)

jarque.bera.test(fntsythresh$fp)





select(kickeranalysis, actualaway.fg_att, awayxppg, actualaway.fg_made, actualhome.fg_att,homexppg ,actualhome.fg_made ,awaycash, homecash, home.fg_att, home.fg_made, away.fg_att, away.fg_made, adjawayexpatt, adjhomeexpatt,homegamesplayed,awaygamesplayed,homefppg,awayfppg) -> threshresultsa
select(threshresultsa, 1:3, 7,11,12,13,16,18) -> athresh
select(threshresultsa, 4:6, 8:10,14,15,17) -> hthresh
colnames(athresh) -> atnms
names(hthresh)<- atnms
rbind(athresh,hthresh) ->threshresults
names(threshresults) <- c("fgatt", "xpmadeg", "fgmade", "cash?", "pastatt","pastmade", "exatt", 'g', 'fp')
mutate(threshresults, pct = pastmade/pastatt ) -> threshresults


model <- lm(fp ~ xpmadeg + exatt * pct, data = threshresults)
summary(model)
coef(model)
exfpxpcttofpcoef <- round(coef(model)[2],3)
ggplot(fantasyhistory, aes(exfp*pct, fp)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  ggtitle(paste("R-squared: ", round(rsq(model), 3)," Coefficient: ",exfpxpcttofpcoef))

#fantasy regrssion model
model <- lm(fp ~ exatt, data = fantasyhistory)
summary(model)
coef(model)
exfpxpcttofpcoef <- round(coef(model)[2],3)
ggplot(fantasyhistory, aes(exfp*pct, fp)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  ggtitle(paste("R-squared: ", round(rsq(model), 3)," Coefficient: ",exfpxpcttofpcoef))


#do past pats lead to more/less fgs
select(kickeranalysis, actualhome.fg_att ,actualaway.fg_att,away.pat_att  ,home.pat_att, awaygamesplayed, homegamesplayed)-> pastpat
select(pastpat, 1,4,6)-> pat_h
select(pastpat, 2, 3,5)-> pat_a
names(pat_h)<-c('fgatt','pastpat', 'g')
names(pat_a)<-c('fgatt','pastpat', 'g')
rbind(pat_a,pat_h)->attemptanalysis
mutate(attemptanalysis, patpg = pastpat/g)-> patattemptanalysis
patattemptanalysis[is.na(patattemptanalysis)] <- 0
cor(patattemptanalysis)


#analyzing effect of pct and exatt on fgatt w linear regression
select(kickeranalysis, actualaway.fg_att, actualaway.pat_made, actualaway.fg_made, actualhome.fg_att,actualhome.pat_made ,actualhome.fg_made ,awaycash, homecash, home.fg_att, home.fg_made, away.fg_att, away.fg_made, awayexpatt, homeexpatt) -> threshresultsa
select(threshresultsa, 1:3, 7,11,12,13) -> athresh
select(threshresultsa, 4:6, 8:10,14) -> hthresh
colnames(athresh) -> atnms
names(hthresh)<- atnms
rbind(athresh,hthresh)-> attemptanalysis
names(attemptanalysis)<-  c("fgatt", "xpmade", "fgmade", "cash?", "pastatt","pastmade", "exatt")
mutate(attemptanalysis, pct = pastmade/pastatt) -> attemptanalysis
cor(attemptanalysis)-> car
data.frame(car)-> car
model <- lm(fgmade ~ pct + exatt, data = attemptanalysis)
summary(model)
attcoef <- round(coef(model)[2],3)  #.658 coefficient
ggplot(attemptanalysis, aes(exatt*pct, fgatt)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  ggtitle(paste("R-squared: ", round(rsq(model), 3)," Coefficient: ",coef))

#analyzing effect of attempts allowed or own team attempts on eventual fg attempts
select(kickeranalysis, actualhome.fg_att ,adjaway.attpgallowed, adjhome.attpg)-> hdefvoff
select(kickeranalysis, actualaway.fg_att ,adjhome.attpgagainst, adjaway.attpg)-> adefvoff
names(adefvoff) <- c('fgatt', 'allowed','kicked')
names(hdefvoff) <- c('fgatt', 'allowed','kicked')
rbind(hdefvoff,adefvoff)->defvoff
model <- lm(fgatt ~allowed * kicked, data = defvoff)
summary(model)
attcoef <- round(coef(model)[2],3)




na.omit(threshresults)-> threshresults
filter(threshresults, pct <= .65) -> threshresults
sum(threshresults$`cash?`)-> unitprofit
(nrow(threshresults)/2 + unitprofit)/nrow(threshresults)
library(tseries)
jarque.bera.test(threshresults$`cash?`)
summary(threshresults)


kickeranalysis %>%
select(home_team, away_team, homeexpatt,awayexpatt, adjhome.attpg,adjhome.attpgagainst, adjaway.attpg, home.fg_att, home.fg_made, away.fg_att, away.fg_made ,adjaway.attpgallowed, actualhome.fg_att, actualaway.fg_att, actualhome.fg_made, actualaway.fg_made, homecash,awaycash, roof) ->an
levels(as.factor(an$roof))
library(ggplot2)
mater <- na.omit(an[3:18])

select(mater, "homeexpatt", "actualhome.fg_att", "actualhome.fg_made", 'home.fg_att', 'home.fg_made' ,"homecash") ->hc
select(mater, "awayexpatt", "actualaway.fg_att","actualaway.fg_made", 'away.fg_att', 'away.fg_made',"awaycash") ->ac
colnames(ac) -> aqw
names(hc)<- aqw
rbind(hc,ac) ->achc
names(achc) <- c("exatt","fgatt","fgm",'pastatt','pastmade',"cash")
mutate(achc, pct = pastmade/pastatt) -> achc
mutate(achc, pctadjexatt = pct * exatt) -> achc


library(ggplot2)
library(rsq)
model <- lm(fgatt ~ exatt, data = achc)
ggplot(data = achc, aes(x = exatt, y = fgatt)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  ggtitle(paste("R-squared: ", round(rsq(model), 3)))
coef(model)



model <- lm(fgm ~ pctadjexatt, data = achc)
ggplot(data = achc, aes(x = pctadjexatt, y = fgm)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  ggtitle(paste("R-squared: ", round(rsq(model), 3)))
coef(model)


#roof impact analysis
select(kickeranalysis, roof, actualaway.fg_made, actualhome.fg_made, week)-> roofdf
select(roofdf, roof, actualaway.fg_made,week) ->aroof
select(roofdf, roof, actualhome.fg_made,week) ->hroof
names(hroof)<- c("roof", "fgm",'week')
names(aroof)<- c("roof", "fgm", 'week')
rbind(aroof,hroof)->roofdf
filter(roofdf, week > 14)-> cold
sqldf("SELECT avg(fgm), roof
FROM cold
GROUP BY roof")-> roofs


