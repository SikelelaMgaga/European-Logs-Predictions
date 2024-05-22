
#Machine Learning Walkthrough: Predicting the World Cup

#Intro
This is a walkthrough on using Machine Learning while trying to predict who is going to win the next FIFA World Cup.

We're going to use historical information about international football (soccer) matches to build a model, which is going to give us the ability to predict future match results.

Afterwards, we're going to use that model to run multiple simulations of the next World Cup tournament, and produce statistics about which teams are the most likely to win it all.

This document is meant for people who are new to Machine Learning, and want to better understand the data science process, as well as the R language.

NOTE: If you attended any of my presentations about this walkthrough, you can find the presentation slides here.

NOTE 2: The GitHub containing all the code I used is here. You can also find the Javascript code I used to acquire the historical matches data from the FIFA website, as well as the Python program which simulates the tournament thousands of times.

Customizing the Tutorial
If you want to do this walkthrough on your own machine, or if you'd like to customize it, you can clone this notebook by using the Clone button at the top of this page. This will copy the entire notebook into your own Azure Notebooks workspace, where you can edit it.

You can also find an R source file containing the entire code on my GitHub repo. Open that with the editor of your choice - RStudio is a popular one.

With that out of the way, let's begin!

The Process
Below is a typical workflow for a Data Science project such as ours.

Data Science Project Workflow

Our Objective
The first step is defining an objective, an outcome for our model to predict:

Given a match between two teams, what is the expected goal differential at the end of the match?

In other words, we're going to attempt to predict the outcome variable in the table below, which is the difference between the number of goals scored by team1 and the number of goals scored by team2.

Outcome - goal differential

Let's note that the outcome value can be zero (for draws), positive (whenever team1 wins) but also negative (whenever team2 wins the match).

The Data
Matches
We're going to use a dataset containing more than thirty-thousand international football matches played between 1950 and 2017. All these matches are played between senior men's national teams - there are no club matches, and no youth / women's games.

The dataset is available as CSV and JSON files.

Below is a small sample from the JSON file:

[
    {
        "date": "19560930",
        "team1": "AUT",
        "team1Text": "Austria",
        "team2": "LUX",
        "team2Text": "Luxembourg",
        "resText": "7-0",
        "statText": "",
        "venue": "Ernst Happel Stadium - Vienna , Austria",
        "IdCupSeason": "10",
        "CupName": "FIFA World Cup™ Qualifier",
        "team1Score": "7",
        "team2Score": "0"
    },
    {
        "date": "19561003",
        "team1": "IRL",
        "team1Text": "Republic of Ireland",
        "team2": "DEN",
        "team2Text": "Denmark",
        "resText": "2-1",
        "statText": "",
        "venue": "DUBLIN - Dublin , Republic of Ireland",
        "IdCupSeason": "10",
        "CupName": "FIFA World Cup™ Qualifier",
        "team1Score": "2",
        "team2Score": "1"
    },
    ...
]
Teams
We also have some information regarding the international associations and the FIFA confederations they are part of. We may find that useful when looking at past opponents of a team.

This dataset is available as a CSV file.

Below is a sample of the data:

csv
confederation,name,fifa_code,ioc_code

CAF,Algeria,ALG,ALG
CAF,Angola,ANG,ANG
CAF,Benin,BEN,BEN
CAF,Botswana,BOT,BOT
...
AFC,Afghanistan,AFG,AFG
AFC,Australia,AUS,---
AFC,Bahrain,BHR,BRN
AFC,Bangladesh,BAN,BAN
...
UEFA,Albania,ALB,ALB
UEFA,Andorra,AND,AND
UEFA,Armenia,ARM,ARM
UEFA,Austria,AUT,AUT
...
CONMEBOL,Argentina,ARG,ARG
CONMEBOL,Bolivia,BOL,BOL
CONMEBOL,Brazil,BRA,BRA
CONMEBOL,Chile,CHI,CHI
Qualified for the World Cup
Last, we have a list of the teams which have qualified for the World Cup, and their group stage draw.

This dataset is available as a CSV file.

Below is a sample of the data:

csv
name,draw

RUS,A1
IRN,F2
KOR,A3
JPN,G2
KSA,H2
AUS,B3
TUN,A2
NGA,B2
CIV,C2
...
Setup
First, we're going to load a few R libraries from CRAN - the Comprehensive R Archive Network - into our environment.


# prepare the R environment
if (!require("pacman")) install.packages("pacman")
pacman::p_load(
  dplyr,            # Data munging functions
  zoo,              # Feature engineering rolling aggregates
  data.table,       # Feature engineering
  ggplot2,          # Graphics
  scales,           # Time formatted axis
  readr,            # Reading input files
  stringr,          # String functions
  reshape2,         # restructure and aggregate data 
  randomForest,     # Random forests
  corrplot,         # correlation plots
  Metrics,          # Eval metrics for ML
  vcd               # Visualizing discrete distributions
)
    
# set options for plots
options(repr.plot.width=6, repr.plot.height=6)
     
Loading required package: pacman
Warning message in library(package, lib.loc = lib.loc, character.only = TRUE, logical.return = TRUE, :
“there is no package called ‘pacman’”Installing package into ‘/home/nbuser/R’
(as ‘lib’ is unspecified)
Installing package into ‘/home/nbuser/R’
(as ‘lib’ is unspecified)

Metrics installed

# Load the matches data

if(!file.exists("matches.csv")){
    tryCatch(download.file('https://github.com/neaorin/PredictTheWorldCup/raw/master/input/matches.csv'
                           ,destfile="./matches.csv",method="auto"))
}
                
if(file.exists("matches.csv")) matches_original <- read_csv("matches.csv")
    
head(matches_original)
     
Parsed with column specification:
cols(
  date = col_integer(),
  team1 = col_character(),
  team1Text = col_character(),
  team2 = col_character(),
  team2Text = col_character(),
  venue = col_character(),
  IdCupSeason = col_integer(),
  CupName = col_character(),
  team1Score = col_integer(),
  team2Score = col_integer(),
  statText = col_character(),
  resText = col_character(),
  team1PenScore = col_character(),
  team2PenScore = col_character()
)
date	team1	team1Text	team2	team2Text	venue	IdCupSeason	CupName	team1Score	team2Score	statText	resText	team1PenScore	team2PenScore
19500308	WAL	Wales	NIR	Northern Ireland	Cardiff, Wales	6	FIFA competition team qualification	0	0	NA	0-0	NA	NA
19500402	ESP	Spain	POR	Portugal	Madrid, Spain	6	FIFA competition team qualification	5	1	NA	5-1	NA	NA
19500409	POR	Portugal	ESP	Spain	Lisbon, Portugal	6	FIFA competition team qualification	2	2	NA	2-2	NA	NA
19500415	SCO	Scotland	ENG	England	Glasgow, Scotland	6	FIFA competition team qualification	0	1	NA	0-1	NA	NA
19500624	BRA	Brazil	MEX	Mexico	Rio De Janeiro, Brazil	7	FIFA competition team final	4	0	NA	4-0	NA	NA
19500625	ENG	England	CHI	Chile	Rio De Janeiro, Brazil	7	FIFA competition team final	2	0	NA	2-0	NA	NA
First let's perform some basic cleanup on the dataset.


# eliminate any duplicates that may exist in the dataset
matches <- matches_original %>%
  distinct(.keep_all = TRUE, date, team1, team2)

# the date field is formatted as a string (e.g. 19560930) - transform that into R date
matches$date <- as.POSIXct(strptime(matches$date, "%Y%m%d"), origin="1960-01-01", tz="UTC")

# generate an id column for future use (joins etc)
matches$match_id = seq.int(nrow(matches))

head(matches)
summary(matches)
     
date	team1	team1Text	team2	team2Text	venue	IdCupSeason	CupName	team1Score	team2Score	statText	resText	team1PenScore	team2PenScore	match_id
1950-03-08	WAL	Wales	NIR	Northern Ireland	Cardiff, Wales	6	FIFA competition team qualification	0	0	NA	0-0	NA	NA	1
1950-04-02	ESP	Spain	POR	Portugal	Madrid, Spain	6	FIFA competition team qualification	5	1	NA	5-1	NA	NA	2
1950-04-09	POR	Portugal	ESP	Spain	Lisbon, Portugal	6	FIFA competition team qualification	2	2	NA	2-2	NA	NA	3
1950-04-15	SCO	Scotland	ENG	England	Glasgow, Scotland	6	FIFA competition team qualification	0	1	NA	0-1	NA	NA	4
1950-06-24	BRA	Brazil	MEX	Mexico	Rio De Janeiro, Brazil	7	FIFA competition team final	4	0	NA	4-0	NA	NA	5
1950-06-25	ENG	England	CHI	Chile	Rio De Janeiro, Brazil	7	FIFA competition team final	2	0	NA	2-0	NA	NA	6
      date                        team1            team1Text        
 Min.   :1950-02-17 00:00:00   Length:31810       Length:31810      
 1st Qu.:1983-08-23 06:00:00   Class :character   Class :character  
 Median :1998-02-17 12:00:00   Mode  :character   Mode  :character  
 Mean   :1994-06-06 00:47:48                                        
 3rd Qu.:2007-10-17 00:00:00                                        
 Max.   :2017-12-29 00:00:00                                        
                                                                    
    team2            team2Text            venue            IdCupSeason       
 Length:31810       Length:31810       Length:31810       Min.   :6.000e+00  
 Class :character   Class :character   Class :character   1st Qu.:7.404e+03  
 Mode  :character   Mode  :character   Mode  :character   Median :2.000e+09  
                                                          Mean   :1.163e+09  
                                                          3rd Qu.:2.000e+09  
                                                          Max.   :2.000e+09  
                                                                             
   CupName            team1Score       team2Score       statText        
 Length:31810       Min.   : 0.000   Min.   : 0.000   Length:31810      
 Class :character   1st Qu.: 1.000   1st Qu.: 0.000   Class :character  
 Mode  :character   Median : 1.000   Median : 1.000   Mode  :character  
                    Mean   : 1.682   Mean   : 1.102                     
                    3rd Qu.: 2.000   3rd Qu.: 2.000                     
                    Max.   :31.000   Max.   :22.000                     
                    NA's   :2        NA's   :2                          
   resText          team1PenScore      team2PenScore         match_id    
 Length:31810       Length:31810       Length:31810       Min.   :    1  
 Class :character   Class :character   Class :character   1st Qu.: 7953  
 Mode  :character   Mode  :character   Mode  :character   Median :15906  
                                                          Mean   :15906  
                                                          3rd Qu.:23858  
                                                          Max.   :31810  
                                                                         
Data Exploration and Visualisation
More often than not, the best way to understand a dataset is to turn it into a picture.

Or rather, multiple pictures.

Fortunately, R has some useful tools in this regard - and a lot of them come with the very popular ggplot2 package.

Some useful resources when learning to use ggplot2 are:

The R for Data Science ebook, Chapter 3. Data Visualisation
Data Visualization with ggplot2 Cheat Sheet
The ggplot2 explorer app
Essential Cheat Sheets for deep learning and machine learning researchers
For example, let's get a sense on the number of games which have been played over the years, and how close they were from a competitive standpoint.


# how many international games have been played over the years?
matches %>%
  ggplot(mapping = aes(year(date))) +
    geom_bar(aes(fill=CupName), width=1, color="black") +
    theme(legend.position = "bottom", legend.direction = "vertical") + ggtitle("Matches played by Year")
     
b'\n

# how many goals have been scored per game over the years? 
matches %>%
  dplyr::group_by(year = year(date)) %>%
  dplyr::summarize(
    totalgames = n(),
    totalgoals = sum(team1Score + team2Score),
    goalspergame = totalgoals / totalgames
    ) %>%
  ggplot(mapping = aes(x = year, y = goalspergame)) +
    geom_point() +
    geom_smooth(method = "loess") + ggtitle("Goals scored per game, over time")
     
Warning message:
“Removed 2 rows containing non-finite values (stat_smooth).”Warning message:
“Removed 2 rows containing missing values (geom_point).”
b'\n

# what values is our dataset missing?

ggplot_missing <- function(x){

  x %>%
    is.na %>%
    melt %>%
    ggplot(mapping = aes(x = Var2,
               y = Var1)) +
    geom_raster(aes(fill = value)) +
    scale_fill_grey(name = "",
                    labels = c("Present","Missing")) +
    theme(axis.text.x  = element_text(angle=45, vjust=0.5)) +
    labs(x = "Variables in Dataset",
         y = "Rows / observations")
}

ggplot_missing(matches)

#Amelia::missmap(matches, main = "Missing values")
     
b'\n
A few things we can note from our graphs above, as well as examining the dataset:

Although our sample size of matches played before 1960 is fairly small, we can note that the era of free-wheeling, mostly attacking football coming to an end with the perfection of devensive tactics like catenaccio in Italy, and eventually with the era of Total Football which took off in the early '70s. We may need to factor in some of these developments into our model.

pen1Score and pen2Score are only present whenever a match ended on penalties. All missing values indicate a match which did not get to a penalty shootout. Therefore, the values aren't really missing, so we can use these features if we want; however, the number of observations is fairly small, and penalty shoot-outs do have a reputation of being a lottery of sorts, especially at the highest level of play when the prize is advancement to a later stage of the World Cup.

One additional thing to note is that, for matches ending on penalties, team1Score = team2Score; these values do not include the penalty shoot-out goals; still, those games weren't really draws, since they were decided on penalties. From a purely performance standpoint however, we might decide to consider a team losing on penalties as being closer to a draw than an actual loss.

statText is also only present for matches which didn't end in regulation time. It includes extra-time matches as well as penalty shoot-outs as above. Unlike pen1Score and pen2Score however, team1Score and team2Score do include all the goals scored in extra time as well. Same as before, we might decide that a team performed better if they lost in extra time vs. a loss in regulation; however, we will disregard this field for now.

venue is interesting for the purpose of determining the home team, which in football has a distinct advantage (as we will later conclude). However, since venue is a text field, we will need to do some pattern matching with team names to determine the correct home team, which may present several problems. Also, about 15 percent of values are missing for this column, which will force us to consider these matches as being played in a neutral venue.

CupName can be useful to determine whether a game was a played as a friendly, a qualifier, or a final tournament. Simple pattern matching will be enough for this task.

IdCupSeason can be ignored at this time.

resText can be ignored as all the information is also contained in other non-text fields.

Finally, we may have an issue with the fact that team1 consistently performs better than team2 - likely because most of the games list the home team first:


summary(matches
team2Score)
     
    Min.  1st Qu.   Median     Mean  3rd Qu.     Max.     NA's 
-22.0000  -1.0000   0.0000   0.5805   2.0000  31.0000        2 
The Mean value for the goal differential is greater than 0.5, which may present a problem later on when training the model - it may capture this bias team1 is better than team2, which is something we'd rather avoid, especially since the World Cup final tournament is played in a single country.

So let's get rid of that by simply randomizing the order in which teams are listed for any one match.


set.seed(4342)
matches$switch = runif(nrow(matches), min = 0, max = 1)

matches <- bind_rows(
  matches %>% dplyr::filter(switch < 0.5),
  matches %>% dplyr::filter(switch >= 0.5) %>%
    dplyr::mutate(
      x_team2 = team2,
      team2 = team1,
      team1 = x_team2,
      
      x_team2Text = team2Text,
      team2Text = team1Text,
      team1Text = x_team2Text,

      x_resText = "",
      
      x_team2Score = team2Score,
      team2Score = team1Score,
      team1Score = x_team2Score,
      
      x_team2PenScore = team2PenScore,
      team2PenScore = team1PenScore,
      team1PenScore = x_team2PenScore
    ) %>%
    dplyr::select(
      date, team1, team1Text, team2, team2Text, resText, statText, venue, IdCupSeason, CupName, team1Score, team2Score, team1PenScore, team2PenScore, match_id, switch
    )
    ) %>% 
  dplyr::arrange(date) %>%
  dplyr::select(-c(switch))

summary(matches
team2Score)
     
     Min.   1st Qu.    Median      Mean   3rd Qu.      Max.      NA's 
-31.00000  -1.00000   0.00000  -0.01339   1.00000  22.00000         2 
Now, we can create some aditional features about the matches.


# is the game played in a neutral venue
matches
team1Text, x=matches$venue, MoreArgs = list(fixed = TRUE, ignore.case = FALSE))
matches
team2Text, x=matches$venue, MoreArgs = list(fixed = TRUE, ignore.case = FALSE))
matches
team1Home | matches$team2Home)

# text-matching the venue is not 100% accurate.
# some games get TRUE for both team1 and team2 (ex. Congo DR vs Congo)
# in this case, team1 is at home
matches
team1Home == TRUE) & (matches$team2Home == TRUE)] <- FALSE

# game type: Friendly, Qualifier, Final Tournament
matches
friendly[matches$CupName == "Friendly"] <- TRUE

matches
qualifier[matches$CupName %like% "qual"] <- TRUE

matches
finaltourn[matches$CupName %like% "final"] <- TRUE

head(matches)
     
date	team1	team1Text	team2	team2Text	venue	IdCupSeason	CupName	team1Score	team2Score	...	resText	team1PenScore	team2PenScore	match_id	team1Home	team2Home	neutralVenue	friendly	qualifier	finaltourn
1950-02-17	EGY	Egypt	GRE	Greece	Cairo, Egypt	2000010101	Friendly	2	0	...	2-0	NA	NA	27	TRUE	FALSE	FALSE	TRUE	FALSE	FALSE
1950-02-25	HAI	Haiti	SLV	El Salvador	Guatemala City, Guatemala	2000010101	Friendly	0	1	...	1-0	NA	NA	28	FALSE	FALSE	TRUE	TRUE	FALSE	FALSE
1950-02-26	SLV	El Salvador	CRC	Costa Rica	Guatemala City, Guatemala	2000010101	Friendly	0	1	...	0-1	NA	NA	29	FALSE	FALSE	TRUE	TRUE	FALSE	FALSE
1950-02-27	CRC	Costa Rica	CUW	Curacao	Guatemala City, Guatemala	2000010101	Friendly	1	0	...	1-0	NA	NA	30	FALSE	FALSE	TRUE	TRUE	FALSE	FALSE
1950-02-27	GUA	Guatemala	COL	Colombia	Guatemala City, Guatemala	2000010101	Friendly	2	1	...	2-1	NA	NA	31	TRUE	FALSE	FALSE	TRUE	FALSE	FALSE
1950-02-28	NCA	Nicaragua	HAI	Haiti	Guatemala City, Guatemala	2000010101	Friendly	2	4	...	2-4	NA	NA	32	FALSE	FALSE	TRUE	TRUE	FALSE	FALSE
At this point, we're going to eliminate friendly matches from the dataset.

This decision is based on the observation that, with few exceptions, the main objective for a team playing a friendly is not to win it, but to evaluate its own players and tactics.

For this reason it's not uncommon for friendlies to allow an unlimited number of substitutions, and for a team to roll out its entire squad during a friendly game.

If you'd like to experiment with keeping friendlies in the dataset, you can comment out the line below.


# only use official matches (no friendlies)
matches <- matches %>% dplyr::filter(friendly == FALSE)
     
Up until this point we've only looked at individual matches. However, what we really need is to look at each team's performance over its history.

When we build our predictive model, we'd like to supply it with as many features about each of the teams about to be involved in a match. For that, we need to have a team-centric dataset with historical data.

Building this dataset is simple: take each observation in matches - which has the form "team1 vs team2" - and produce two separate observations of the form "team1 played against team2" and "team2 played against team1" respectively.


# transform the matches table into a team performance table, where each team being 
# involved in a game is a separate observation (row)

teamperf <- bind_rows(
    (matches %>%
    dplyr::mutate(
      name = team1,
      opponentName = team2,
      homeVenue = team1Home,
      neutralVenue = neutralVenue,
      gs = team1Score,
      ga = team2Score,
      gd = gs - ga,
      w = (team1Score > team2Score),
      l = (team1Score < team2Score),
      d = (team1Score == team2Score),
      friendly = friendly,
      qualifier = qualifier,
      finaltourn = finaltourn
    ) %>%
    dplyr::select (match_id, date, name, opponentName, homeVenue, neutralVenue, gs, ga, gd, w, l, d, friendly, qualifier, finaltourn))
    ,
    (matches %>%
    dplyr::mutate(
      name = team2,
      opponentName = team1,
      homeVenue = team2Home,
      neutralVenue = neutralVenue,
      gs = team2Score,
      ga = team1Score,
      gd = gs - ga,
      w = (team1Score < team2Score),
      l = (team1Score > team2Score),
      d = (team1Score == team2Score),
      friendly = friendly,
      qualifier = qualifier,
      finaltourn = finaltourn
    ) %>%
      dplyr::select (match_id, date, name, opponentName, homeVenue, neutralVenue, gs, ga, gd, w, l, d, friendly, qualifier, finaltourn))
  ) %>%
  dplyr::arrange(date)

head(teamperf)
     
match_id	date	name	opponentName	homeVenue	neutralVenue	gs	ga	gd	w	l	d	friendly	qualifier	finaltourn
1	1950-03-08	NIR	WAL	FALSE	FALSE	0	0	0	FALSE	FALSE	TRUE	FALSE	TRUE	FALSE
1	1950-03-08	WAL	NIR	TRUE	FALSE	0	0	0	FALSE	FALSE	TRUE	FALSE	TRUE	FALSE
2	1950-04-02	ESP	POR	TRUE	FALSE	5	1	4	TRUE	FALSE	FALSE	FALSE	TRUE	FALSE
2	1950-04-02	POR	ESP	FALSE	FALSE	1	5	-4	FALSE	TRUE	FALSE	FALSE	TRUE	FALSE
3	1950-04-09	ESP	POR	FALSE	FALSE	2	2	0	FALSE	FALSE	TRUE	FALSE	TRUE	FALSE
3	1950-04-09	POR	ESP	TRUE	FALSE	2	2	0	FALSE	FALSE	TRUE	FALSE	TRUE	FALSE
In order to capture some information about how good each team is, let's define a winning percentage formula:

winpercentage = (wins + 0.5 * draws) / games played

Then, let's plot that for each team which has played a significant number of games.

We're going to define the win percentage formula and plot as R functions, since we might want to re-use them after we further tweak our dataset.


# Out of the teams who have played at least 100 games, what are the winning percentages for each of those teams?

formula_winpercentage <- function(totalgames, wins, draws) {
    return ((wins + 0.5 * draws) / totalgames)
}

plot_winpercentage <- function(teamperf, mingames) {
  teamperf %>%
  group_by(name) %>%
  summarize(
    totalgames = n(),
    wins = length(w[w==TRUE]),
    draws = length(d[d==TRUE]),
    winpercentage = formula_winpercentage(totalgames, wins, draws)
  ) %>%
  filter(totalgames >= mingames ) %>%
  ggplot(mapping = aes(x = winpercentage, y = totalgames)) +
  geom_point(size = 1.5) + 
  geom_text(aes(label=name), hjust=-.2 , vjust=-.2, size=3) +
  geom_vline(xintercept = .5, linetype = 2, color = "red") +
  ggtitle("Winning Percentage vs Games Played") +
  expand_limits(x = c(0,1))
} 

plot_winpercentage(teamperf, 100)
     
b'\n
Straight away we can see that there are some potential issues with our dataset.

For one thing, some countries have ceased to exist, either because they dissolved into multiple countries - for example the Soviet Union (URS), Yugoslavia (YUG) or Czechoslovakia (TCH), or because they united into one country - like it was the case with the German reunification of 1990. In the latter case, West Germany (FRG) and East Germany (GDR) unified into a single Germany (GER).

Another case was when a country would rename itself - for example from Zaire (ZAI) to Democratic Republic of the Congo (COD).

Here is a complete list of all the FIFA obsolete country codes which stood for countries and territories that no longer exist.

From our perspective, for the purposes of continuity we would like to consider the new countries as successors to (part of) the old ones, because it will allow us to take past performance into account instead of starting from scratch. The process is not 100% straightforward - for example, which of the six countries should we consider as a succesor to Yugoslavia? - but we will undergo a best effort approach.


# transform old country codes into new ones.
countryCodeMappings <- matrix(c(
  "FRG","GER",
  "TCH","CZE",
  "URS","RUS",
  "SCG","SRB",
  "ZAI","COD"
  ), ncol=2, byrow = TRUE)

for (i in 1:nrow(countryCodeMappings)) {
  teamperf
name == countryCodeMappings[i,1]] <- countryCodeMappings[i,2]
  teamperf
opponentName == countryCodeMappings[i,1]] <- countryCodeMappings[i,2]
  
  matches
team1 == countryCodeMappings[i,1]] <- countryCodeMappings[i,2]
  matches
team2 == countryCodeMappings[i,1]] <- countryCodeMappings[i,2]
}
     

# let's run the win percentage graph again
plot_winpercentage(teamperf, 100)
     
b'\n
Since our model will predict match results, it would be useful to also look at the distribution of match scores as well as total number of goals scored per game.


# what is the occurence frequency for match scores?

scorefreq <- matches %>%
  group_by(team1Score, team2Score) %>%
  summarise(
    n = n(),
    freq = n / nrow(matches)
  ) %>%
  ungroup() %>%
  mutate(
    scoretext = paste(team1Score,"-",team2Score)
  ) %>%
  arrange(desc(freq)) 

  head(scorefreq, 20)
     
team1Score	team2Score	n	freq	scoretext
1	1	1279	0.09601381	1 - 1
1	0	1168	0.08768111	1 - 0
0	1	1164	0.08738083	0 - 1
0	0	1115	0.08370242	0 - 0
0	2	881	0.06613618	0 - 2
2	0	845	0.06343368	2 - 0
1	2	840	0.06305833	1 - 2
2	1	821	0.06163201	2 - 1
3	0	543	0.04076271	3 - 0
0	3	490	0.03678403	0 - 3
2	2	458	0.03438180	2 - 2
1	3	393	0.02950229	1 - 3
3	1	376	0.02822611	3 - 1
4	0	303	0.02274604	4 - 0
0	4	279	0.02094437	0 - 4
1	4	207	0.01553937	1 - 4
3	2	205	0.01538924	3 - 2
2	3	199	0.01493882	2 - 3
4	1	187	0.01403799	4 - 1
5	0	152	0.01141055	5 - 0

# distribution of goals scored per match
gsfreq <- matches %>%
  group_by(gs = team1Score + team2Score) %>%
  summarise(
    n = n(),
    freq = n / nrow(matches)
  ) %>%
  ungroup() %>%
  arrange(desc(freq)) 

head(gsfreq, 10)

gsfreq %>%
  filter(freq >= 0.01) %>%
  ggplot(mapping = aes(x = gs, y = freq)) + geom_bar(stat = "identity") + ggtitle("Goals scored per match distribution")
     
gs	n	freq
2	3005	0.225583665
3	2694	0.202237069
1	2332	0.175061932
4	1809	0.135800616
0	1115	0.083702425
5	1100	0.082576383
6	597	0.044816455
7	326	0.024472637
8	153	0.011485624
9	84	0.006305833
b'\n

# distribution of goal differential
gdfreq <- matches %>%
  group_by(gd = team1Score - team2Score) %>%
  summarise(
    n = n(),
    freq = n / nrow(matches)
  ) %>%
  ungroup() %>%
  arrange(gd) 

head(gdfreq %>% filter(abs(gd)<=4), 10)

gdfreq %>%
  filter(abs(gd)<=4) %>%
  ggplot(mapping = aes(x = gd, y = freq)) + geom_bar(stat = "identity") + ggtitle("Goal differential distribution")
     
gd	n	freq
-4	369	0.02770062
-3	740	0.05555139
-2	1380	0.10359583
-1	2233	0.16763006
0	2932	0.22010360
1	2229	0.16732978
2	1321	0.09916673
3	764	0.05735305
4	395	0.02965243
b'\n
Outliers
Since our aim is to predict the goal differential (win margin) between the two teams in a match, we'd like to get rid of outliers - values which are far away at the end of the spectrum of possible values for this variable. The reason is that outliers can drastically change the results of the data analysis and statistical modeling. Outliers increase the error variance, reduce the power of statistical tests, and ultimately they can bias or influence estimates.

So let's deal with all matches where the goal differential is greater than 7.

First, let's verify how many of those we've got in the first place:


# how many outliers do we have?
temp <- matches %>% dplyr::filter(abs(team1Score - team2Score) > 7)
head(temp)
paste(nrow(temp), "matches, or", (nrow(temp)/nrow(matches)*100), "% of total.")
     
date	team1	team1Text	team2	team2Text	venue	IdCupSeason	CupName	team1Score	team2Score	...	resText	team1PenScore	team2PenScore	match_id	team1Home	team2Home	neutralVenue	friendly	qualifier	finaltourn
1950-07-02	URU	Uruguay	BOL	Bolivia	Belo Horizonte, Brazil	7	FIFA competition team final	8	0	...	8-0	NA	NA	18	FALSE	FALSE	TRUE	FALSE	FALSE	TRUE
1952-07-15	YUG	Yugoslavia	IND	India	Helsinki, Finland	197058	FIFA competition team final	10	1	...	10-1	NA	NA	198	FALSE	FALSE	TRUE	FALSE	FALSE	TRUE
1952-07-16	ITA	Italy	USA	USA	Tampere, Finland	197058	FIFA competition team final	8	0	...	8-0	NA	NA	202	FALSE	FALSE	TRUE	FALSE	FALSE	TRUE
1953-07-19	HAI	Haiti	MEX	Mexico	Mexico City, Mexico	8	FIFA competition team qualification	0	8	...	8-0	NA	NA	315	FALSE	TRUE	FALSE	FALSE	TRUE	FALSE
1953-09-26	POR	Portugal	AUT	Austria	Vienna, Austria	8	FIFA competition team qualification	1	9	...	9-1	NA	NA	322	FALSE	TRUE	FALSE	FALSE	TRUE	FALSE
1953-12-17	FRA	France	LUX	Luxembourg	Paris, France	8	FIFA competition team qualification	8	0	...	8-0	NA	NA	341	TRUE	FALSE	FALSE	FALSE	TRUE	FALSE
'199 matches, or 1.49388184070265 % of total.'
Not bad - only a very small percentage of games would be outliers as per our definition.

NOTE: As a rule of thumb, any value which is out of range of the 5th and 95th percentile may be considered an outlier.

Let's deal with the outliers in teamperf by capping the goal differential to the [-7, +7] interval:


# get rid of all the outliers by capping the gd to [-7, +7]
teamperf
gd < -7] <- -7
teamperf
gd > +7] <- +7
     
Strength of opposition
We may also want to take into account the fact that teams play most of their matches against opponents from the same FIFA Confederation - for example, European teams play mostly against other UEFA members, while African teams face other CAF members for the most part. Only during final tournaments like the Olympics, the World Cup and the Confederations Cup will teams play official (non-friendly) matches against non-confederation opponents.

Since not all conferences are the same general strength, we can adjust our teamperf dataset to also include information about the conference the opponent belongs to. We can assign adjustment coefficients to each conference, in a similar way to how FIFA's World ranking algorithm accounts for regional strength.


# get information about the various FIFA confederations and the teams they contain
if(!file.exists("teams.csv")){
  tryCatch(download.file('https://raw.githubusercontent.com/neaorin/PredictTheWorldCup/master/input/teams.csv'
                         ,destfile="./teams.csv",method="auto"))
}

if(file.exists("teams.csv")) teams <- read_csv("teams.csv")

# confederations and adjustment coefficients for them
confederations <- as.data.frame(matrix(c(
  "UEFA","0.99",
  "CONMEBOL","1.00",
  "CONCACAF","0.85",
  "AFC","0.85",
  "CAF","0.85",
  "OFC","0.85"
), ncol=2, byrow = TRUE, dimnames = list(NULL, c("confederation","adjust"))), stringsAsFactors = FALSE)

confederations$confederation <- as.vector(confederations
adjust <- as.numeric(confederations$adjust)

# add a confederation coefficient for the opponent faced 
teamperf <- teamperf %>%
  dplyr::left_join(teams, by=c("opponentName" = "fifa_code")) %>%
  dplyr::left_join(confederations, by=c("confederation")) %>%
  dplyr::mutate(
    opponentConfederationCoefficient = adjust
  ) %>%
dplyr::select(match_id, date, name = name.x, opponentName, opponentConfederationCoefficient,  homeVenue, neutralVenue, gs, ga, gd, w, l, d, friendly, qualifier, finaltourn)

# set missing values to 1
teamperf$opponentConfederationCoefficient[is.na(teamperf$opponentConfederationCoefficient)] <- 1

     
Parsed with column specification:
cols(
  confederation = col_character(),
  name = col_character(),
  fifa_code = col_character(),
  ioc_code = col_character()
)
Feature Engineering
Now, let's calculate some lag features for each team which is about to play a game.

We'll look at the previous N games a team has played, up to the game in question, and we'll calculate the percentage of wins, draws, losses, as well as the goal differential, per game, for those past N games.

For example, taking N=10:

last10games_w_per = (number of wins in the past 10 games) / 10
last10games_d_per = (number of draws in the past 10 games) / 10
last10games_l_per = (number of losses in the past 10 games) / 10
last10games_gd_per = (goals scored - goals conceeded in the past 10 games) / 10 
We'll use three different values for N (10, 30 and 50) to capture short-, medium-, and long-term form.

We'll calculate those values for every team and every game in our dataset.

To model the strength of opposition faced, we'll use the same technique with respect to the opponentConfederationCoefficient values we introduced earlier.


# Let's calculate some lag features for each team which is about to play a game
# we'll take three windows: last 5 games, last 20 games, last 35 games.
# for each window we'll calculate some values

lagfn <- function(data, width) {
  return (rollapplyr(data, width = width + 1, FUN = sum, fill = NA, partial=TRUE) - data)
}

lagfn_per <- function(data, width) {
  return (lagfn(data, width) / width)
}

team_features <- teamperf %>%
  dplyr::arrange(name, date) %>%
  dplyr::group_by(name) %>%
  dplyr::mutate(
    last10games_w_per = lagfn_per(w, 10),
    last30games_w_per = lagfn_per(w, 30),
    last50games_w_per = lagfn_per(w, 50),

    last10games_l_per = lagfn_per(l, 10),
    last30games_l_per = lagfn_per(l, 30),
    last50games_l_per = lagfn_per(l, 50),

    last10games_d_per = lagfn_per(d, 10),
    last30games_d_per = lagfn_per(d, 30),
    last50games_d_per = lagfn_per(d, 50),
            
    last10games_gd_per = lagfn_per(gd, 10),
    last30games_gd_per = lagfn_per(gd, 30),
    last50games_gd_per = lagfn_per(gd, 50),
      
    last10games_opp_cc_per = lagfn_per(opponentConfederationCoefficient, 10),
    last30games_opp_cc_per = lagfn_per(opponentConfederationCoefficient, 30),
    last50games_opp_cc_per = lagfn_per(opponentConfederationCoefficient, 50)

  ) %>%
  dplyr::select (
    match_id, date, name, opponentName, gs, ga,
    w, last10games_w_per, last30games_w_per, last50games_w_per,
    l, last10games_l_per, last30games_l_per, last50games_l_per,
    d, last10games_d_per, last30games_d_per, last50games_d_per,
    gd, last10games_gd_per, last30games_gd_per, last50games_gd_per,
    opponentConfederationCoefficient, last10games_opp_cc_per, last30games_opp_cc_per, last50games_opp_cc_per

          ) %>%
  dplyr::ungroup()

head((team_features %>% dplyr::filter(name == "BRA" & date >= '1970-01-01')), n = 20)
summary(team_features)
     
match_id	date	name	opponentName	gs	ga	w	last10games_w_per	last30games_w_per	last50games_w_per	...	last30games_d_per	last50games_d_per	gd	last10games_gd_per	last30games_gd_per	last50games_gd_per	opponentConfederationCoefficient	last10games_opp_cc_per	last30games_opp_cc_per	last50games_opp_cc_per
3484	1970-06-03	BRA	CZE	4	1	TRUE	0.6	0.5666667	0.60	...	0.1666667	0.18	3	1.8	1.1000000	1.02	0.99	0.968	0.9716667	0.9818
3492	1970-06-07	BRA	ENG	1	0	TRUE	0.7	0.5666667	0.60	...	0.1666667	0.18	1	2.3	1.1666667	1.06	0.99	0.968	0.9713333	0.9816
3496	1970-06-10	BRA	ROU	3	2	TRUE	0.8	0.5666667	0.62	...	0.1666667	0.18	1	2.5	1.0333333	1.14	0.99	0.968	0.9710000	0.9814
3504	1970-06-14	BRA	PER	4	2	TRUE	0.9	0.6000000	0.64	...	0.1666667	0.16	2	2.6	1.1333333	1.16	1.00	0.982	0.9710000	0.9812
3506	1970-06-17	BRA	URU	3	1	TRUE	1.0	0.6000000	0.64	...	0.1666667	0.16	2	2.8	1.1333333	1.18	1.00	0.997	0.9760000	0.9812
3509	1970-06-21	BRA	ITA	4	1	TRUE	1.0	0.6333333	0.64	...	0.1333333	0.16	3	2.8	1.2000000	1.16	0.99	0.997	0.9763333	0.9814
4013	1972-08-27	BRA	DEN	2	3	FALSE	1.0	0.6333333	0.66	...	0.1333333	0.14	-1	2.6	1.2666667	1.22	0.99	0.996	0.9763333	0.9814
4019	1972-08-29	BRA	HUN	2	2	FALSE	0.9	0.6000000	0.64	...	0.1333333	0.14	0	2.2	1.1666667	1.16	0.99	0.995	0.9763333	0.9814
4027	1972-08-31	BRA	IRN	0	1	FALSE	0.8	0.5666667	0.62	...	0.1666667	0.16	-1	1.8	1.1000000	1.14	0.85	0.994	0.9760000	0.9814
4680	1974-06-13	BRA	YUG	0	0	FALSE	0.7	0.5333333	0.60	...	0.1666667	0.16	0	1.1	1.0000000	1.06	1.00	0.979	0.9713333	0.9786
4689	1974-06-18	BRA	SCO	0	0	FALSE	0.6	0.5000000	0.58	...	0.2000000	0.18	0	1.0	0.9666667	1.00	0.99	0.979	0.9713333	0.9788
4698	1974-06-22	BRA	COD	3	0	TRUE	0.5	0.4666667	0.58	...	0.2333333	0.18	3	0.7	0.8333333	1.00	0.85	0.979	0.9710000	0.9786
4706	1974-06-26	BRA	GDR	1	0	TRUE	0.5	0.5000000	0.58	...	0.2333333	0.18	1	0.9	1.0000000	1.00	1.00	0.965	0.9660000	0.9756
4708	1974-06-30	BRA	ARG	2	1	TRUE	0.5	0.5333333	0.58	...	0.2333333	0.18	1	0.9	1.1333333	0.98	1.00	0.966	0.9660000	0.9756
4713	1974-07-03	BRA	NED	0	2	FALSE	0.5	0.5666667	0.58	...	0.2000000	0.18	-2	0.8	1.1666667	0.96	0.99	0.966	0.9660000	0.9756
4716	1974-07-06	BRA	POL	0	1	FALSE	0.4	0.5666667	0.56	...	0.2000000	0.18	-1	0.4	1.1333333	0.86	0.99	0.965	0.9656667	0.9754
5156	1975-07-31	BRA	VEN	4	0	TRUE	0.3	0.5666667	0.56	...	0.1666667	0.16	4	0.0	1.1000000	0.84	1.00	0.965	0.9703333	0.9752
5158	1975-08-06	BRA	ARG	2	1	TRUE	0.4	0.5666667	0.56	...	0.1666667	0.16	1	0.5	1.1000000	0.90	1.00	0.966	0.9753333	0.9752
5163	1975-08-13	BRA	VEN	6	0	TRUE	0.5	0.6000000	0.58	...	0.1666667	0.16	6	0.6	1.1666667	0.98	1.00	0.967	0.9756667	0.9752
5165	1975-08-16	BRA	ARG	1	0	TRUE	0.6	0.6000000	0.58	...	0.1666667	0.16	1	1.3	1.3000000	1.06	1.00	0.982	0.9760000	0.9752
    match_id          date                         name          
 Min.   :    1   Min.   :1950-03-08 00:00:00   Length:26916      
 1st Qu.: 9757   1st Qu.:1987-11-15 00:00:00   Class :character  
 Median :16789   Median :1999-10-10 00:00:00   Mode  :character  
 Mean   :16679   Mean   :1996-05-26 23:07:59                     
 3rd Qu.:23953   3rd Qu.:2007-10-28 00:00:00                     
 Max.   :31810   Max.   :2017-11-16 00:00:00                     
 opponentName             gs              ga             w          
 Length:26916       Min.   : 0.00   Min.   : 0.000   Mode :logical  
 Class :character   1st Qu.: 0.00   1st Qu.: 0.000   FALSE:16464    
 Mode  :character   Median : 1.00   Median : 1.000   TRUE :10452    
                    Mean   : 1.43   Mean   : 1.447                  
                    3rd Qu.: 2.00   3rd Qu.: 2.000                  
                    Max.   :31.00   Max.   :31.000                  
 last10games_w_per last30games_w_per last50games_w_per     l          
 Min.   :0.0000    Min.   :0.0000    Min.   :0.0000    Mode :logical  
 1st Qu.:0.2000    1st Qu.:0.2000    1st Qu.:0.1600    FALSE:16374    
 Median :0.4000    Median :0.3667    Median :0.3600    TRUE :10542    
 Mean   :0.3755    Mean   :0.3520    Mean   :0.3296                   
 3rd Qu.:0.5000    3rd Qu.:0.5000    3rd Qu.:0.4800                   
 Max.   :1.0000    Max.   :0.8667    Max.   :0.8000                   
 last10games_l_per last30games_l_per last50games_l_per     d          
 Min.   :0.0000    Min.   :0.0000    Min.   :0.0000    Mode :logical  
 1st Qu.:0.2000    1st Qu.:0.2000    1st Qu.:0.2000    FALSE:20994    
 Median :0.3000    Median :0.3000    Median :0.2800    TRUE :5922     
 Mean   :0.3693    Mean   :0.3344    Mean   :0.3036                   
 3rd Qu.:0.5000    3rd Qu.:0.4333    3rd Qu.:0.4000                   
 Max.   :1.0000    Max.   :1.0000    Max.   :1.0000                   
 last10games_d_per last30games_d_per last50games_d_per       gd          
 Min.   :0.0000    Min.   :0.0000    Min.   :0.0000    Min.   :-7.00000  
 1st Qu.:0.1000    1st Qu.:0.1333    1st Qu.:0.1000    1st Qu.:-1.00000  
 Median :0.2000    Median :0.2000    Median :0.2000    Median : 0.00000  
 Mean   :0.2108    Mean   :0.1948    Mean   :0.1806    Mean   :-0.01341  
 3rd Qu.:0.3000    3rd Qu.:0.2667    3rd Qu.:0.2600    3rd Qu.: 1.00000  
 Max.   :0.9000    Max.   :0.5333    Max.   :0.4400    Max.   : 7.00000  
 last10games_gd_per last30games_gd_per last50games_gd_per
 Min.   :-6.50000   Min.   :-4.90000   Min.   :-4.12000  
 1st Qu.:-0.60000   1st Qu.:-0.36667   1st Qu.:-0.30000  
 Median : 0.10000   Median : 0.13333   Median : 0.12000  
 Mean   : 0.01643   Mean   : 0.05403   Mean   : 0.08073  
 3rd Qu.: 0.80000   3rd Qu.: 0.66667   3rd Qu.: 0.62000  
 Max.   : 4.50000   Max.   : 2.76667   Max.   : 2.44000  
 opponentConfederationCoefficient last10games_opp_cc_per last30games_opp_cc_per
 Min.   :0.850                    Min.   :0.0000         Min.   :0.0000        
 1st Qu.:0.850                    1st Qu.:0.8500         1st Qu.:0.8500        
 Median :0.850                    Median :0.8790         Median :0.8650        
 Mean   :0.916                    Mean   :0.8762         Mean   :0.8094        
 3rd Qu.:0.990                    3rd Qu.:0.9900         3rd Qu.:0.9810        
 Max.   :1.000                    Max.   :1.0000         Max.   :1.0000        
 last50games_opp_cc_per
 Min.   :0.0000        
 1st Qu.:0.6120        
 Median :0.8590        
 Mean   :0.7487        
 3rd Qu.:0.9762        
 Max.   :1.0000        
Now that we have built a series of team-specific features, we need to fold them back into match-specific features.

We will then have a set of features for both teams about to face each other.


# fold per-team features into per-match features
match_features <- matches %>%
  left_join(team_features, by=c("match_id", "team1" = "name")) %>%
  left_join(team_features, by=c("match_id", "team2" = "name"), suffix=c(".t1",".t2")) %>%
  dplyr::select(
    date, match_id, team1, team2, team1Home, team2Home, neutralVenue, team1Score, team2Score, friendly, qualifier, finaltourn,
    last10games_w_per.t1,
    last30games_w_per.t1,
    last50games_w_per.t1,
    last10games_l_per.t1,
    last30games_l_per.t1,
    last50games_l_per.t1,
    last10games_d_per.t1,
    last30games_d_per.t1,
    last50games_d_per.t1,
    last10games_gd_per.t1, 
    last30games_gd_per.t1,
    last50games_gd_per.t1,
    last10games_opp_cc_per.t1, 
    last30games_opp_cc_per.t1, 
    last50games_opp_cc_per.t1,
    last10games_w_per.t2,
    last30games_w_per.t2,
    last50games_w_per.t2,
    last10games_l_per.t2,
    last30games_l_per.t2,
    last50games_l_per.t2,
    last10games_d_per.t2,
    last30games_d_per.t2,
    last50games_d_per.t2,
    last10games_gd_per.t2, 
    last30games_gd_per.t2,
    last50games_gd_per.t2,
    last10games_opp_cc_per.t2, 
    last30games_opp_cc_per.t2, 
    last50games_opp_cc_per.t2,
    outcome = gd.t1
  )

head(match_features)
names(match_features)
     
date	match_id	team1	team2	team1Home	team2Home	neutralVenue	team1Score	team2Score	friendly	...	last10games_d_per.t2	last30games_d_per.t2	last50games_d_per.t2	last10games_gd_per.t2	last30games_gd_per.t2	last50games_gd_per.t2	last10games_opp_cc_per.t2	last30games_opp_cc_per.t2	last50games_opp_cc_per.t2	outcome
1950-03-08	1	NIR	WAL	FALSE	TRUE	FALSE	0	0	FALSE	...	0	0	0	0.0	0.0000000	0.00	0.000	0.000	0.0000	0
1950-04-02	2	ESP	POR	TRUE	FALSE	FALSE	5	1	FALSE	...	0	0	0	0.0	0.0000000	0.00	0.000	0.000	0.0000	4
1950-04-09	3	ESP	POR	FALSE	TRUE	FALSE	2	2	FALSE	...	0	0	0	-0.4	-0.1333333	-0.08	0.099	0.033	0.0198	0
1950-04-15	4	SCO	ENG	TRUE	FALSE	FALSE	0	1	FALSE	...	0	0	0	0.0	0.0000000	0.00	0.000	0.000	0.0000	-1
1950-06-24	5	MEX	BRA	FALSE	TRUE	FALSE	0	4	FALSE	...	0	0	0	0.0	0.0000000	0.00	0.000	0.000	0.0000	-4
1950-06-25	7	ESP	USA	FALSE	FALSE	TRUE	3	1	FALSE	...	0	0	0	0.0	0.0000000	0.00	0.000	0.000	0.0000	2
'date'
'match_id'
'team1'
'team2'
'team1Home'
'team2Home'
'neutralVenue'
'team1Score'
'team2Score'
'friendly'
'qualifier'
'finaltourn'
'last10games_w_per.t1'
'last30games_w_per.t1'
'last50games_w_per.t1'
'last10games_l_per.t1'
'last30games_l_per.t1'
'last50games_l_per.t1'
'last10games_d_per.t1'
'last30games_d_per.t1'
'last50games_d_per.t1'
'last10games_gd_per.t1'
'last30games_gd_per.t1'
'last50games_gd_per.t1'
'last10games_opp_cc_per.t1'
'last30games_opp_cc_per.t1'
'last50games_opp_cc_per.t1'
'last10games_w_per.t2'
'last30games_w_per.t2'
'last50games_w_per.t2'
'last10games_l_per.t2'
'last30games_l_per.t2'
'last50games_l_per.t2'
'last10games_d_per.t2'
'last30games_d_per.t2'
'last50games_d_per.t2'
'last10games_gd_per.t2'
'last30games_gd_per.t2'
'last50games_gd_per.t2'
'last10games_opp_cc_per.t2'
'last30games_opp_cc_per.t2'
'last50games_opp_cc_per.t2'
'outcome'
We're also going to get rid of some columns which should not be used in training - specifically team1Score and team2Score. We will use the new outcome column instead - which is the difference between team1Score and team2Score.


# drop all non-interesting columns, and those which should not be supplied for new data (like scores)
match_features <- match_features %>%
  dplyr::select(-c(match_id,team1Score,team2Score))

head(match_features)
names(match_features)
     
date	team1	team2	team1Home	team2Home	neutralVenue	friendly	qualifier	finaltourn	last10games_w_per.t1	...	last10games_d_per.t2	last30games_d_per.t2	last50games_d_per.t2	last10games_gd_per.t2	last30games_gd_per.t2	last50games_gd_per.t2	last10games_opp_cc_per.t2	last30games_opp_cc_per.t2	last50games_opp_cc_per.t2	outcome
1950-03-08	NIR	WAL	FALSE	TRUE	FALSE	FALSE	TRUE	FALSE	0.0	...	0	0	0	0.0	0.0000000	0.00	0.000	0.000	0.0000	0
1950-04-02	ESP	POR	TRUE	FALSE	FALSE	FALSE	TRUE	FALSE	0.0	...	0	0	0	0.0	0.0000000	0.00	0.000	0.000	0.0000	4
1950-04-09	ESP	POR	FALSE	TRUE	FALSE	FALSE	TRUE	FALSE	0.1	...	0	0	0	-0.4	-0.1333333	-0.08	0.099	0.033	0.0198	0
1950-04-15	SCO	ENG	TRUE	FALSE	FALSE	FALSE	TRUE	FALSE	0.0	...	0	0	0	0.0	0.0000000	0.00	0.000	0.000	0.0000	-1
1950-06-24	MEX	BRA	FALSE	TRUE	FALSE	FALSE	FALSE	TRUE	0.0	...	0	0	0	0.0	0.0000000	0.00	0.000	0.000	0.0000	-4
1950-06-25	ESP	USA	FALSE	FALSE	TRUE	FALSE	FALSE	TRUE	0.1	...	0	0	0	0.0	0.0000000	0.00	0.000	0.000	0.0000	2
'date'
'team1'
'team2'
'team1Home'
'team2Home'
'neutralVenue'
'friendly'
'qualifier'
'finaltourn'
'last10games_w_per.t1'
'last30games_w_per.t1'
'last50games_w_per.t1'
'last10games_l_per.t1'
'last30games_l_per.t1'
'last50games_l_per.t1'
'last10games_d_per.t1'
'last30games_d_per.t1'
'last50games_d_per.t1'
'last10games_gd_per.t1'
'last30games_gd_per.t1'
'last50games_gd_per.t1'
'last10games_opp_cc_per.t1'
'last30games_opp_cc_per.t1'
'last50games_opp_cc_per.t1'
'last10games_w_per.t2'
'last30games_w_per.t2'
'last50games_w_per.t2'
'last10games_l_per.t2'
'last30games_l_per.t2'
'last50games_l_per.t2'
'last10games_d_per.t2'
'last30games_d_per.t2'
'last50games_d_per.t2'
'last10games_gd_per.t2'
'last30games_gd_per.t2'
'last50games_gd_per.t2'
'last10games_opp_cc_per.t2'
'last30games_opp_cc_per.t2'
'last50games_opp_cc_per.t2'
'outcome'
Let's also have a look at how correlated our numeric features are:


# correlation matrix
cormatrix <- cor(match_features %>% dplyr::select(-c(date, team1, team2, team1Home, team2Home, neutralVenue, friendly, qualifier, finaltourn)) )
corrplot(cormatrix, type = "upper", order = "original", tl.col = "black", tl.srt = 45, tl.cex = 0.5)
     
b'\n
Well, we don't have a lot of surprises here. It looks like goal differential lag values are strongly correlated with win and loss lag values (as expected).

Also, we have strong positive correlations between lag features measuring the same metric on different lag windows. For example, last10games_w_per.t1, last30games_w_per.t1 and last50games_w_per.t1 are correlated. Also unsurprisingly, the correlation between 10- and 50- window metrics are weaker than between 10- and 30-, or 30- and 50-. But it does seem to suggest that good teams generally keep being good over time, and bad teams keep being bad.

Last, none of our features has a correlation value (positive or negative) with our outcome that's much stronger than others.

Training our first model
The next step is to create a training formula for our model - it is going to describe the features we want to use and the outcome we're trying to predict.


# create the training formula 
trainformula <- as.formula(paste('outcome',
                                 paste(names(match_features %>% dplyr::select(-c(date,team1,team2,outcome))),collapse=' + '),
                                 sep=' ~ '))
trainformula
     
outcome ~ team1Home + team2Home + neutralVenue + friendly + qualifier + 
    finaltourn + last10games_w_per.t1 + last30games_w_per.t1 + 
    last50games_w_per.t1 + last10games_l_per.t1 + last30games_l_per.t1 + 
    last50games_l_per.t1 + last10games_d_per.t1 + last30games_d_per.t1 + 
    last50games_d_per.t1 + last10games_gd_per.t1 + last30games_gd_per.t1 + 
    last50games_gd_per.t1 + last10games_opp_cc_per.t1 + last30games_opp_cc_per.t1 + 
    last50games_opp_cc_per.t1 + last10games_w_per.t2 + last30games_w_per.t2 + 
    last50games_w_per.t2 + last10games_l_per.t2 + last30games_l_per.t2 + 
    last50games_l_per.t2 + last10games_d_per.t2 + last30games_d_per.t2 + 
    last50games_d_per.t2 + last10games_gd_per.t2 + last30games_gd_per.t2 + 
    last50games_gd_per.t2 + last10games_opp_cc_per.t2 + last30games_opp_cc_per.t2 + 
    last50games_opp_cc_per.t2
We're going to split our match_features into a training and a testing dataset. We're going to be using the training data to fit our model, then we're going to use the testing data to evaluate its accuracy.

We're going to use the matches from 1960 - 2009 to train our model, and the matches from 2010 - present to validate it.

NOTE: Although we're going to skip this step for the tutorial, model cross validation) is an important step to verify how a model will respond to a new, unknown data set. We would be creating multiple "folds" of training and testing set combinations from our original data set, and validate each combination to obtain a more complete picture of our model's predictive power.


# training and testing datasets

data.train1 <- match_features %>% dplyr::filter(date < '2009/1/1')
data.test1 <- match_features %>% dplyr::filter(date >= '2009/1/1' & date <= '2015/1/1')

nrow(data.train1)
nrow(data.test1)
     
10812
1605
Now it's time to train our model.

Since we're going to train a model to predict a numeric value (goal differential), we have a wide choice of regression algorithms we could use:

linear regression
neural network regression
decision forest regression
boosted decision tree regression
etc.
Indeed we might decide to try several algorithms, with a variety of parameter combinations for each of them, in order to find the optimal model and training strategy.

For this tutorial we're going to use a random forest, an algorithm which grows multiple decision trees from the features presented to it, and has each individual tree "vote" on the outcome for each new input vector (or in other words, new match to predict). It's fast, fairly accurate, and it gives an unbiased estimate of the generalization error, which makes cross-validation unnecessary for this particular algorithm.

The R implementation of the random forest algorithm is available in the randomForest package.

We're going to tell the algorithm to grow 500 trees.

NOTE: The training process should take several minutes.


# train a random forest
model.randomForest1 <- randomForest::randomForest(trainformula, data = data.train1, 
                                                  importance = TRUE, ntree = 500)

summary(model.randomForest1)
     
                Length Class  Mode     
call                5  -none- call     
type                1  -none- character
predicted       10812  -none- numeric  
mse               500  -none- numeric  
rsq               500  -none- numeric  
oob.times       10812  -none- numeric  
importance         72  -none- numeric  
importanceSD       36  -none- numeric  
localImportance     0  -none- NULL     
proximity           0  -none- NULL     
ntree               1  -none- numeric  
mtry                1  -none- numeric  
forest             11  -none- list     
coefs               0  -none- NULL     
y               10812  -none- numeric  
test                0  -none- NULL     
inbag               0  -none- NULL     
terms               3  terms  call     
Model Evaluation
In order to understand the importance of our predictors to the predicted outcome, we can use some built-in functions from the randomForest package:


randomForest::importance(model.randomForest1, type=1)
randomForest::varImpPlot(model.randomForest1, type=1)
     
%IncMSE
team1Home	43.219776
team2Home	52.081486
neutralVenue	19.728239
friendly	0.000000
qualifier	8.695162
finaltourn	11.374801
last10games_w_per.t1	23.160897
last30games_w_per.t1	25.832558
last50games_w_per.t1	28.426781
last10games_l_per.t1	28.281955
last30games_l_per.t1	30.014322
last50games_l_per.t1	32.592522
last10games_d_per.t1	16.455230
last30games_d_per.t1	27.133504
last50games_d_per.t1	27.915199
last10games_gd_per.t1	33.355178
last30games_gd_per.t1	34.543821
last50games_gd_per.t1	34.815678
last10games_opp_cc_per.t1	30.927270
last30games_opp_cc_per.t1	34.794435
last50games_opp_cc_per.t1	36.175053
last10games_w_per.t2	24.056440
last30games_w_per.t2	29.834927
last50games_w_per.t2	31.284813
last10games_l_per.t2	25.447341
last30games_l_per.t2	26.045157
last50games_l_per.t2	31.715173
last10games_d_per.t2	15.652467
last30games_d_per.t2	20.752684
last50games_d_per.t2	23.616185
last10games_gd_per.t2	32.062712
last30games_gd_per.t2	35.095094
last50games_gd_per.t2	40.409490
last10games_opp_cc_per.t2	31.582733
last30games_opp_cc_per.t2	26.720990
last50games_opp_cc_per.t2	35.977968
b'\n
We can now expose our trained model to the test dataset, and calculate performance metrics.


data.pred.randomForest1 <- predict(model.randomForest1, data.test1, predict.all = TRUE)

metrics.randomForest1.mae <- Metrics::mae(data.test1
aggregate)
metrics.randomForest1.rmse <- Metrics::rmse(data.test1
aggregate)

paste("Mean Absolute Error:", metrics.randomForest1.mae)
paste("Root Mean Square Error:",metrics.randomForest1.rmse)

abs_error <- abs(data.test1
aggregate)
plot(abs_error, main="Mean Absolute Error") 
     
'Mean Absolute Error: 1.32775005192108'
'Root Mean Square Error: 1.73915204826423'
b'\n
Simulating the Tournament
With a trained model at our disposal, we can now run tournament simulations on it.

For example, let's take the qualified teams for the FIFA 2018 World Cup.


if(!file.exists("wc2018qualified.csv")){
    tryCatch(download.file('https://raw.githubusercontent.com/neaorin/PredictTheWorldCup/master/src/TournamentSim/wc2018qualified.csv'
                           ,destfile="./wc2018qualified.csv",method="auto"))
}
                
if(file.exists("wc2018qualified.csv")) qualified <- read_csv("wc2018qualified.csv")
     
Parsed with column specification:
cols(
  name = col_character(),
  draw = col_character()
)
We can now run the entire tournament 10,000 times, calling into the model to get a prediction for each match.

For performance reasons, we could generate all the the possible two-team combinations, then ask the model for predictions for each combination, and then store those predictions.

We can store the mean values, as well as the standard deviation of the predicted values from every one of our decision trees. This will allow us to simulate a more realistic distribution of results, for multiple iterations of the same match.


# get a list of possible matches to be played at the world cup

data.topredict <- expand.grid(team1 = qualified
name, stringsAsFactors = FALSE) %>% filter(team1 < team2)

temp <- teamperf %>%
  semi_join(qualified, by = c("name")) %>%
  group_by(name) %>%
  summarise(
    date = max(date)
  )

temp <- team_features %>%
  semi_join(temp, by = c("name", "date"))

# calculate the features for every possbile match

data.topredict <- data.topredict %>%
  left_join(temp, by = c("team1" = "name")) %>%
  left_join(temp, by = c("team2" = "name"), suffix = c(".t1", ".t2")) %>%
  dplyr::select(
    team1, team2,
    last10games_w_per.t1,
    last30games_w_per.t1,
    last50games_w_per.t1,
    last10games_l_per.t1,
    last30games_l_per.t1,
    last50games_l_per.t1,
    last10games_d_per.t1,
    last30games_d_per.t1,
    last50games_d_per.t1,
    last10games_gd_per.t1, 
    last30games_gd_per.t1,
    last50games_gd_per.t1,
    last10games_opp_cc_per.t1, 
    last30games_opp_cc_per.t1, 
    last50games_opp_cc_per.t1,
    last10games_w_per.t2,
    last30games_w_per.t2,
    last50games_w_per.t2,
    last10games_l_per.t2,
    last30games_l_per.t2,
    last50games_l_per.t2,
    last10games_d_per.t2,
    last30games_d_per.t2,
    last50games_d_per.t2,
    last10games_gd_per.t2, 
    last30games_gd_per.t2,
    last50games_gd_per.t2,
    last10games_opp_cc_per.t2, 
    last30games_opp_cc_per.t2, 
    last50games_opp_cc_per.t2      
  ) %>%
  mutate(
    date = as.POSIXct("2018-06-14"), 
    team1Home = (team1 == "RUS"), team2Home = (team2 == "RUS"), neutralVenue = !(team1Home | team2Home), 
    friendly = FALSE, qualifier = FALSE, finaltourn = TRUE
  )

head(data.topredict)
     
team1	team2	last10games_w_per.t1	last30games_w_per.t1	last50games_w_per.t1	last10games_l_per.t1	last30games_l_per.t1	last50games_l_per.t1	last10games_d_per.t1	last30games_d_per.t1	...	last10games_opp_cc_per.t2	last30games_opp_cc_per.t2	last50games_opp_cc_per.t2	date	team1Home	team2Home	neutralVenue	friendly	qualifier	finaltourn
KSA	RUS	0.4	0.4333333	0.42	0.4	0.3333333	0.32	0.2	0.2333333	...	0.948	0.976	0.9816	2018-06-14	FALSE	TRUE	FALSE	FALSE	FALSE	TRUE
EGY	RUS	0.6	0.7000000	0.70	0.2	0.2000000	0.22	0.2	0.1000000	...	0.948	0.976	0.9816	2018-06-14	FALSE	TRUE	FALSE	FALSE	FALSE	TRUE
POR	RUS	0.8	0.6333333	0.60	0.0	0.1000000	0.12	0.2	0.2666667	...	0.948	0.976	0.9816	2018-06-14	FALSE	TRUE	FALSE	FALSE	FALSE	TRUE
ESP	RUS	0.8	0.7000000	0.70	0.1	0.1666667	0.16	0.1	0.1333333	...	0.948	0.976	0.9816	2018-06-14	FALSE	TRUE	FALSE	FALSE	FALSE	TRUE
MAR	RUS	0.4	0.2666667	0.36	0.3	0.3000000	0.30	0.3	0.4333333	...	0.948	0.976	0.9816	2018-06-14	FALSE	TRUE	FALSE	FALSE	FALSE	TRUE
IRN	RUS	0.7	0.6000000	0.58	0.0	0.1333333	0.12	0.3	0.2666667	...	0.948	0.976	0.9816	2018-06-14	FALSE	TRUE	FALSE	FALSE	FALSE	TRUE
At this point, our data.topredict table contains all the possible two-team match combinations, with calculated features for each team.

We can now ask our model to predict outcomes for these matches:


# ask the model to predict our world cup matches
data.predicted <- predict(model.randomForest1, data.topredict, predict.all = TRUE)

head(data.predicted
aggregate)
     
1	-3	-0.2	1.000000e+00	-3.0000000	-1.0	-1.000000e+00	-3	-2	-1.00	-3.5	...	-2.0	-6.000000e-01	-5.000000e+00	-1.0	-0.75	-2.000000	6.938894e-18	-2.2	-1.0	0.25
2	-2	-1.0	1.400000e+00	0.3333333	0.4	-1.000000e+00	-1	-1	-1.25	0.0	...	1.4	-5.000000e-01	-1.000000e+00	0.5	-2.00	-0.750000	-4.000000e+00	1.6	2.5	-1.50
3	-1	-2.0	6.938894e-18	0.0000000	1.0	2.333333e+00	3	1	3.00	0.5	...	-3.0	-1.000000e+00	3.000000e+00	-1.0	-1.50	0.000000	-1.333333e+00	0.0	0.0	-1.50
4	1	-1.0	1.400000e+00	2.0000000	1.0	6.938894e-18	-3	-2	3.00	-1.0	...	1.0	-5.000000e-01	-4.500000e+00	-1.0	-1.50	0.000000	-2.000000e+00	-2.0	0.0	6.00
5	-3	-1.0	1.000000e+00	-2.7500000	2.0	-3.400000e+00	-3	-3	2.50	0.0	...	-3.0	-1.000000e+00	-2.081668e-17	-6.0	-2.00	-2.666667	6.938894e-18	-2.0	-1.0	0.25
6	1	-2.0	6.938894e-18	0.0000000	-2.0	-1.000000e+00	-3	-1	-1.25	0.5	...	1.4	6.938894e-18	-1.500000e+00	-0.8	-2.00	0.000000	3.000000e+00	0.0	0.0	-1.50
1-0.9398333333333332-0.33916666666666730.3300666666666674-0.08003333333333345-0.97456-0.362133333333333
So, for every game in our input dataset, we've got the individual predictions from every one of our decision trees, as well as the mean value of those predictions.

We're going to save the mean values, as well as the standard deviation of the 100 individual predictions. The standard deviation is a measure of how dispersed our values are; in other words, how close (or far away from) the mean the individual values are.


# calculate the standard deviation of the individual predictions of each match

data.predicted
individual, c(1), sd)

# keep only the interesting columns for running tournament simulations
data.staticpred <- data.topredict %>% 
  dplyr::select(team1, team2)

data.staticpred
aggregate
data.staticpred
sd

head(data.staticpred)
     
team1	team2	outcome	sd
KSA	RUS	-0.93983333	1.820216
EGY	RUS	-0.33916667	1.962173
POR	RUS	0.33006667	1.668364
ESP	RUS	-0.08003333	2.083721
MAR	RUS	-0.97450000	1.816624
IRN	RUS	-0.36213333	1.968706
We can use the mean and standard deviation values to pick an individual outcome for a match. For example, we can use the normal distribution in conjunction with R's rnorm function to pick an outcome for a match where we have obtained a predicted mean and standard deviation from the model.

For instance, let's assume we need to provide predicted outcomes for a Brazil vs Argentina match.


temp <- data.staticpred %>% dplyr::filter(team1 == "ARG" & team2 == "BRA")
temp
     
team1	team2	outcome	sd
ARG	BRA	-0.4193667	2.002721

set.seed(4342)
draw_threshold <- 0.4475

temp2 <- rnorm(100, temp
sd)
temp2

plot(round(temp2),xlab="Match Index",ylab="Goal Diff", main="ARG vs BRA, 100 simulated matches")
abline(h = 0, v = 0, col = "gray60")
abline(h = -0.4475, v = 0, col = "gray60", lty=3)
abline(h = +0.4475, v = 0, col = "gray60", lty=3)
mtext(c("BRA","Draw","ARG"),side=2,line=-3,at=c(-3,0,3),col= "red")

paste("ARG won", length(temp2[temp2 > +draw_threshold]), "matches.")
paste("BRA won", length(temp2[temp2 < -draw_threshold]), "matches.")
paste(length(temp2[temp2 >= -draw_threshold & temp2 <= +draw_threshold]), "matches drawn.")
     
-0.342462962128619
-0.16993004933908
3.12013276898305
-3.66527827261082
-2.19308168696971
-4.01591077619749
-0.668732970273266
3.17165535554102
-0.105009506738923
-0.423873111624488
-1.64637875426684
-0.137295317392057
-3.56541895652084
-2.62300910888646
-3.95188629798082
-1.47055995083686
-0.482062464122396
0.0982534028286994
1.72527267532365
3.65596070276114
-0.163345178029263
-0.92494081838041
1.25653051557013
0.698150412296971
1.8713996592451
-1.08157922345642
-1.22407209329059
2.08220505791513
0.0379906317067356
-1.81680253307121
-3.67929213670896
-1.59588534770788
-1.36711978819466
0.34382309371824
1.12534659706185
0.106410171507352
1.78069587075024
-1.67849535714188
-0.378728820590507
2.6208809135048
1.77753727206812
-1.80477958410629
-2.46916722344627
-1.06251641926751
-1.7091850191995
0.532175604602257
-2.91128725089548
0.408980828589186
-2.05091433698149
1.44080519401948
2.03503175911652
-1.21369389637285
1.44210995739254
-0.924422490374622
0.0453529440479914
-1.78488530273687
-0.749195795371025
-0.557137528086248
1.32882960506691
-1.8139561024053
-6.14827479089838
0.0114606148129078
1.97678429933238
0.874694160460164
1.03285450159118
-3.95262858279771
-0.656586079813424
-3.87992153596966
-0.602179531202975
-2.20492239700507
-0.591413603715395
0.0584095692556688
5.39739719582818
-0.0950180692038243
-1.98164543856218
2.02093434660847
-0.683960449481511
-1.4796487197539
0.959746807819605
-2.89102864888588
1.30971718925574
0.568157172213166
4.09645660731329
-2.20848850201793
-0.442366069796936
4.05917533147172
0.248560242268595
0.354727795463984
2.06374196330724
1.53008253869018
-0.815062225837368
-1.90948598413341
-2.38190323585531
1.64364651537957
-0.582857584769466
-1.71070372204549
-1.57317985304724
-3.24910199379244
2.21279962817777
2.40494310881284
'ARG won 32 matches.'
'BRA won 49 matches.'
'19 matches drawn.'
b'\n
The points in the above graph are individual outcomes (goal differentials) which we can round to the nearest integer, or to a predicted draw - in the case of values that are "close enough" to zero.

Let's have a look at the result distribution for the real-world matches between these two teams:


# real results of ARG vs BRA matches
temp <- as.vector(teamperf %>% filter(name == "ARG" & opponentName == "BRA") %>% dplyr::select(gd))

plot(temp$gd,xlab="Match Index",ylab="Goal Diff", main="ARG vs BRA, real matches")
abline(h = 0, v = 0, col = "gray60")
abline(h = -0.4475, v = 0, col = "gray60", lty=3)
abline(h = +0.4475, v = 0, col = "gray60", lty=3)
mtext(c("BRA","Draw","ARG"),side=2,line=-3,at=c(-3,0,3),col= "red")

paste("ARG won", nrow(temp %>% dplyr::filter(gd > 0)), "matches.")
paste("BRA won", nrow(temp %>% dplyr::filter(gd < 0)), "matches.")
paste(nrow(temp %>% dplyr::filter(gd == 0)), "matches drawn.")
     
'ARG won 11 matches.'
'BRA won 16 matches.'
'10 matches drawn.'
b'\n
For comparison, let's also generate a plot for outcomes of a more unbalanced match-up: Brazil vs Egypt.


set.seed(4342)

temp <- data.staticpred %>% dplyr::filter(team1 == "BRA" & team2 == "EGY")
temp

temp2 <- rnorm(100, temp
sd)

plot(round(temp2),xlab="Match Index",ylab="Goal Diff", main="BRA vs EGY, 100 simulated matches")
abline(h = 0, v = 0, col = "gray60")
abline(h = -0.4475, v = 0, col = "gray60", lty=3)
abline(h = +0.4475, v = 0, col = "gray60", lty=3)
mtext(c("EGY","Draw","BRA"),side=2,line=-3,at=c(-3,0,3), col="red")

paste("BRA won", length(temp2[temp2 > +draw_threshold]), "matches.")
paste("EGY won", length(temp2[temp2 < -draw_threshold]), "matches.")
paste(length(temp2[temp2 >= -draw_threshold & temp2 <= +draw_threshold]), "matches drawn.")
     
team1	team2	outcome	sd
BRA	EGY	1.091033	1.801582
'BRA won 65 matches.'
'EGY won 17 matches.'
'18 matches drawn.'
b'\n
As we can see, starting from a rather inconspicuous-looking prediction we can generate a number of possible match results which, while respecting a World Cup's surprising nature, is still in line with what experience tells us should happen.

Tournament Statistics
Now that we're armed with a way of generating multiple predictions for every possible game in the World Cup, we can write a rather straightforward program to run the tournament a large number of times - for example 10,000 iterations.

We can save our predictions to a CSV file and use it as input for the simulator program.


write_csv(data.staticpred, "wc2018staticPredictions.csv")
     
Now we can run the program. You simply need Python 3.x installed in order to run it.

Clone the GitHub repository, then cd to the correct folder, and run it.

Assuming a Windows machine, the steps you need to perform should look like the following, with Python 3 already installed:

git clone https://github.com/neaorin/PredictTheWorldCup.git
cd PredictTheWorldCup\src\TournamentSim
python simulateworldcup.py
On a Linux or Mac the steps should be similar.

The program will output a list of tournament winners (one for each iteration) to a simresults.csv file.

The R code that's going to run after this step will attempt to download a version of this file from the GitHub repository; however, if you ran the Python program yourself and would like to use your own simresults.csv file, you can simply upload it into this Azure Notebook library by using the Data / Upload... menu at the top of this notebook.

Once we have the results inside the simresults.csv file, we can load it up into R and see who won tournaments, and who didn't:


# Load the results of the simulation program

if(!file.exists("simresults.csv")){
    tryCatch(download.file('https://raw.githubusercontent.com/neaorin/PredictTheWorldCup/master/src/TournamentSim/simresults.csv'
                           ,destfile="./simresults.csv",method="auto"))
}
                
if(file.exists("simresults.csv")) simresults <- read_csv("simresults.csv")
    
head(simresults, 20)
     
Parsed with column specification:
cols(
  iteration = col_integer(),
  winner = col_character()
)
iteration	winner
1	SUI
2	PER
3	GER
4	BRA
5	GER
6	ESP
7	POR
8	ENG
9	POR
10	GER
11	MEX
12	SWE
13	FRA
14	ARG
15	GER
16	ESP
17	MEX
18	BEL
19	BRA
20	MEX

# plot the winners

winsperteam <- simresults %>%
  dplyr::group_by(winner) %>%
  dplyr::summarize(
    wins = n()
  ) %>%
  dplyr::arrange(desc(wins)) 

winsperteam
winner, levels = winsperteam
wins, decreasing = FALSE)])

ggplot(winsperteam, mapping = aes(x=winner, y=wins)) +
  geom_bar(stat="identity") +
  coord_flip() +
  geom_text(aes(label=paste(wins / 100, "%")), vjust=0.3, hjust=-0.1, size=2.1) +
  ggtitle("Tournament simulation winners (10,000 iterations)")
     
b'\n
Just for fun, let's calculate the odds to win the tournament predicted by our model:


# calculate the sports odds 

winsperteam
wins
writeLines(paste(winsperteam$winner, ": ",round(winsperteam$odds), " to 1\n"))
     
GER :  7  to 1

BRA :  9  to 1

ENG :  10  to 1

ESP :  12  to 1

SUI :  14  to 1

POR :  14  to 1

FRA :  16  to 1

RUS :  20  to 1

ARG :  22  to 1

DEN :  25  to 1

BEL :  27  to 1

NGA :  40  to 1

CRO :  45  to 1

COL :  49  to 1

MEX :  54  to 1

SWE :  75  to 1

SRB :  82  to 1

URU :  88  to 1

IRN :  98  to 1

CRC :  109  to 1

JPN :  112  to 1

POL :  135  to 1

PER :  149  to 1

EGY :  169  to 1

KOR :  256  to 1

ISL :  303  to 1

AUS :  417  to 1

SEN :  455  to 1

TUN :  455  to 1

MAR :  556  to 1

KSA :  909  to 1

PAN :  1111  to 1

And that's it! We now have a surefire way of making money by betting on sports!

(pretty sure I'm not the first person ever to say those words :P)

Sorin
