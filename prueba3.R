packs <- c('dplyr', 'tidyr', 'skimr', 'dbscan')

# install.packages(packs)
library(dplyr)
library(tidyr)
library(skimr)
library(dbscan)

# -------------------------
# Read data
df <- read.csv('Player Per Game.csv')

# -------------------------
# Filter by last decade season
df <- df %>% 
  filter(season >= 2013)

unique(df$gs)

# -------------------------
# Select variables to use in clusterization
#          Age
#          experience
#          variables per game

vars <- c(7, 8, 13:ncol(df))
df_filter <- df[, vars]

# Delete variables of percentage
vars_toDelete <- grep("percent", names(df_filter))
df_filter <- df_filter[, -vars_toDelete]

str(df_filter)

# -------------------------
# Check quality of data
skimmed_data <- skim(df_filter)
skimmed_data

# Data is without missing values


# -------------------------
# Apply clusterization


# 1) DBSCAN
dbscan_results <- dbscan(df_filter, eps = 4, minPts = 10)
dbscan_results

kNNdistplot(df_filter, minPts = 100)

# 2) OPTICS
optics_result <- optics(df_filter, eps = 4, minPts = 10)
names(optics_result)
optics_result$predecessor



