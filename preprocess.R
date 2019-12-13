library(readr)
library(tidyverse)
library(readxl)
library(lubridate)
library(zoo)
raw_tweets_zip <-  read_csv(unzip("/Users/Kevin/Documents/TweetSentimentAnalysis/data/Sentiment-Analysis-Dataset.zip"))
# Stratified Sample of Twitter file for publishing to Github
set.seed(42)
# Retrieve indexes for partitioning
Twitter_raw_indexes <- createDataPartition(raw_tweets_zip$Sentiment, times = 1, p = 0.05, list = FALSE)
# Create dataframe
raw_tweets <- raw_tweets_zip[Twitter_raw_indexes,]
saveRDS(raw_tweets, file = "raw_tweets.rds")


raw_data_path <- "/Users/Kevin/Documents/FitBit/fitbit_interiew_project/DATA/raw_data_sheet.xlsx"
sheets <- raw_data_path %>%
        excel_sheets() %>% 
        set_names()
crude_oil <- read_excel(raw_data_path, sheet = sheets[2], skip = 2, col_types = c("date", "numeric", "numeric")) %>% 
        mutate("Date2" = as.Date(as.yearmon(Date, "%b-%Y"), frac = 1),
               "Month" = month(Date2),
               "Year" = year(Date2))
saveRDS(crude_oil, "crude_oil.rds")

conv_gasoline <- read_excel(raw_data_path, sheet = sheets[3], skip = 2, col_types = c("date", "numeric", "numeric")) %>% 
        mutate("Month" = month(Date), "Year" = year(Date))
saveRDS(conv_gasoline, "conv_gasoline.rds")
RBOB_gasoline <- read_excel(raw_data_path, sheet = sheets[4], skip = 2, col_types = c("date", "numeric")) %>% 
        mutate("Month" = month(Date), "Year" = year(Date))
saveRDS(RBOB_gasoline, "RBOB_gasoline.rds")
heating_oil <- read_excel(raw_data_path, sheet = sheets[5], skip = 2, col_types = c("date", "numeric")) %>% 
        mutate("Month" = month(Date), "Year" = year(Date))
saveRDS(heating_oil, "heating_oil.rds")
uls_diesel <- read_excel(raw_data_path, sheet = sheets[6], skip = 2, col_types = c("date", "numeric", "numeric", "numeric")) %>% 
        mutate("Month" = month(Date), "Year" = year(Date))
saveRDS(uls_diesel, "uls_diesel.rds")
jet <- read_excel(raw_data_path, sheet = sheets[7], skip = 2, col_types = c("date", "numeric")) %>% 
        mutate("Month" = month(Date), "Year" = year(Date))
saveRDS(jet, "jet.rds")
propane <- read_excel(raw_data_path, sheet = sheets[8], skip = 2, col_types = c("date", "numeric")) %>% 
        mutate("Month" = month(Date), "Year" = year(Date))
saveRDS(propane, "propane.rds")

