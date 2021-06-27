# Title     : TODO
# Objective : TODO
# Created by: jinyangli
# Created on: 6/27/21

library(dplyr)
library(ggplot2)
f <- file.choose("./InputData/COMPAS_republica/compas-analysis-master/compas-scores-two-years.csv")
raw_data <- read.csv(f)
nr <- nrow(raw_data)
print(nr)
