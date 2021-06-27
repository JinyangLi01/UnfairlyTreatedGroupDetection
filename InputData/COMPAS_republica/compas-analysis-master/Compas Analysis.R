# Title     : TODO
# Objective : TODO
# Created by: jinyangli
# Created on: 6/27/21

library(dplyr)
library(ggplot2)
library(survival)
library(ggfortify)

data <- filter(filter(read.csv("./cox-parsed.csv"), score_text != "N/A"), end > start) %>%
  mutate(race_factor = factor(race,
                              labels = c("African-American", 
                                         "Asian",
                                         "Caucasian", 
                                         "Hispanic", 
                                         "Native American",
                                         "Other"))) %>%
  within(race_factor <- relevel(race_factor, ref = 3)) %>%
  mutate(score_factor = factor(score_text)) %>%
  within(score_factor <- relevel(score_factor, ref=2))

grp <- data[!duplicated(data$id),]
nrow(grp)



# write filtered data to file


write.csv(grp,"cox-parsed-filtered.csv")





violent_data <- filter(filter(read.csv("./cox-violent-parsed.csv"), score_text != "N/A"), end > start) %>%
  mutate(race_factor = factor(race,
                              labels = c("African-American", 
                                         "Asian",
                                         "Caucasian", 
                                         "Hispanic", 
                                         "Native American",
                                         "Other"))) %>%
  within(race_factor <- relevel(race_factor, ref = 3)) %>%
  mutate(score_factor = factor(score_text)) %>%
  within(score_factor <- relevel(score_factor, ref=2))


vf <- Surv(start, end, event, type="counting") ~ score_factor
vmodel <- coxph(vf, data=violent_data)
vgrp <- violent_data[!duplicated(violent_data$id),]
print(nrow(vgrp))
summary(vmodel)

write.csv(vgrp,"cox-violent-parsed-filtered.csv")

