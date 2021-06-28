# Title     : TODO
# Objective : TODO
# Created by: jinyangli
# Created on: 6/27/21

library(dplyr)
library(ggplot2)
library(survival)
library(ggfortify)

from truth_tables import PeekyReader, Person, table, is_race, count, vtable, hightable, vhightable
from csv import DictReader

people = []
with open("./cox-parsed.csv") as f:
  reader = PeekyReader(DictReader(f))
try:
  while True:
  p = Person(reader)
if p.valid:
  people.append(p)
except StopIteration:
  pass

pop = list(filter(lambda i: ((i.recidivist == True and i.lifetime <= 730) or
                             i.lifetime > 730), list(filter(lambda x: x.score_valid, people))))
recid = list(filter(lambda i: i.recidivist == True and i.lifetime <= 730, pop))
rset = set(recid)
surv = [i for i in pop if i not in rset]