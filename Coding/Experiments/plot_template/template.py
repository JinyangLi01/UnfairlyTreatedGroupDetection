import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import PercentFormatter
import os

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


sns.set_palette("Paired")
# sns.set_palette("deep")
sns.set_context("poster", font_scale=2)
sns.set_style("whitegrid")
# sns.palplot(sns.color_palette("deep", 10))
# sns.palplot(sns.color_palette("Paired", 9))

line_style = ['o-', 's--', '^:', '-.p']
color = ['C0', 'C1', 'C2', 'C3', 'C4']
plt_title = ["BlueNile", "COMPAS", "Credit Card"]

label = ["Naive", "Optimized"]
line_width = 8
marker_size = 15
# f_size = (14, 10)

f_size = (14, 7)

def plot_runtime(input_file):
   fig, ax = plt.subplots(1, 1, figsize=f_size)
   delim = '\t'
   with open(input_file) as f:
      lines = [line.rstrip('\n') for line in f]

   bound = []
   naive = []
   opt = []

   for line in lines[1:]:
      bound.append(float(line.split(delim)[0].strip()))
      naive_time = float(line.split(delim)[2].strip())
      if naive_time > 0:
         naive.append(naive_time)
      opt.append(float(line.split(delim)[3].strip()))

   plt.plot(bound[0:len(naive)], naive, line_style[0], color=color[0], label=label[0], linewidth=line_width,
          markersize=marker_size)
   plt.plot(bound, opt, line_style[1], color=color[1], label=label[1], linewidth=line_width,
             markersize=marker_size)

   plt.xlabel('Bound')
   plt.ylabel('Time [sec]')
   plt.legend(loc='best')
   plt.grid(True)

   fig = plt.gcf()
   plt.savefig(input_file + '_plot.pdf', bbox_inches='tight')
   plt.close()


