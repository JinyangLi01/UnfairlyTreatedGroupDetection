
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def thousands_formatter(x, pos):
    return int(x/1000)


SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 20
plt.rc('figure', figsize=(7, 5.6))

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

Thc_list = list()
execution_time1 = list()
execution_time2 = list()
num_patterns1 = list()
num_patterns2 = list()

input_path = r'num_att.txt'
input_file = open(input_path, "r")
num_lines = len(Thc_list)

# Using readlines()
Lines = input_file.readlines()

# execution time
count = 0
# Strips the newline character
for line in Lines:
    if line == '\n':
        break
    if count < 1:
        count += 1
        continue
    count += 1
    items = line.strip().split(' ')
    Thc_list.append(int(items[0]))
    execution_time1.append(float(items[1]))
    execution_time2.append(float(items[2]))


while count < len(Lines):
    if Lines[count] != '\n':
        break
    else:
        count += 1
        continue

while count < len(Lines):
    if Lines[count] == '\n':
        break
    if Lines[count][0] == 'n':
        count += 1
        continue
    items = Lines[count].strip().split(' ')
    num_patterns1.append(items[1])
    num_patterns2.append(items[2])
    count += 1


print(num_patterns1)
print(num_patterns2)



plt.plot(Thc_list, execution_time1, label="optimized algorithm", color='blue', linewidth = 3.4)
plt.plot(Thc_list[:12], execution_time2[:12], label="naive algorithm", color='orange', linewidth = 3.4)

# plt.yscale('log')
plt.xlabel('number of attributes')
plt.ylabel('execution time (s)')
plt.xticks([3, 5, 7, 9, 10, 12, 14, 16])
plt.subplots_adjust(bottom=0.15, left=0.18)
plt.legend()
plt.savefig("num_att_time_naive_14.png")
plt.show()


# fig, ax = plt.subplots()
# plt.plot(Thc_list, num_patterns1, label="optimized algorithm", color='blue', linewidth = 3.4)
# plt.plot(Thc_list[:13], num_patterns2[:13], label="naive algorithm", color='orange', linewidth = 3.4)
# plt.xlabel('number of attributes')
# plt.ylabel('number of patterns visited (K)')
#
# # ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
#
#
# plt.xticks(Thc_list)
# plt.subplots_adjust(bottom=0.15, left=0.18)
# plt.legend()
# plt.savefig("num_att_calculations_naive_15.png")
# plt.show()
#
#
#
#
# print(num_patterns2[:13])
#
# fig, ax = plt.subplots()
# plt.plot(Thc_list, num_patterns1, label="optimized algorithm", color='blue', linewidth = 3.4)
# plt.plot(Thc_list[:13], num_patterns2[:13], label="naive algorithm", color='orange', linewidth = 3.4)
# plt.xlabel('number of attributes')
# plt.ylabel('number of patterns visited (K)')
# # plt.yticks()
# # ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
#
#
# plt.xticks(Thc_list)
# plt.subplots_adjust(bottom=0.15, left=0.18)
# plt.legend()
# plt.savefig("num_att_calculations_naive_15.png")
# plt.show()
#
# plt.close()
# plt.clf()
#
#
#
#
