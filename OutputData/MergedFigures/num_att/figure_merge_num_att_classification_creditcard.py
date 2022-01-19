import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


sns.set_palette("Paired")
# sns.set_palette("deep")
sns.set_context("poster", font_scale=2)
sns.set_style("whitegrid")
# sns.palplot(sns.color_palette("deep", 10))
# sns.palplot(sns.color_palette("Paired", 9))

line_style = ['o-', 's:', 'o-', 's:', 'o:', 's:', 'p:', '^:']
color = ['C0', 'C1', 'C6', 'C7', 'C4', 'C5', 'C6', 'C7', 'C8']
# plt_title = ["BlueNile", "Creditcard", "Credit Card"]

label = ["DUC (acc)", "Naive (acc)",
         "DUC (FPR)", "Naive (FPR)"]
line_width = 8
marker_size = 15
# f_size = (14, 10)

f_size = (14, 10)





def thousands_formatter(x, pos):
    return int(x/1000)




x_list = [[], [], [], []]
execution_time = [[], [], [], [], [], [], [], []]
num_patterns_visited = [[], [], [], [], [], [], [], []]

input_path = [r"../../LowAccDetection_2/CreditcardDataset/num_att.txt",
              r"../../General_2/CreditcardDataset/num_att.txt"]


input_file = open(input_path[0], "r")
# Using readlines()
Lines = input_file.readlines()
count = 0
# Strips the newline character
for line in Lines:
    count += 1
    if line == '\n':
        continue
    if count < 3:
        continue
    if count > 15 and count < 19:
        continue
    if count > 32:
        break
    items = line.strip().split(' ')
    if count < 9:
        x_list[0].append(int(items[0]))
        x_list[1].append(int(items[0]))
        execution_time[0].append(float(items[1]))
        execution_time[1].append(float(items[2]))
    elif count < 16:
        x_list[0].append(int(items[0]))
        execution_time[0].append(float(items[1]))
    elif count < 25:
        num_patterns_visited[0].append(float(items[1]))
        num_patterns_visited[1].append(float(items[2]))
    else:
        num_patterns_visited[0].append(float(items[1]))


input_file = open(input_path[1], "r")
# Using readlines()
Lines = input_file.readlines()
count = 0
for line in Lines:
    count += 1
    if line == '\n':
        continue
    if count < 3:
        continue
    if count > 12 and count < 16:
        continue
    if count > 25:
        break
    items = line.strip().split(' ')
    if count < 8:
        x_list[2].append(int(items[0]))
        x_list[3].append(int(items[0]))
        execution_time[2].append(float(items[1]))
        execution_time[3].append(float(items[2]))
    elif count < 13:
        x_list[2].append(int(items[0]))
        execution_time[2].append(float(items[1]))
    elif count < 21:
        num_patterns_visited[2].append(float(items[1]))
        num_patterns_visited[3].append(float(items[2]))
    else:
        num_patterns_visited[2].append(float(items[1]))




fig, ax = plt.subplots(1, 1, figsize=f_size)
for i in range(4):
    plt.plot(x_list[i], execution_time[i], line_style[i], color=color[i], label=label[i], linewidth=line_width,
          markersize=marker_size)

plt.xlabel('Number of attributes')
plt.xticks([2, 4, 6, 8, 10, 12, 14, 16])
plt.ylabel('Execution time (s)')
# plt.legend(fontsize='small', bbox_to_anchor=(-0.04, 1.1), loc='upper left')
plt.legend(fontsize='small', bbox_to_anchor=(0.42, 1.07), loc='upper left')
plt.grid(True)
fig.tight_layout()
plt.savefig("num_att_Creditcard_time.png",
            bbox_inches='tight')
plt.show()
plt.close()






fig, ax = plt.subplots(1, 1, figsize=f_size)
for i in range(4):
    plt.plot(x_list[i], num_patterns_visited[i], line_style[i], color=color[i], label=label[i], linewidth=line_width,
          markersize=marker_size)
plt.xlabel('Number of attributes')
plt.xticks([2, 4, 6, 8, 10, 12, 14, 16])
plt.ylabel('Number of patterns visited (K)')
ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
plt.legend(fontsize='small')
plt.grid(True)
fig.tight_layout()
plt.savefig("num_att_Creditcard_calculations.png",
            bbox_inches='tight')
plt.show()
plt.close()


plt.clf()


