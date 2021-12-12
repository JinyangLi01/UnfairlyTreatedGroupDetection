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

line_style = ['o-', 's--', '^:', '-.p']
color = ['C0', 'C1', 'C2', 'C3', 'C4']
plt_title = ["BlueNile", "COMPAS", "Credit Card"]

label = ["Optimized", "Naive"]
line_width = 8
marker_size = 15
# f_size = (14, 10)

f_size = (14, 10)

x_new = list()
x_naive = list()
execution_time1 = list()
execution_time2 = list()


input_path = r'num_attribute.txt'
input_file = open(input_path, "r")


# Using readlines()
Lines = input_file.readlines()

count = 0
# Strips the newline character
for line in Lines:
    if line == '\n':
        break
    if count < 2:
        count += 1
        continue
    count += 1
    if count < 10:
        items = line.strip().split(' ')
        x_new.append(int(items[0]))
        x_naive.append(int(items[0]))
        execution_time1.append(float(items[1]))
        execution_time2.append(float(items[2]))
    else:
        items = line.strip().split(' ')
        x_new.append(int(items[0]))
        execution_time1.append(float(items[1]))





fig, ax = plt.subplots(1, 1, figsize=f_size)
plt.plot(x_new, execution_time1, line_style[0], color=color[0], label=label[0], linewidth=line_width,
          markersize=marker_size)
plt.plot(x_naive, execution_time2, line_style[1], color=color[1], label=label[1], linewidth=line_width,
             markersize=marker_size)
plt.xlabel('Number of attributes')
plt.xticks([2, 4, 6, 8, 10, 12, 14, 16])
plt.ylabel('Execution time (s)')
plt.legend(loc='best')
plt.grid(True)
fig.tight_layout()
plt.savefig("num_att_time.png",
            bbox_inches='tight')
plt.show()
plt.close()




def thousands_formatter(x, pos):
    return int(x/1000)

#
#
# fig, ax = plt.subplots(1, 1, figsize=f_size)
# plt.plot(x_new, num_calculation1, line_style[0], color=color[0], label=label[0], linewidth=line_width,
#           markersize=marker_size)
# plt.plot(x_naive, num_calculation2, line_style[1], color=color[1], label=label[1], linewidth=line_width,
#              markersize=marker_size)
# plt.xlabel('Number of attributes')
# plt.xticks([2, 4, 6, 8, 10, 12, 14, 16])
# plt.ylabel('Number of patterns visited (K)')
# ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
# plt.legend(loc='best')
# plt.grid(True)
# fig.tight_layout()
# plt.savefig("../../../../OutputData/LowAccDetection_withStopCond/CompasDataset/num_att_calculations.png",
#             bbox_inches='tight')
# plt.show()
# plt.close()
#
