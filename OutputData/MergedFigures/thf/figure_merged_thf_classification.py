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

line_style = ['o-', 's-', 'p-', '^-', 'o:', 's:', 'p:', '^:']
color = ['C3', 'C5', sns.xkcd_rgb["amber"], 'C9', 'C3', 'C5', sns.xkcd_rgb["amber"], 'C9']

label = ["Adult (acc)", "COMPAS (acc)", "Credit card (acc)", "Medical (acc)",
         "Adult (FPR)", "COMPAS (FPR)", "Credit card (FPR)", "Medical (FPR)"]
line_width = 8
marker_size = 15
# f_size = (14, 10)

f_size = (28, 10)





def thousands_formatter(x, pos):
    return int(x/1000)




x_list = list()
execution_time = [[], [], [], [], [], [], [], []]
num_patterns_visited = [[], [], [], [], [], [], [], []]

input_path = [r"../../LowAccDetection_2/AdultDataset/tha.txt",
              r"../../LowAccDetection_2/CompasDataset/tha.txt",
              r"../../LowAccDetection_2/CreditcardDataset/tha.txt",
              r"../../LowAccDetection_2/MedicalDataset/tha.txt",
              r"../../General_2/AdultDataset/tha.txt",
              r"../../General_2/CompasDataset/tha.txt",
              r"../../General_2/CreditcardDataset/tha.txt",
              r"../../General_2/MedicalDataset/tha.txt"]


for i in range(8):
    input_file = open(input_path[i], "r")
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
        if count > 8 and count < 12:
            continue
        if count > 17:
            break
        items = line.strip().split(' ')
        if count < 12:
            execution_time[i].append(float(items[1]))
        else:
            num_patterns_visited[i].append(float(items[1]))


x_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
fig, ax = plt.subplots(1, 1, figsize=f_size)
for i in range(8):
    plt.plot(x_list, execution_time[i], line_style[i], color=color[i], label=label[i], linewidth=line_width,
          markersize=marker_size)
# plt.plot(x_list, execution_time[0], line_style[0], color=color[0], label=label[0], linewidth=line_width,
#           markersize=marker_size)
# plt.plot(x_list, execution_time[4], line_style[4], color=color[4], label=label[4], linewidth=line_width,
#           markersize=marker_size)
plt.yscale('log')
plt.xlabel('Delta fairness value')
plt.xticks(x_list)
plt.ylabel('Execution time (s)')
# plt.yticks(([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]))
# plt.legend(bbox_to_anchor=(1.01, 1.02), loc='upper right', fontsize='x-small', ncol=2)
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
plt.grid(True)
fig.tight_layout()
plt.savefig("thf_time.png",
            bbox_inches='tight')
plt.show()
plt.close()



fig, ax = plt.subplots(1, 1, figsize=f_size)
for i in range(8):
    plt.plot(x_list, num_patterns_visited[i], line_style[i], color=color[i], label=label[i], linewidth=line_width,
          markersize=marker_size)
plt.xlabel('Delta fairness value')
plt.xticks(x_list)
plt.ylabel('Number of patterns visited (K)')
ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
plt.legend(bbox_to_anchor=(0.95, 0.9), loc='center', ncol=1)
plt.grid(True)
fig.tight_layout()
plt.savefig("thf_calculations.png",
            bbox_inches='tight')
plt.show()
plt.close()

plt.clf()


