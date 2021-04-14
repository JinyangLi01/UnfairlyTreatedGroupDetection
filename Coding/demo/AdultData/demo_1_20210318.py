"""
This script is to do experiment on the threshold of minority group sizes.

two charts: running time, number of patterns checked
y axis: running time, number of patterns checked

x axis: Thc, from 1 to 1000

Other parameters:
CleanAdult2.csv
selected_attributes = ['age', 'education', 'marital-status', 'race', 'gender', 'workclass', 'relationship']
threshold of minority group accuracy: overall acc - 20


"""


import pandas as pd
from Algorithms import pattern_count
from Algorithms import WholeProcess_0_20201211 as wholeprocess
from Algorithms import NewAlg_0_20201128 as newalg
from Algorithms import NaiveAlg_0_20201111 as naivealg
from Algorithms import Predict_0_20210127 as predict
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from gooey import Gooey, GooeyParser


def ComparePatternSets(set1, set2):
    len1 = len(set1)
    len2 = len(set2)
    if len1 != len2:
        return False
    for p in set1:
        found = False
        for q in set2:
            if newalg.PatternEqual(p, q):
                found = True
                break
        if found is False:
            return False
    return True

def thousands_formatter(x, pos):
    return int(x/1000)



selected_attributes = ['age', 'education', 'marital-status', 'race', 'gender', 'workclass', 'relationship']
Thc_list = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 500, 1000]
original_data_file = "../../../InputData/AdultDataset/CleanAdult2.csv"
att_to_predict = 'income'
time_limit = 20*60
execution_time1 = list()
execution_time2 = list()
num_calculation1 = list()
num_calculation2 = list()
num_pattern_skipped_mis_c1 = list()
num_pattern_skipped_mis_c2 = list()
num_pattern_skipped_whole_c1 = list()
num_pattern_skipped_whole_c2 = list()
num_patterns_found = list()
patterns_found = list()
num_loops = 1



@Gooey(dump_build_config=True, program_name="DENOUNCER")
def main():
    desc = "DENOUNCER demo"
    file_help_msg = "Name of the file you want to process"

    my_cool_parser = GooeyParser(description=desc)

    my_cool_parser.add_argument(
        "Dataset", help="Choose from provided", widget="Listbox")
    my_cool_parser.add_argument(
        "", help="Upload dataset", widget="FileChooser")

    my_cool_parser.add_argument(
        "FileSaver", help=file_help_msg, widget="FileSaver")
    my_cool_parser.add_argument(
        "MultiFileChooser", nargs='*', help=file_help_msg, widget="MultiFileChooser")
    my_cool_parser.add_argument("directory", help="Directory to store output")

    my_cool_parser.add_argument('-d', '--duration', default=2,
                                type=int, help='Duration (in seconds) of the program output')
    my_cool_parser.add_argument('-s', '--cron-schedule', type=int,
                                help='datetime when the cron should begin', widget='DateChooser')
    my_cool_parser.add_argument('--cron-time',
                                help='datetime when the cron should begin', widget="TimeChooser")
    my_cool_parser.add_argument(
        "-c", "--showtime", action="store_true", help="display the countdown timer")
    my_cool_parser.add_argument(
        "-p", "--pause", action="store_true", help="Pause execution")
    my_cool_parser.add_argument('-v', '--verbose', action='count')
    my_cool_parser.add_argument(
        "-o", "--overwrite", action="store_true", help="Overwrite output file (if present)")
    my_cool_parser.add_argument(
        '-r', '--recursive', choices=['yes', 'no'], help='Recurse into subfolders')
    my_cool_parser.add_argument(
        "-w", "--writelog", default="writelogs", help="Dump output to local file")
    my_cool_parser.add_argument(
        "-e", "--error", action="store_true", help="Stop process on error (default: No)")
    verbosity = my_cool_parser.add_mutually_exclusive_group()
    verbosity.add_argument('-t', '--verbozze', dest='verbose',
                           action="store_true", help="Show more details")
    verbosity.add_argument('-q', '--quiet', dest='quiet',
                           action="store_true", help="Only output on error")

    my_cool_parser.parse_args()
    display_message()


less_attribute_data, mis_class_data, overall_acc = predict.PredictWithML(original_data_file,
                                                                         selected_attributes,
                                                                         att_to_predict)

thc = 50
print("\nthc = {}".format(thc))
tha = overall_acc - 0.2
t1, calculation1 = 0, 0


print("tha = {}, thc = {}".format(tha, thc))
pattern_with_low_accuracy1, calculation1_, t1_ = newalg.GraphTraverse(less_attribute_data,
                                                                      mis_class_data, tha,
                                                                          thc, time_limit)
print("newalg, time = {} s, num_calculation = {}".format(t1_, calculation1_), "\n", pattern_with_low_accuracy1)
t1 += t1_
calculation1 += calculation1_
result_cardinality = len(pattern_with_low_accuracy1)
patterns_found.append(pattern_with_low_accuracy1)
num_patterns_found.append(result_cardinality)

t1 /= num_loops
calculation1 /= num_loops

execution_time1.append(t1)
num_calculation1.append(calculation1)



output_path = r'./mode_1.txt'
output_file = open(output_path, "w")
num_lines = len(execution_time1)

output_file.write("execution time\n")
output_file.write('{}\n'.format(execution_time1[0]))


output_file.write("\n\nnumber of calculations\n")
output_file.write('{}\n'.format(num_calculation1[0]))


output_file.write("\n\nnumber of patterns found\n")
output_file.write('{} \n {}\n'.format(num_patterns_found[0], patterns_found[0]))



