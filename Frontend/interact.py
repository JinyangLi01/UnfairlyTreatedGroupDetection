import PySimpleGUI as sg
import pandas as pd
from Algorithms import NewAlgGeneral_SizeFairnessValue_2_20210528 as newalggeneral
from Algorithms import NewAlg_1_20210529 as newalgclassification

sg.theme('BlueMono')
font = ("Arial", 20)
sg.set_options(font=font)
sg.set_options(element_size=(100, 30))


def read_with_att(original_data_file, selected_attributes):
    original_data = pd.read_csv(original_data_file)
    less_attribute_data = original_data[selected_attributes]
    return less_attribute_data


# ==================== preparing dicts =========================
datasets = dict()
datasets['COMPAS'] = r"../InputData/COMPAS_ProPublica/compas-analysis-master/cox-parsed/cox-parsed_7214rows_cat.csv"
datasets['COMPAS-TP'] = r"../InputData/COMPAS_ProPublica/compas-analysis-master/cox-parsed/cox-parsed-TP-cat.csv"
datasets['COMPAS-FP'] = r"../InputData/COMPAS_ProPublica/compas-analysis-master/cox-parsed/cox-parsed-FP-cat.csv"
datasets['COMPAS-TN'] = r"../InputData/COMPAS_ProPublica/compas-analysis-master/cox-parsed/cox-parsed-TN-cat.csv"
datasets['COMPAS-FN'] = r"../InputData/COMPAS_ProPublica/compas-analysis-master/cox-parsed/cox-parsed-FN-cat.csv"

general_fairness_definition = {
    'False positive error rate balance (predictive equality)': 1,
    'False negative error rate balance (equal opportunity)': 2,
    'Predictive parity': 0,
    'Equalized odds': 3,
    'Conditional use accuracy equality': 4,
    'Treatment equality': 5}

# used in interaction
dataset_name = str()
dataset_file = str()
classifier_name = str()
classifier_file = str()
size_threshold = 0
fairness_delta = 0
whole_data = pd.DataFrame
TP = pd.DataFrame
TN = pd.DataFrame
FP = pd.DataFrame
FN = pd.DataFrame
time_limit = 5 * 60
fairness_definition = 1
selected_attributes = []

# ==================== preparing dicts =========================


# ==================== to preview data =====================
data_for_preview = []
header_list = []
PreviewTable = sg.Table(
    visible=False,
    values=data_for_preview,
    headings=header_list,
    display_row_numbers=True,
    auto_size_columns=False,
    key='-preview_table-',
    num_rows=min(25, len(data_for_preview))
)


# attributes
Attributes = []

# ==================== to preview data =====================


def ReadCateFile(cate_file):
    translation = dict()
    f = open(cate_file, "r")
    Lines = f.readlines()
    start = True
    key = str()
    att = dict()
    LastLineIsEmpty = False
    for line in Lines:
        if line == "\n":
            if LastLineIsEmpty:
                break
            LastLineIsEmpty = True
            translation[key] = att
            att = dict()
            start = True
            continue
        LastLineIsEmpty = False
        line = line.strip()
        if start:
            att = dict()
            key = line
            start = False
        else:
            items = line.split(":")
            att[items[0]] = items[1]
    if not LastLineIsEmpty:
        translation[key] = att
    print(translation)
    return translation


def TranslatePatternsToNonNumeric(pattern_with_low_fairness, dataset_name, selected_attributes):
    cate_file = dataset_name.split(" ")[0] + "_categorization.txt"

    translaion = ReadCateFile(cate_file)
    results = []
    for p in pattern_with_low_fairness:
        re = dict()
        idx = 0
        for i in p:
            if i == -1:
                idx += 1
                continue
            else:
                attribute = selected_attributes[idx]
                re[attribute] = translaion[attribute][str(i)]
            idx += 1
        results.append(re)
    return results


# pattern_with_low_fairness = [[-1, -1, 0], [-1, 0, -1] ]
# selected_attributes = ["sex", "age_cat", "race"]
# TranslatePatternsToNonNumeric(pattern_with_low_fairness, "COMPAS Data", selected_attributes)


# Define the window's contents
layout = [
    [sg.Text('Dataset')],
    [sg.InputCombo(['COMPAS Data', 'Adult Data', 'Creditcard Data'], size=(30, 30),
                   default_value='COMPAS Data', key='-InputData-', enable_events=True),
     sg.FileBrowse("Upload dataset", size=(30, 1), key='-upload_dataset-', enable_events=True)],
    [sg.Button("Preview data", size=(30, 1))],

    [PreviewTable],
    [Attributes],

    [sg.Text("ML classifier")],
    [sg.InputCombo(['Decision tree', 'Random forest', 'Ada boost'],
                   size=(30, 30), default_value='Decision tree', key='-MLClassifier-', enable_events=True),
     sg.FileBrowse("Upload ML classifier", size=(30, 1), key='-upload_classifier-', enable_events=True)],

    [sg.Text("Fairness definition", size=(31, 1))],
    [sg.InputCombo(['Overall accuracy equality',
                    'False positive error rate balance (predictive equality)',
                    'False negative error rate balance (equal opportunity)',
                    'Predictive parity',
                    'Equalized odds',
                    'Conditional use accuracy equality',
                    'Treatment equality'],
                   size=(30, 30), default_value='False positive error rate balance (predictive equality)',
                   key='-fairness_definition-')],
     [sg.Button("Show original fairness value", size=(30, 1), key='-show_original_fairness_value-'),
        sg.Multiline("0.0", size=(30, 1), key='-original_fairness_value-')],
    [sg.Text("Size threshold", size=(31, 1)), sg.Text("Fairness delta", size=(30, 1))],
    [sg.Spin([i for i in range(1, 1000)], initial_value=20, key='-size_threshold-', enable_events=True, size=(29, 1)),
     sg.Slider(range=(0, 1), default_value=0.1, resolution=0.01, size=(27, 20), orientation="h",
               enable_events=True, key='-fairness_delta-', tick_interval=1)],

    # [sg.Input(key='-input_size_threshold-', size=(30, 1), default_text="20"),
    #  sg.Input(key='-input_fairness_delta-', size=(30, 1), default_text="0.1")],
    [sg.Button("RUN", size=(30, 1)), sg.Button('Quit', size=(30, 1))],
    [sg.Text(size=(100, 100), key='-OUTPUT-')]
    # [sg.MLine(key='-OUTPUT-'+sg.WRITE_ONLY_KEY,  size=(40,100))],
]

# Create the window
window = sg.Window('Window Title', layout, size=(1000, 1000), font=font)

UploadDataFlag = False
UploadClassifierFlag = False

# Display and interact with the Window using an Event Loop
while True:
    event, values = window.read()
    # See if user wants to quit or window was closed
    if event == sg.WINDOW_CLOSED or event == 'Quit':
        break
    elif event == "-InputData-":
        UploadDataFlag = False
        dataset_name = values['-InputData-']
        window['-OUTPUT-'].update("")
        print(event, dataset_name)
    elif event == "-upload-dataset-":
        UploadDataFlag = True
        dataset_file = values['-upload-dataset-']
        window['-OUTPUT-'].update("")
        print(event, dataset_file)
    elif event == "-MLClassifier-":
        classifier_name = values['-MLClassifier-']
        window['-OUTPUT-'].update("")
        print(event, classifier_name)
    elif event == "-upload_classifier-":
        UploadClassifierFlag = True
        classifier_file = values['-upload_classifier-']
    elif event == "Preview data":
        print(values['-InputData-'])
        if UploadDataFlag:
            dataset_file = values['-upload_dataset-']
            df = pd.read_csv(dataset_file, sep=',', engine='python', header=None)
        else:
            dataset_name = values['-InputData-']
            data_name_short = dataset_name.split(' ')[0]
            df = pd.read_csv(datasets[data_name_short], sep=',', engine='python', header=None)
        data_for_preview = df.values.tolist()               # read everything else into a list of rows
        # Uses the first row (which should be column names) as columns names
        header_list = df.iloc[0].tolist()
        # Drops the first row in the table (otherwise the header names and the first row will be the same)
        data_for_preview = df[1:].values.tolist()
        # TODO: update table
        # PreviewTable = sg.Table(
        #     visible=True,
        #     values=data_for_preview,
        #     headings=header_list,
        #     display_row_numbers=True,
        #     auto_size_columns=False,
        #     key='-preview_table-',
        #     num_rows=min(25, len(data_for_preview)))
        window['-preview_table-'].update(visible=True)
        window['-preview_table-'].update(values=data_for_preview)
        window['-preview_table-'].update(num_rows=min(10, len(data_for_preview)))
        # window['-preview_table-'].update(headings=header_list)
    elif event == "-show_original_fairness_value-":
        dataset_name = values['-InputData-']
        classifier_name = values['-MLClassifier-']
        # TODO: selected attributes
        selected_attributes = ["sex", "age_cat", "race"]
        print(event, dataset_name, classifier_name)
        data_name_short = dataset_name.split(' ')[0]
        whole_data = read_with_att(datasets[data_name_short], selected_attributes)
        TP = read_with_att(datasets[data_name_short + '-TP'], selected_attributes)
        FP = read_with_att(datasets[data_name_short + '-FP'], selected_attributes)
        TN = read_with_att(datasets[data_name_short + '-TN'], selected_attributes)
        FN = read_with_att(datasets[data_name_short + '-FN'], selected_attributes)
        fairness_definition = int(general_fairness_definition[values['-fairness_definition-']])
        original_fairness_value = newalggeneral.ComputeOriginalFairnessValue(whole_data, TP, TN,
                                                                             FP, FN, fairness_definition)
        print(original_fairness_value)
        window['-original_fairness_value-'].update(original_fairness_value)
    elif event == "-size_threshold-":
        window['-OUTPUT-'].update("")
    elif event == "-fairness_delta-":
        window['-OUTPUT-'].update("")
    elif event == "RUN":
        size_threshold = int(values['-size_threshold-'])
        fairness_delta = float(values['-fairness_delta-'])

        pattern_with_low_fairness, sizes_of_patterns, fairness_values_of_patterns, \
        num_pattern_checked, run_time = newalggeneral.GraphTraverse(whole_data,
                                                                    TP, TN, FP, FN, fairness_delta,
                                                                    size_threshold, time_limit, fairness_definition)
        print("num of patterns detected = {}".format(len(pattern_with_low_fairness)))
        for p in pattern_with_low_fairness:
            print(p)
        results = TranslatePatternsToNonNumeric(pattern_with_low_fairness, dataset_name, selected_attributes)
        update_text = str(len(pattern_with_low_fairness)) + " patterns detected: (with size and fairness value)\n"
        for i in range(len(pattern_with_low_fairness)):
            update_text += str(results[i]) + "    " + str(sizes_of_patterns[i]) + "    " \
                           + str(fairness_values_of_patterns[i]) + "\n"
        # window['-OUTPUT-' + sg.WRITE_ONLY_KEY].print(update_text)
        window['-OUTPUT-'].update(update_text)

    # Output a message to the window
    # window['-OUTPUT-'].update('Hello ' + values['-INPUT-'] + "! Thanks for trying PySimpleGUI")

# Finish up by removing from the screen
window.close()
