import PySimpleGUI as sg
import pandas as pd
import numpy as np


# =================================================== example 1 ===============================================
# sg.theme('Dark Brown 1')

# headings = ['HEADER 1', 'HEADER 2', 'HEADER 3','HEADER 4']
# header =  [[sg.Text('  ')] + [sg.Text(h, size=(14,1)) for h in headings]]
#
# input_rows = [[sg.Input(size=(15,1), pad=(0,0)) for col in range(4)] for row in range(10)]
#
# layout = header + input_rows
#
# window = sg.Window('Table Simulation', layout, font='Courier 12')
#
# event, values = window.read()
#

# =================================================== example 2 ===============================================
# data_values = []
# data_headings = ['File ID', 'Type', 'Description', 'Remarks']
# data_values.append(['', '', '', ''])
# data_cols_width = [5, 8, 35, 35]
# tab5_layout = [
#
# [sg.Table(values=data_values, headings=data_headings,
#                             max_col_width=65,
#                             col_widths=data_cols_width,
#                             auto_size_columns=False,
#                             justification='left',
#                             enable_events=True,
#                             num_rows=6, key='_filestable_')],
#
# [sg.Button('Select Row', key='_rowselected_')] ]
#
# window = sg.Window('Window Title', tab5_layout, size=(1000, 1000))
#
# while True:
#     event, values = window.read()
#     if event == '_filestable_':
#         data_selected = [data_values[row] for row in values[event]]




# =================================================== example 3 ===============================================
matrix = [[str(x * y) for x in range(5)] for y in range(10)]
header=["one","two","three","four","five"]
Table = sg.Table(values=[], key="_table1_", headings=header, visible=False)
layout=[[sg.Text("Table")], [Table],[sg.Button("refresh")], [sg.Button("Exit")]]
window = (
    sg.Window("Table", default_element_size=(20, 22), resizable=True).Layout(layout).Finalize()
)

while True:

        event, values = window.Read()
        if event is None or event == "Exit":
            break
        elif event == "refresh":
            matrix[0][1]="10000"
            window.FindElement("_table1_").Update(values=matrix)
            window["_table1_"].update(visible=True)
        print(event, values)
window.Close()



# =================================================== example 4 ===============================================
#!/usr/bin/env python

# Yet another example of showing CSV data in Table

# def table_example():
#
#     sg.set_options(auto_size_buttons=True)
#     filename = sg.popup_get_file(
#         'filename to open', no_window=True, file_types=(("CSV Files", "*.csv"),))
#     # --- populate table with file contents --- #
#     if filename == '':
#         return
#
#     data = []
#     header_list = []
#     button = sg.popup_yes_no('Does this file have column names already?')
#
#     if filename is not None:
#         try:
#             # Header=None means you directly pass the columns names to the dataframe
#             df = pd.read_csv(filename, sep=',', engine='python', header=None)
#             data = df.values.tolist()               # read everything else into a list of rows
#             if button == 'Yes':                     # Press if you named your columns in the csv
#                 # Uses the first row (which should be column names) as columns names
#                 header_list = df.iloc[0].tolist()
#                 # Drops the first row in the table (otherwise the header names and the first row will be the same)
#                 data = df[1:].values.tolist()
#             elif button == 'No':                    # Press if you didn't name the columns in the csv
#                 # Creates columns names for each column ('column0', 'column1', etc)
#                 header_list = ['column' + str(x) for x in range(len(data[0]))]
#         except:
#             sg.popup_error('Error reading file')
#             return
#
#     layout = [
#         [sg.Table(values=data,
#                   headings=header_list,
#                   display_row_numbers=True,
#                   auto_size_columns=False,
#                   num_rows=min(25, len(data))
#                   )
#
#          ]
#     ]
#
#     window = sg.Window('Table', layout, grab_anywhere=False)
#     event, values = window.read()
#     window.close()
#
#
# table_example()
#
#
