import PySimpleGUI as sg
import pandas as pd

sg.theme('BlueMono')
font = ("Arial", 20)
sg.set_options(font=font)
sg.set_options(element_size=(1000, 500))


# ==================== to preview data =====================
matrix = [[str(x * y) for x in range(5)] for y in range(10)]
header=["        ","          ","          ","            ","             "]
Table = sg.Table(values=matrix, key="-preview_table-", headings=header, visible=False,
                 num_rows=5, col_widths=[500, 500, 500, 500, 500], max_col_width=1000,
                 size=(1000, 1000))
layout=[
        [sg.Text("Table")],
        [Table],
        [sg.Button("Visible")],
        [sg.Button("ChangeTable")],
        [sg.Button("Exit")]
]
window = (
    sg.Window("Table", default_element_size=(1000, 1000), resizable=True, size=(1000, 1000)).Layout(layout).finalize()
)

# Display and interact with the Window using an Event Loop
while True:
    event, values = window.read()
    # See if user wants to quit or window was closed
    if event == sg.WINDOW_CLOSED or event == 'Exit':
        break
    elif event == "Visible":
        matrix[0][1] = "10000"
        window["-preview_table-"].update(visible=True)
    elif event == "ChangeTable":
        df = pd.read_csv(r"../InputData/COMPAS_ProPublica/compas-analysis-master/cox-parsed/cox-parsed_7214rows_cat.csv", header=None)
        # Uses the first row (which should be column names) as columns names
        header_list = df.iloc[0].tolist()
        # Drops the first row in the table (otherwise the header names and the first row will be the same)
        data_for_preview = df[0:].values.tolist()
        print(df[1:].values.tolist()[:10])
        print(df[0:].values.tolist()[:10])
        # TODO: update table
        window['-preview_table-'].update(visible=True)
        window['-preview_table-'].update(values=data_for_preview)
        window['-preview_table-'].update(num_rows=min(10, len(data_for_preview)))
        # window['-preview_table-'].update(ColunmHeadings=header_list)

    # Output a message to the window
    # window['-OUTPUT-'].update('Hello ' + values['-INPUT-'] + "! Thanks for trying PySimpleGUI")

# Finish up by removing from the screen
window.close()
