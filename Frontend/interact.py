import PySimpleGUI as sg

font = ("Arial", 20)
sg.set_options(font=font)
sg.set_options(element_size=(100, 30))

# Define the window's contents
layout = [
    [sg.Text('Dataset')],
    [sg.InputCombo(['Compas Data', 'Adult Data', 'Creditcard Data'], size=(30, 30), default_value='Compas Data'),
            sg.FileBrowse("Upload dataset", size=(30, 1))],
    [sg.Button("Preview data", size=(30, 1))],

    [sg.Text(size=(80,10), key='-preview_data-')],

    [sg.Text("ML classifier")],
    [sg.InputCombo(['Decision tree -- COMPAS', 'Random forest -- COMPAS', 'Ada boost -- COMPAS',
                  'Decision tree -- Adult', 'Random forest -- Adult', 'Ada boost -- Adult',
                  'Decision tree -- Creditcard', 'Random forest -- Creditcard', 'Ada boost -- Creditcard'],
                 size=(30, 30), default_value='Decision tree -- COMPAS'),
            sg.FileBrowse("Upload ML classifier", size=(30, 1))],
    [sg.Text("Fairness definition", size=(31, 1)), sg.Text("Fairness value", size=(30, 1))],
    [sg.InputCombo(['Overall accuracy equality', 'Predictive parity', 'False positive error rate balance (predictive equality) ',
                  'False negative error rate balance (equal opportunity)', 'Equalized odds', 'Conditional use accuracy equality',
                  'Treatment equality'],
                  size=(30, 30), default_value='Overall accuracy equality'),
            sg.SimpleButton("0.75", size=(30, 1))],
    [sg.Text("Size threshold", size=(31, 1)), sg.Text("Fairness delta", size=(30, 1))],
    [sg.Input(key='-input_size_threshold-', size=(30, 1)), sg.Input(key='-input_fairness_delta-', size=(30, 1))],
  [sg.Input(key='-INPUT-')],
  # [sg.Text(size=(80,10), key='-OUTPUT-')],
    [sg.Button("RUN", size=(30, 1)), sg.Button('Quit', size=(30, 1))]
]

# Create the window
window = sg.Window('Window Title', layout, size=(1000, 1000), font=font)

# Display and interact with the Window using an Event Loop
while True:
    event, values = window.read()
    # See if user wants to quit or window was closed
    if event == sg.WINDOW_CLOSED or event == 'Quit':
        break
    elif event == "Preview data":
        window['-preview_data-'].update()
    # Output a message to the window
    window['-OUTPUT-'].update('Hello ' + values['-INPUT-'] + "! Thanks for trying PySimpleGUI")

# Finish up by removing from the screen
window.close()
