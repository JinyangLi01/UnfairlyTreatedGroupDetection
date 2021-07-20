import PySimpleGUI as sg

TaggerList = ["viking", "saddle", "beast", "ze", "princess", "vet", "art", "two", "hood", "mosaic",
              "viking1", "saddle1", "beast1", "ze1", "princess1", "vet1", "art1", "two1", "hood1", "mosaic1"]

TaggerListLen = len(TaggerList)
Tags1 = TaggerList[:int(TaggerListLen/3)]
Tags2 = TaggerList[int(TaggerListLen/3):int(TaggerListLen/3*2)]
Tags3 = TaggerList[int(TaggerListLen/3*2):]


def CBtn(BoxText):
    return sg.Checkbox(BoxText, size=(8, 1), default=False)

col2 = [[CBtn(i)] for i in range(len(Tags2))]

col5 = sg.Column([[sg.Checkbox("BoxText1", size=(8, 1), default=False)],
            [sg.Checkbox("BoxText2", size=(8, 1), default=False)],
            [sg.Checkbox("BoxText3", size=(8, 1), default=False)],
            [sg.Checkbox("BoxText4", size=(8, 1), default=False)]])

layout = [
    # [sg.Menu(menu_def, tearoff=True)],
    [sg.Text('Image Tagger', size=(
        30, 1), justification='center', font=("Helvetica", 25), relief=sg.RELIEF_RIDGE)],
    [sg.Text('Your Folder', size=(15, 1), justification='right'),
        sg.InputText('Default Folder'), sg.FolderBrowse()],
    [sg.Text('Column 2', justification='center', size=(10, 1))],
    [sg.Column(col2)],
    [col5]]

window = sg.Window('Everything bagel', layout)

while True:
    event, values = window.read()
    if event == sg.WINDOW_CLOSED or event == 'Quit':
        break
window.close()
