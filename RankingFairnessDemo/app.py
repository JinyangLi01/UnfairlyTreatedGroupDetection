from flask import Flask, render_template, jsonify
import csv

app = Flask(__name__)

def get_csv_data(dataset_path):
    data = []
    column_order = []

    with open(dataset_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        column_order = reader.fieldnames
        for row in reader:
            data.append(row)

    return column_order, data

@app.route('/')
def index():
    return render_template('input.html')

@app.route('/api/datasets/<dataset>', methods=['GET'])
def get_dataset(dataset):
    dataset_path = 'datasets/' + dataset + '.csv'
    column_order, data = get_csv_data(dataset_path)

    return jsonify({'column_order': column_order, 'data': data})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
