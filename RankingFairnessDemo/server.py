
import pandas as pd
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the dataset into a pandas DataFrame
df = pd.read_csv('datasets/student-mat_cat_ranked_demovideo.csv')  # replace with your actual csv file path

@app.route('/api/datasets/student', methods=['GET'])
def get_student_dataset():
    # Convert the DataFrame into a list of dictionaries for JSONification
    data = df.to_dict(orient='records')

    # Return a preview of the data (e.g., the first 5 rows)
    return jsonify(data[:5])

if __name__ == '__main__':
    app.run(host='localhost', port=5000)
