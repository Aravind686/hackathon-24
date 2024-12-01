import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

# Load the model
regmodel = pickle.load(open('student_exam_model (2).pkl', 'rb'))

@app.route('/')
def home():
    return render_template("studentdetails.html")

@app.route("/predict_api", methods=['POST'])
def predict_api():
    try:
        # Parse JSON data from the request
        data = request.json
        print(f"Received data: {data}")

        # Validate that two inputs are provided
        if 'input1' not in data or 'input2' not in data:
            return jsonify({'error': 'Two inputs (input1 and input2) are required.'}), 400

        # Extract inputs
        input1 = data['input1']
        input2 = data['input2']
        print(f"Input1: {input1}, Input2: {input2}")

        # Prepare input data for the model (reshape if necessary)
        input_data = np.array([[input1, input2]])
        print(f"Prepared input data: {input_data}")

        # Make prediction
        prediction = regmodel.predict(input_data)
        print(f"Prediction result: {prediction}")

        # Return the prediction as a JSON response
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
