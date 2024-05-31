import joblib
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the model
model = joblib.load(open('emailfab.pkl','rb'))
label = joblib.load(open('label.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json
    print(data)
    incoming_data = np.array(list(data.values())).reshape(-1)
    incoming_data[0]['first_name'] = 'abc'
    incoming_data[0]['last_name'] = 'xyz'
    #incoming_data[0]['first_last'] = 'axyz'
    data['data'] = incoming_data[0]

    ordered_keys = list(data['data'].keys())
    ordered_values = [data['data'][key] for key in ordered_keys]
    features = pd.DataFrame([ordered_values], columns=ordered_keys)

    for col, encoder in label.items():
         features[col] = encoder.transform(features[col])

    """
    # Converting into labels
    values_to_encode = list(data['data'].values())
    encoded_values = label.transform(values_to_encode)
    encoded_dict = {key: encoded_values[i] for i, key in enumerate(data['data'])}
    data['data'] = encoded_dict

    ordered_keys = list(data['data'].keys())
    ordered_values = [data['data'][key] for key in ordered_keys]
    features = pd.DataFrame([ordered_values], columns=ordered_keys)
    """

    # new_data = np.array(list(data.values())).reshape(-1)
    print(features)
    output = model.predict(features)
    print(output[0])
    return jsonify(output[0])

if __name__=="__main__":
      app.run(debug=True)



