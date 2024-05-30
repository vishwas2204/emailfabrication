import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
# Load the model
model = pickle.load(open("emailfab.pkl","rb"))
label = pickle.load(open("label.pkl", "rb"))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict.api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = label.transform(data.iloc[:])
    output = model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

if __name__=="__main__":
      app.run(debug=True)



