import json
import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import pandas as pd
import numpy as np

app=Flask(__name__)
#load model
regmodel = pickle.load(open("regmodel.pkl","rb"))
#load preprocessing
preprocessing_pipe = pickle.load(open("preprocessing.pkl","rb"))

@app.route('/')
def home():
    return render_template("home.html")

@app.route("/predict_api", methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    x_x = np.array([list(data.values())])
    columns=["Year","Present_Price","Driven_kms","Fuel_Type","Selling_type","Transmission","Owner"]
    new_val = pd.DataFrame(x_x,columns=columns)
    output=regmodel.predict(preprocessing_pipe.transform(new_val))
    print(output[0])
    return jsonify(output[0])

@app.route("/predict",methods=['POST'])
def predict():
    data = [request.form.values()]

if __name__=="__main__":
    app.run(debug=True)