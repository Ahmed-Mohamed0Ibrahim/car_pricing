
import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import pandas as pd
import numpy as np

app=Flask(__name__)
#load model
regmodel = pickle.load(open("regmodel.pkl","rb"))
#load preprocessing
preprocessing_pipe = pickle.load(open("preprocessing.pkl","rb"))

#columns names
columns=["Year","Present_Price","Driven_kms","Fuel_Type","Selling_type","Transmission","Owner"]

@app.route('/')
def home():
    return render_template("home.html")

@app.route("/predict_api", methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    data = np.array([list(data.values())])
    new_val = pd.DataFrame(data,columns=columns)
    output=regmodel.predict(preprocessing_pipe.transform(new_val))
    print(output[0])
    return jsonify(output[0])

@app.route("/predict",methods=['POST'])
def predict():
    # get the values from the form
    data = np.array([list(request.form.values())],dtype="object")
    # convert the numirical values to float
    data[:,0:3] = data[:,0:3].astype(np.float_)
    new_val = pd.DataFrame(data,columns=columns)
    output=regmodel.predict(preprocessing_pipe.transform(new_val))
    print("The car price prediction is :",output[0])
    return render_template("home.html",Prediction_text="The car price prediction is : {}".format(output[0]))

if __name__=="__main__":
    app.run(debug=True)