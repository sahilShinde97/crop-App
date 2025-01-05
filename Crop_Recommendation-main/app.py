from flask import Flask, jsonify, request
import IPython
import pandas as pd   
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import get_ipython
import warnings
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from flask_cors import CORS, cross_origin


app = Flask(__name__)
CORS(app)
data = pd.read_csv("Crop_recommendation.csv")
model = lgb.LGBMClassifier()



@app.route('/',methods=['GET'])
def Health():
   return jsonify("Running properly")
    

@app.route('/predict',methods=['POST'])
def predict():
    print(request.json)
    n = float(request.json['n'])
    p = float(request.json['p'])
    k = float(request.json['k'])
    temperature = float(request.json['temperature'])
    humidity = float(request.json['humidity'])
    ph = float(request.json['ph'])
    rainfall = float(request.json['rainfall'])
    try:
        input = [n,p,k,temperature,humidity,ph,rainfall]
        result =model.predict([input])
        data = {
        "crop":f"{result[0]}" 
        }
        response = jsonify(data)
        return response
    except Exception as e:
        print(e)
        return f"${e} Request is wrong!", 400
    

def train():
    #detect and remove the outliers
    df_boston= data
    df_boston.columns=df_boston.columns
    df_boston.head()
    
    print("Data got")

    #IQR
    Q1=np.percentile(df_boston['rainfall'],25,interpolation='midpoint')
    Q3=np.percentile(df_boston['rainfall'],75,interpolation='midpoint')

    IQR=Q3-Q1
    #upper bound
    upper=np.where(df_boston['rainfall']>=(Q3+1.5*IQR))

    #lower bound
    lower=np.where(df_boston['rainfall']<=(Q1-1.5*IQR))

    #removing the outliers
    df_boston.drop(upper[0],inplace=True)
    df_boston.drop(lower[0],inplace=True)

    print("Dropped!")
    x=data.drop('label',axis=1)
    y=data['label']
    print("started the")
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,shuffle=True,random_state=0)
    model.fit(x_train,y_train)
    print("Model Fitted")
    y_pred=model.predict(x_test)
    accuracy=accuracy_score(y_pred,y_test)
    print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test,y_pred)))

train(); 

if __name__ == '__main__':
    app.run(debug=True)
