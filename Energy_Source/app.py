import os

import pandas as pd
import numpy as np

import sqlalchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine

from flask import Flask, jsonify, render_template
from flask import redirect, request
from flask_sqlalchemy import SQLAlchemy
import json

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pickle

app = Flask(__name__)

#################################################
# Database Setup
#################################################

# app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///db/Alltypes.sqlite"
# db = SQLAlchemy(app)

# # reflect an existing database into a new model
# Base = automap_base()
# # reflect the tables
# Base.prepare(db.engine, reflect=True)

# # Save references to each table
# Samples_Metadata = Base.classes.sample_metadata
# Samples = Base.classes.samples

# db ="sqlite:///db/Alltypes.sqlite"
db ="postgres://xxlseihlmohnre:6667aa7c4ad666c7e6c92755418ff0486278365bde892a02ab7243afd5ca65ad@ec2-50-19-127-115.compute-1.amazonaws.com:5432/denrnahfu7g51u"

ratio = 1000


def one_var_prediction(pred_model, temperature):
    data = pd.DataFrame(np.array([[temperature]]))
        
    polynomial_features= PolynomialFeatures(degree=2)
    x_poly = polynomial_features.fit_transform(data)

    result = pred_model.predict(x_poly) / ratio
            
    prediction = float(result.round(2))
    return prediction

def two_var_prediction(pred_model, temperature , coal_consumption):
    data = pd.DataFrame(np.array([[coal_consumption * ratio,temperature ]]))
        
    polynomial_features= PolynomialFeatures(degree=2)
    x_poly = polynomial_features.fit_transform(data)

    result = pred_model.predict(x_poly) / ratio
            
    prediction = float(result.round(2))
    return prediction

@app.route("/")
def index():
    """Return the homepage."""
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    
    return render_template("dashboard.html")

@app.route("/machinelearning", methods=["GET", "POST"])
def machinelearning():
    natural_gas_message = ""
    coal_message = ""
    natural_gas_refined_message=""
    temperature=""

    if request.method == "POST":
        #temperature = float(request.form["temperature"])
        # natural gas model
        with open(f'MachineLearning/code/model/linear_regression_gas_TempOnly.pickle', "rb") as gas:
        #with open(f'Energy_Source/MachineLearning/code/model/linear_regression_gas_TempOnly.pickle', "rb") as gas:
            natural_gas_model = pickle.load(gas)
            temperature = float(request.form["temperature"])
            # data must be converted to df with matching feature names before predict
            natural_gas_message = one_var_prediction(natural_gas_model, temperature)
            

        # coal model
        with open(f'MachineLearning/code/model/linear_regression_coal_TempOnly.pickle', "rb") as coal:
        #with open(f'Energy_Source/MachineLearning/code/model/linear_regression_coal_TempOnly.pickle', "rb") as coal:
            coal_model = pickle.load(coal)
            temperature1 = float(request.form["temperature"])
            coal_message = one_var_prediction(coal_model, temperature1)

        # gas model using coal prediction
        with open(f'MachineLearning/code/model/linear_regression_gas_TempCoal.pickle', "rb") as gas_refine:
        #with open(f'Energy_Source/MachineLearning/code/model/linear_regression_gas_TempCoal.pickle', "rb") as gas_refine:
            gas_refined_model = pickle.load(gas_refine)
            temperature2 = float(request.form["temperature"])
            natural_gas_refined_message = two_var_prediction(gas_refined_model, temperature2, coal_message)           

    
        return render_template("machinelearning.html", natural_gas_message=natural_gas_message, coal_message=coal_message, temperature=temperature, natural_gas_refined_message=natural_gas_refined_message)
    else:
        return render_template("machinelearning_light.html")


@app.route("/bio")
def bio():
    
    return render_template("bio.html")

@app.route("/credits")
def credits():
    
    return render_template("credits.html")

@app.route("/resources")
def resources():
    
    return render_template("resources.html")

@app.route("/future")
def future():
    
    return render_template("future.html")

@app.route("/now")
def now():
    
    return render_template("now.html")    

@app.route("/history")
def history():
    
    return render_template("history.html")   

@app.route("/coal")
def coal():
    
    return render_template("indexBarCoal.html")   

@app.route("/naturalgas")
def naturalgas():
    
    return render_template("indexBarNaturalGas.html")   

@app.route("/nuclear")
def nuclear():
    
    return render_template("indexBarNuclear.html")   

@app.route("/petroleum")
def petroleum():
    
    return render_template("indexBarPetroleum.html")     

@app.route("/renewable")
def renewable():
    
    return render_template("indexBarRenewable.html")   

@app.route("/data")
def get_data():

  engine = create_engine(db)
  conn = engine.connect()
  
  sql = f"select * from Alltypes"
  Alltypes_df = pd.read_sql(sql, conn)
  conn.close()
  return Alltypes_df.to_json(orient="records")

@app.route("/data/year")
def get_year():
    engine = create_engine(db)
    conn = engine.connect()
  
    sql = f"select * from Alltypes"
    data = pd.read_sql(sql, conn)
    conn.close()
    # Return a list of the column names (sample names)
    yr = (data.columns)[2:]
    yr_reverse  = []

    for i in range(len(yr)-1 , -1, -1):
        yr_reverse.append(yr[i])
    return jsonify(yr_reverse)

@app.route("/data/state")
def get_state():
    engine = create_engine(db)
    conn = engine.connect()
  
    sql = f"select * from Alltypes"
    data = pd.read_sql(sql, conn)
    conn.close()

    # Return a list of the column names (sample names)
    return jsonify(list(data["state"].head(52)))

@app.route("/data/energyType")
def set_energyType():
    energyType = ['Coal','Natural Gas','Petroleum','Nuclear','Renewable']

    return jsonify(energyType)


@app.route("/<energy_type>/<yr>")
def select_data(energy_type, yr):
    engine = create_engine(db)

    sql = f"select * from Alltypes"

    conn = engine.connect()
  
    data = pd.read_sql(sql, conn)
    conn.close()

    tmp_data = data.loc[:,['type','state',yr]]
    
    selected_data = tmp_data[tmp_data['type']==energy_type ]

    data = {
        "State":selected_data['state'], 
        "consumption": selected_data[yr]
    }
    # print(data)

    data = pd.DataFrame(data)
    return data.to_json(orient="records")
    # return jsonify(data)
    
@app.route("/data/<energy_type>/<state>")
def select_data_per_state(energy_type,state):
    
    engine = create_engine(db)
    conn = engine.connect()
  
    sql = f"select * from Alltypes"
    data = pd.read_sql(sql, conn)
    conn.close()

    tmp = data[data['type'] == energy_type]
    consumption_data_tmp = tmp[tmp['state']==state].iloc[0].tolist()

    # test_Data ={}

    yr = []
    consumption = []
    # print(consumption_data_tmp)
    for i in range(2,len(data.columns)):
        yr.append( (data.columns)[i])
        consumption.append(consumption_data_tmp[i])
        # print(selected_data.loc[0])
        # print()

    # print(yr)
    
    selected_data={
        "consumption": consumption,
        "yr": yr, 
    }

    # data = pd.DataFrame(selected_data)
    
    # return data.to_json(orient="records")
    return jsonify(selected_data)

@app.route("/energy_type/<state>/<yr>")
def select_energyType_per_state_year(state,yr):
    engine = create_engine(db)

    sql = f"select * from Alltypes"
    
    # result = engine.execute("sql statement")

    conn = engine.connect()
  
    data = pd.read_sql(sql, conn)
    conn.close()
    tmp = data[data['state'] == state]
    sel_data =tmp[yr].tolist()
    # print(sel_data)
    data_label =['Coal','NaturalGas','Petroleum','Nuclear','Renewable']
    selected_data ={
        "EnergyType": data_label, 
        "consumption" : sel_data
    }

    # print(selected_data)
    return jsonify(selected_data)

if __name__ == "__main__":
    app.run(port=5006, debug=True)
    # app.run()
