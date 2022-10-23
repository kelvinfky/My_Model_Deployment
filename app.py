import pandas as pd
import flask
from flask import Flask, request, render_template
import pickle
from sklearn.preprocessing import StandardScaler


app = Flask(__name__, template_folder='template')
model = pickle.load(open('Lasso Regression Prediction.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('Lasso.html')

@app.route('/result', methods=['POST'])
def predict():
    Jet_Fuel_avi_BTU = flask.request.form['Jet Fuel_avi_BTU'] #1
    Gasoline_avi_BTU = flask.request.form['Gasoline_avi_BTU'] #2
    LDV_SWB_road_BTU = flask.request.form['LDV_SWB_road_BTU'] #3
    LDV_LWB_road_BTU = flask.request.form['LDV_LWB_road_BTU'] #4
    Combination_Truck_road_BTU = flask.request.form['Combination_Truck_road_BTU'] #5
    Bus_Road_BTU = flask.request.form['Bus_Road_BTU'] #6
    Railways_BTU = flask.request.form['Railways_BTU'] #7
    Water_BTU = flask.request.form['Water_BTU'] #8
    Natural_Gas_BTU = flask.request.form['Natural_Gas_BTU'] #9
    LDV_SWB_EFF = flask.request.form['LDV_SWB_EFF'] #10
    LDV_LWB_EFF = flask.request.form['LDV_LWB_EFF'] #11
    Passenger_Car_EFF = flask.request.form['Passenger_Car_EFF'] #12
    Domestic_EFF = flask.request.form['Domestic_EFF'] #13
    Imported_EFF = flask.request.form['Imported_EFF'] #14
    Light_Truck_EFF = flask.request.form['Light_Truck_EFF'] #15
    Passenger_Car_Age = flask.request.form['Passenger_Car_Age'] #16
    Light_Truck_Age = flask.request.form['Light_Truck_Age'] #17
    Light_vehicle_Age = flask.request.form['Light_vehicle_Age'] #18
    Demand_petroleum_transportation_mil_lit = flask.request.form['Demand_petroleum_transportation)mil_lit'] #19
    Average_MC_15000_miles_dollars = flask.request.form['Average_MC/15000_miles(dollars)'] #20

    input_variables = pd.DataFrame([[Jet_Fuel_avi_BTU, Gasoline_avi_BTU, LDV_SWB_road_BTU,LDV_LWB_road_BTU,Combination_Truck_road_BTU,
                                     Bus_Road_BTU,Railways_BTU,Water_BTU,Natural_Gas_BTU,LDV_SWB_EFF,LDV_LWB_EFF,Passenger_Car_EFF,
                                     Domestic_EFF,Imported_EFF,Light_Truck_EFF,Passenger_Car_Age,Light_Truck_Age,Light_vehicle_Age,
                                     Demand_petroleum_transportation_mil_lit,Average_MC_15000_miles_dollars]],
                                     columns=['Jet Fuel_avi_BTU', 'Gasoline_avi_BTU', 'LDV_SWB_road_BTU','LDV_LWB_road_BTU','Combination_Truck_road_BTU',
                                              'Bus_Road_BTU','Railways_BTU','Water_BTU','Natural_Gas_BTU','LDV_SWB_EFF','LDV_LWB_EFF','Passenger_Car_EFF',
                                              'Domestic_EFF','Imported_EFF','Light_Truck_EFF','Passenger_Car_Age','Light_Truck_Age','Light_vehicle_Age',
                                              'Demand_petroleum_transportation)mil_lit','Average_MC/15000_miles(dollars)'],
                                       dtype=float)
    #standardization of numerical features
    list_numerical = input_variables.columns
    scaler = StandardScaler().fit(input_variables[list_numerical])
    input_variables[list_numerical] = scaler.transform(input_variables[list_numerical])
                                     
    prediction = model.predict(input_variables)
    
    return render_template('result.html',prediction_text = 'CO2 Emission is predicted to be {}'.format(prediction))

if __name__ == '__main__':
    app.run(debug=True)