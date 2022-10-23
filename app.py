import pandas as pd
import flask
from flask import Flask, request, render_template
import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np


app = Flask(__name__, template_folder='template')
model = pickle.load(open('Lasso Regression Prediction.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('Lasso.html')

@app.route('/result', methods=['POST'])
def predict():
        X_mean = [1503.527254973513,34.285880421082496,9212.633512069324,4792.013362464089,2752.2455498121544,172.3499724258531,
              490.26451999999995,268.033279122,692.4856178378379,22.520944226035926,17.139473393096033,32.50862800170194,
              32.07837837837839,33.28378378378379,23.818918918918925,9.759459459459464,9.505405405405408,9.68108108108108,
              475495.27829619515,6845.216216216216]
        
        X_stddev = [358.6982807158301,10.1030081655065,1448.401091041892,1565.0033125449118,896.3650017608462,70.80367485361496,
                    44.18505360029845,64.59010250331002,127.75646720286554,2.0255630027880054,1.0914934571979704,4.744899829788143,
                    4.932777847957204,4.340027746551956,3.2728544997565323,1.2796395888989514,1.320342116007553,1.300392641097688,
                    75413.3249976061, 2238.523647098377]
        
        data = [float(x) for x in request.form.values()]
        
        input_variables = [[(data[0]-X_mean[0])/X_stddev[0],(data[1]-X_mean[1])/X_stddev[1],(data[2]-X_mean[2])/X_stddev[2],(data[3]-X_mean[3])/X_stddev[3],(data[4]-X_mean[4])/X_stddev[4],(data[5]-X_mean[5])/X_stddev[5],(data[6]-X_mean[6])/X_stddev[6],(data[7]-X_mean[7])/X_stddev[7],(data[8]-X_mean[8])/X_stddev[8],(data[9]-X_mean[9])/X_stddev[9],(data[10]-X_mean[10])/X_stddev[10],(data[11]-X_mean[11])/X_stddev[11],(data[12]-X_mean[12])/X_stddev[12],(data[13]-X_mean[13])/X_stddev[13],(data[14]-X_mean[14])/X_stddev[14],(data[15]-X_mean[15])/X_stddev[15],(data[16]-X_mean[16])/X_stddev[16],(data[17]-X_mean[17])/X_stddev[17],(data[18]-X_mean[18])/X_stddev[18],(data[19]-X_mean[19])/X_stddev[19]]]
                                     
        prediction = model.predict(input_variables)
    
        return render_template('result.html',prediction_text = 'CO2 Emission is predicted to be {}'.format(prediction))

if __name__ == '__main__':
    app.run(debug=True)