import pandas as pd
import numpy as np
import itertools  
import os

from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler

from flask import Flask, render_template, request

path0=os.path.abspath(os.path.dirname(__file__))
file_0=os.path.join(path0, 'ressources/02_df_Origin.csv')
file_1=os.path.join(path0, 'ressources/03_df_Dest.csv')


df_origin = pd.read_csv(file_0, index_col=0)
df_dest = pd.read_csv(file_1, index_col=0)


app = Flask(__name__) # Creer app et charger les fonctionalités de Flask


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def input_dashboard(path): #Afficher les inputs dans la page HTML dashboard
    
  origin_list =''
  for origin in df_origin['Input'].sort_values():
    origin_list+='<option value="'+origin+'">'+origin+'</option>'	
    
  dest_list =''
  for dest in df_dest['Input'].sort_values():
    dest_list+='<option value="'+dest+'">'+dest+'</option>'	
     
  return render_template('dashboard.html', origin_list=origin_list, dest_list=dest_list) 
                                                     

@app.route('/predict_delay', methods=['POST','GET'])
def predict_delay():
    if request.method=='POST':
      # Appeler les Inputs de la page HTML dashboard
      month  = request.form['month']
      day_of_month = request.form['day_of_month']
      day_of_week = request.form['day_of_week']
      carrier = request.form['carrier']
      origin = request.form['origin']
      dest = request.form['dest']
      dep = request.form['dep']
    
    
      API_info = open('input_API.joblib', 'rb')
      input_API = joblib.load(API_info)
      input_API_0 = np.zeros(len(input_API))
    
      input_API_0[input_API['MONTH_'+str(month)]] = 1
      input_API_0[input_API['DAY_OF_MONTH_'+str(day_of_month)]] = 1
      input_API_0[input_API['DAY_OF_WEEK_'+str(day_of_week)]] = 1
      input_API_0[input_API['CARRIER_'+str(carrier)]] = 1
      input_API_0[input_API['ORIGIN_CITY_NAME_'+str(origin)]] = 1
      input_API_0[input_API['DEST_CITY_NAME_'+str(dest)]] = 1
      input_API_0[input_API['CRS_DEP_TIME_'+str(dep)]] = 1
    
      input_API_on = input_API_0
    
      sc = StandardScaler()
      delay_var_output = []
      input_API_on_1 = input_API_on.reshape(-1, 1)
    
    #Prédiction carrier_delay
      API_info_carrier = open('gscv_reg_carrier_delay_knn.joblib', 'rb')
      gscv_reg_carrier_delay_knn = joblib.load(API_info_carrier)
      predict_carrier = gscv_reg_carrier_delay_knn.predict(input_API_on_1.T)
      delay_var_output.append(predict_carrier)
    
    #Prédiction NAS_delay
      API_info_NAS = open('gscv_reg_NAS_delay_knn.joblib', 'rb')
      gscv_reg_NAS_delay_knn = joblib.load(API_info_NAS)
      predict_NAS = gscv_reg_NAS_delay_knn.predict(input_API_on_1.T)
      delay_var_output.append(predict_NAS)
    
    #Prédiction Aircraft_delay
      API_info_AIRCRAFT = open('gscv_reg_AIRCRAFT_delay_knn.joblib', 'rb')
      gscv_reg_AIRCRAFT_delay_knn = joblib.load(API_info_AIRCRAFT)
      predict_AIRCRAFT = gscv_reg_AIRCRAFT_delay_knn.predict(input_API_on_1.T)
      delay_var_output.append(predict_AIRCRAFT)
    
      delay_var_output = sc.fit_transform(delay_var_output)
    
      def oneDArray(x):
          
         return list(itertools.chain(*x))
      delay_var_output = oneDArray(delay_var_output)

      input_Final = np.concatenate((input_API_on,delay_var_output))
      input_Final = input_Final.reshape(-1,1)

      API_info_dep = open('gscv_reg_ridge.joblib', 'rb')
      gscv_reg_ridge = joblib.load(API_info_dep)
      predict_dep_delay = int(gscv_reg_ridge.predict(input_Final.T))
    
      return render_template('prediction.html', prediction = predict_dep_delay, destination=dest)
                  

if __name__== '__main__': #Executer directement
    app.run(debug=True, port=4040) #Lancer le serveur local (localhost/adresse ip 127.0.0.1)



