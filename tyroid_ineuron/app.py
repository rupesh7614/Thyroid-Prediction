from flask import Flask, render_template, request
import re
import pandas as pd
import copy
import pickle
import joblib

model = pickle.load(open('model6.pkl','rb'))
median_impute = joblib.load('medianimpute')
mode_impute = joblib.load('modeimpute')
encoding = joblib.load('encoding')
minmax = joblib.load('minmax')




def xgboost(data_new):
    clean1 = pd.DataFrame(median_impute.transform(data_new), columns = data_new.select_dtypes(exclude = ['object']).columns)
    clean2 = pd.DataFrame(mode_impute.transform(data_new), columns = data_new.select_dtypes(include = ['object']).columns)
    clean3 = pd.DataFrame(encoding.transform(clean2))
    
    clean4 = pd.DataFrame(minmax.transform(clean1))
    
    clean_data = pd.concat([clean4, clean3], axis = 1, ignore_index = True)
    prediction = pd.DataFrame(model.predict(clean_data), columns = ['Prediction'])
    final_data = pd.concat([prediction, data_new], axis = 1)
    return(final_data)
    
            
#define flask
app = Flask(__name__)



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/success', methods = ['POST'])
def success():
    if request.method == 'POST' :
        f = request.files['file']
        data_new = pd.read_csv(f)
       
        final_data = xgboost(data_new)

       
        return render_template("new.html", Y = final_data.to_html(justify = 'center'))

if __name__=='__main__':
    app.run(debug = True, port=8000)
