from flask import Flask, request, jsonify, render_template
import pickle
import json
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

__locations = None
data_columns = None
model = None

'''
def get_estimated_price(location,sqft,bhk,bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index>=0:
        x[loc_index] = 1

    return round(__model.predict([x])[0],2)
'''

def load_saved_artifacts():
    print("loading saved artifacts...start")
    
    global data_columns
    global __locations

    with open("columns.json", "r") as f:
        data_columns = json.load(f)['data_columns']
        __locations = data_columns[4:]  # first 3 columns are sqft, bath, bhk

    global model
    if model is None:
        with open('banglore_home_prices_model.pickle', 'rb') as f:
            model = pickle.load(f)
    print("loading saved artifacts...done")

'''
def get_data_columns():
    return __data_columns
'''
'''
@app.route('/get_location_names', methods=['GET'])
def get_location_names():
    response = jsonify({
        'locations': __locations
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response
'''

@app.route('/predict_home_price', methods=['POST'])
def predict_home_price():
    
    total_sqft = float(request.form['total_sqft'])
    location = request.form['location']
    bhk = int(request.form['bhk'])
    bath = int(request.form['bath'])

#    response = jsonify({
#        'estimated_price': get_estimated_price(location,total_sqft,bhk,bath)
#    })

    try:
        loc_index = data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(data_columns))
    x[0] = total_sqft
    x[1] = bath
    x[2] = bhk
    if loc_index>=0:
        x[loc_index] = 1

    output=round(model.predict([x])[0],2)
    return render_template('index.html', prediction_text=output)

if __name__ == "__main__":
    print("Starting Python Flask Server For Home Price Prediction...")
    load_saved_artifacts()
    app.run(debug=True)