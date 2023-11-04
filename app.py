import numpy as np
import pandas as pd
import pickle

from flask import Flask, request, jsonify, render_template
from waitress import serve

app = Flask(__name__)

# read and prepare model 
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('./index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ''' 
    Rendering results on HTML
    '''
    # get data
    features = dict(request.form)
    # Hent ut år, måned og tid fra År og Tid
    dato = features["År"]
    date = features["År"].split("-")
    År = int(date[0])
    Måned = int(date[1])

    Tid = int(features["Tid"].split(":")[0])

    # Beregn dag i uken (0 for mandag, 6 for søndag)
    Dager = pd.Timestamp(features["År"]).dayofweek

    # Oppdater features-dictionary med de beregnede verdiene
    features["År"] = År
    features["Måned"] = Måned
    features["Dager"] = Dager
    features["Tid"] = Tid

    print(features)

    # expected keys
    numeric_features = ["Globalstraling", "Solskinstid", "Lufttemperatur", "Vindretning", "Vindstyrke", "Lufttrykk", "Vindkast"]
    categorical_features = ["Dager", "Tid", "Måned", "År"]
    


    # handle wrong input
    def to_numeric(key, value, numeric_features = numeric_features):
        if key not in numeric_features:
            return value
        try:
            return float(value)
        except:
            return np.nan
    features = {key: to_numeric(key, value) for key, value in features.items()}

    # prepare for prediction
    features_df = pd.DataFrame(features, index=[0]).loc[:, numeric_features + categorical_features]
    print(features_df)
    

    # sjekk input
    if features_df.loc[0, 'Tid'] <= 0:
        return render_template('./index.html',
                               prediction_text='Tid must be positive')

    # predict
    prediction = model.predict(features_df)
    prediction = np.round(prediction[0])
    prediction = np.clip(prediction, 0, np.inf)
    
    # prepare output
    return render_template('./index.html',
                           prediction_text=f'Totalt antall syklister i {dato}: {prediction}!')
if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=8080)
