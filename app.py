import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open('static/wine_quality_pred_model1.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    df = pd.read_csv('static/winequality-red.csv')
    df_mean = df[["alcohol", "sulphates", "volatile acidity", "citric acid"]].mean()
    df_std = df[["alcohol", "sulphates", "volatile acidity", "citric acid"]].std()
    
    features = [float(x) for x in request.form.values()]
    c = 0
    for i in df_mean.keys():
        features[c] = (features[c] - df_mean[i])/df_std[i]
        c = c+1

    features = np.array(features).reshape(1, 4)
    prediction = model.predict(features)
    
    result = prediction[0]
    quality = ''
    if result == 0:
        quality = 'low'
    elif result == 1:
        quality = 'medium'
    elif result == 2:
        quality = 'high'

    return render_template('index.html', prediction_text=f"Quality: {quality}")
    
if __name__ == "__main__":
    app.run(debug=True)