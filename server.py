from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

@app.route("/predict", methods=['POST'])
def do_prediction():
    json = request.get_json()
    model = joblib.load('model/arima_model.pkl')
    df = pd.DataFrame(json, index=[0])

    from sklearn.preprocessing import StandardScaler
    
    # Calculate the 24h_avg and sub_index values of the selected features
    """
        To-do
    """
    scaler = StandardScaler()
    scaler.fit(df)

    df_x_scaled = scaler.transform(df)

    df_x_scaled = pd.DataFrame(df_x_scaled, columns=df.columns)
    y_predict = model.predict(df_x_scaled)

    result = {"Predicted AQI" : np.expm1(y_predict[0])}
    return jsonify(result)

if __name__ == "__main__":
    app.run(host='0.0.0.0')