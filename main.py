from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle

# load the model and encoders
model = pickle.load(open('models/model.pkl','rb'))
le_monatszahl = pickle.load(open('encoders/le_monatszahl.pkl','rb'))
le_auspraegung = pickle.load(open('encoders/le_auspraegung.pkl','rb'))

app = FastAPI()

class PredictionInput(BaseModel):
    MONATSZAHL: str
    AUSPRAEGUNG: str
    JAHR: int
    MONAT: int

# function to generalize calculating rolling features
def calculate_lagging_features(df, group_cols, value_col, lags):
    for lag in lags:
        df[f'{value_col}_mean_{lag}'] = df.groupby(group_cols)[value_col].transform(lambda x: x.rolling(center=False, window=lag, min_periods=1, closed="left").mean())
        df[f'{value_col}_sum_{lag}'] = df.groupby(group_cols)[value_col].transform(lambda x: x.rolling(center=False, window=lag, min_periods=1, closed="left").sum())
        df[f'{value_col}_median_{lag}'] = df.groupby(group_cols)[value_col].transform(lambda x: x.rolling(center=False, window=lag, min_periods=1, closed="left").median())
        df[f'{value_col}_std_{lag}'] = df.groupby(group_cols)[value_col].transform(lambda x: x.rolling(center=False, window=lag, min_periods=1, closed="left").std())

# load historical data for calculating features
def prepare_historical_data():
    
    historical_data = pd.read_csv("data/monatszahlen2402_verkehrsunfaelle_export_29_02_24_r.csv")

    # drop the rows with month==summe
    historical_data = historical_data.loc[historical_data['JAHR'] <= 2020]
    historical_data = historical_data[historical_data["MONAT"] != "Summe"]

    # reset the index after dropping
    historical_data = historical_data.reset_index(drop=True)
    historical_data['MONAT'] = historical_data['MONAT'].str[4:6]
    historical_data['MONAT'] = historical_data['MONAT'].astype(int)

    # then let's srot the rows according to their dates, acsending 
    historical_data = historical_data.sort_values(by=['JAHR', "MONAT"]).reset_index(drop=True)

    return historical_data

def prepare_new_instance(input):

    historical_data = prepare_historical_data()

    # handle the month being 1 (Jan), then the previous month is 12 (Dec)
    last_month = 12 if input['MONAT'] == 1 else input['MONAT'] - 1
    
    # calculate features for the new instance
    input['VORJAHRESWERT'] = historical_data[(historical_data['MONATSZAHL'] == input['MONATSZAHL']) & 
                                                     (historical_data['AUSPRAEGUNG'] == input['AUSPRAEGUNG']) & 
                                                     (historical_data['MONAT'] == input['MONAT']) &
                                                     (historical_data['JAHR'] == input['JAHR'] - 1)]['WERT'].values[0]
    
    input['VERAEND_VORMONAT_PROZENT'] = (input['VORJAHRESWERT'] - historical_data[(historical_data['MONATSZAHL'] == input['MONATSZAHL']) & 
                                                 (historical_data['AUSPRAEGUNG'] == input['AUSPRAEGUNG']) & 
                                                 (historical_data['MONAT'] == last_month)]['WERT'].values[0]) / input['VORJAHRESWERT'] * 100
    
    input['VERAEND_VORJAHRESMONAT_PROZENT'] = (input['VORJAHRESWERT'] - historical_data[(historical_data['MONATSZAHL'] == input['MONATSZAHL']) & 
                                                     (historical_data['AUSPRAEGUNG'] == input['AUSPRAEGUNG']) & 
                                                     (historical_data['JAHR'] == input['JAHR'] - 1)]['WERT'].values[0]) / input['VORJAHRESWERT'] * 100
    
    input['ZWOELF_MONATE_MITTELWERT'] = historical_data[(historical_data['MONATSZAHL'] == input['MONATSZAHL']) & 
                                                               (historical_data['AUSPRAEGUNG'] == input['AUSPRAEGUNG'])].tail(12)['WERT'].mean()
    
    # convert to DF and append to historical data for feature calculation
    new_instance_df = pd.DataFrame([input])
    historical_data = pd.concat([historical_data, new_instance_df], ignore_index=True)
    
    # Calculate lagging features for the new instance
    lags = [3, 5, 10]
    calculate_lagging_features(historical_data, ['MONATSZAHL', 'AUSPRAEGUNG', 'MONAT'], 'WERT', lags)

    new_instance_df = historical_data.iloc[[-1]]

    # encode categorical variables
    new_instance_df['MONATSZAHL_ENC'] = le_monatszahl.transform([new_instance_df['MONATSZAHL']])[0]
    new_instance_df['AUSPRAEGUNG_ENC'] = le_auspraegung.transform([new_instance_df['AUSPRAEGUNG']])[0]

    # return the new instance with the calculated features
    return new_instance_df.drop(['WERT', 'MONATSZAHL', 'AUSPRAEGUNG'], axis=1)



@app.post("/predict")
def predict(input: PredictionInput):
    new_instance = prepare_new_instance(input.dict())
    
    # Predict the future value
    prediction = model.predict(new_instance)
    
    return {'prediction': prediction[0]}
