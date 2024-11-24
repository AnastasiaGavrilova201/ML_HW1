from fastapi import FastAPI, File, UploadFile
import uvicorn
from pydantic import BaseModel
import pickle 
import pandas as pd
from typing import List
import sklearn
from fastapi.responses import FileResponse

app = FastAPI()

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float
    
with open("best_model.pickle", "rb") as f:
    model = pickle.load(f)

@app.post('/predict_item')
def predict(data: Item):
    data = data.dict()
    my_data = pd.DataFrame([data])
    df = my_data.drop(columns = ['torque', 'name', 'selling_price'])
    #km
    df['mileage'] = df['mileage'].str.replace(' kmpl', '').str.replace(' km/kg', '').astype('float')
    #CC
    df['engine'] = df['engine'].str.replace(' CC', '').astype('float')
    #bhp
    df['max_power'] = df['max_power'].str.replace(' bhp', '').astype('float')
    y_pred = model.predict(df)[0]
    return {'price_predict' : y_pred}

@app.post("/predict_items")
def predict_items(file: UploadFile = File(...)):
    df0 = pd.read_csv(file.file)
    df = df0.drop(columns = ['torque', 'name', 'selling_price'])
    #km
    df['mileage'] = df['mileage'].str.replace(' kmpl', '').str.replace(' km/kg', '').astype('float')
    #CC
    df['engine'] = df['engine'].str.replace(' CC', '').astype('float')
    #bhp
    df['max_power'] = df['max_power'].str.replace(' bhp', '').astype('float')
    y_pred = model.predict(df)
    df_final = pd.concat([df0, pd.DataFrame([{'price_predict' : y_pred}])], axis = 1)
    df_final.to_csv("res.csv", index=False)
    return FileResponse("res.csv", media_type="text/csv", filename="res.csv")
