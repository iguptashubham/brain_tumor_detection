import sys
import os
sys.path.append(r'C:\Users\gupta\OneDrive\Desktop\Projects\brain_tumor\Brain-tumor-detection')

from braintumordetection.augmentation import data_augmentation
from braintumordetection.modeling.predict import modelpred

from fastapi import FastAPI,UploadFile

app = FastAPI(title='Brain Tumor Detection')

@app.get('/data_augmentation')
def run_data_augmentation(file_dir: str, generation: int):
    data_augmentation(file_dir=file_dir, n_generation=generation)
    return {'message': 'Augmentation Successfully', 'generations': generation}

@app.post(path='/predict')
def get_predict(imgpath:str,model_path:str=r"C:\Users\gupta\OneDrive\Desktop\Projects\brain_tumor\Brain-tumor-detection\models",model:str='model1.keras'):
    pred=modelpred(model_path=model_path,modelname=model,image_path=imgpath)
    return {'prediction':pred}