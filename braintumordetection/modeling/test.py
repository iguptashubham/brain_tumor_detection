import sys
sys.path.append(r'C:\Users\gupta\OneDrive\Desktop\Projects\brain_tumor\Brain-tumor-detection')
import numpy as np
from tensorflow.keras.models import load_model
from braintumordetection.modeling.model import testdf
import os

def modeltest(path,modelname,test):
    model_path = os.path.join(path,modelname)
    model=load_model(model_path)
    testinfo = model.predict(test,steps = len(test.filenames), verbose=1)
    loss,accuracy = model.evaluate(test)
    return loss,accuracy,testinfo

x,y,test = modeltest(path=r'C:\Users\gupta\OneDrive\Desktop\Projects\brain_tumor\Brain-tumor-detection\models',modelname='model1.keras',test = testdf())

print("loss=", x, "accuracy=", y, "pred=", np.argmax(test,axis=1))
print('----')