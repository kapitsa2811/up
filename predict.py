from keras.models import model_from_json
import os
import pandas as pd
import numpy as np
from numpy import genfromtxt



testData=pd.read_csv("/home/kapitsa/PycharmProjects/upwork/splitSave/test.csv",header=None)
print("\n\t shape=",testData.shape)

#labels=pd.read_csv("/home/kapitsa/PycharmProjects/upwork/splitSave/labels.csv",header=None)
labels = genfromtxt("/home/kapitsa/PycharmProjects/upwork/splitSave/labels.csv", delimiter=',')

json_file=open("/home/kapitsa/PycharmProjects/upwork/models//model.json",'r')
loaded_model_json=json_file.read()
json_file.close()
loaded_model=model_from_json(loaded_model_json)
loaded_model.load_weights("/home/kapitsa/PycharmProjects/upwork/models//model.h5")
y_pred=loaded_model.predict(testData)

y_pred =(y_pred>0.5)

print("\n\t y_pred =",y_pred.shape,"\t labels=",labels.shape)

with open("/home/kapitsa/PycharmProjects/upwork/splitSave//pred.csv","w") as fout:
    np.savetxt(fout,y_pred,delimiter=",")

y_pred = genfromtxt("/home/kapitsa/PycharmProjects/upwork/splitSave//pred.csv", delimiter=',')

results=np.hstack([labels,y_pred])
print("\n\t 1.labels=",labels[0:10])
print("\n\t 2.pred=",y_pred[0:10])
print("\n\t 2.results=",results[0:10])


# with open("/home/kapitsa/PycharmProjects/upwork/splitSave//fResult.csv","w") as fout:
#     np.savetxt(fout,results,delimiter=",")

pred=pd.DataFrame(data=results)
print("\n\t head=",pred.head())

pred.to_csv("/home/kapitsa/PycharmProjects/upwork/results//final.csv")

#testData=pd.read_csv("/home/kapitsa/PycharmProjects/upwork/dataset/SAS Shootout 2010.csv")

#testData=testData[]


