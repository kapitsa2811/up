import pandas as pd
import os

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import glob
import cv2
import os
from keras.models import model_from_json
from keras import backend as K

basePath="/home/kapitsa/PycharmProjects/upwork/kaggle/dataSet//"
basePath2="/home/kapitsa/PycharmProjects/upwork/dataset//"
dataPath3=basePath2+"SAS Shootout 2010.csv"

columns=["Number_of_times pregnant","Plasma_glucose","Diastolic_blood","Triceps_skinfold","2_Hour_serum","Body_mass","Diabetes_pedigree",
         "Age","Class"]


#columns=["SEX","CENSUS_REGION	AGE	MARITAL_STATUS	YEARS_EDUC	HIGHEST_DEGREE	SERVED_ARMED_FORCES	FOODSTAMPS_PURCHASE	TOTAL_INCOME	MORE_THAN_ONE_JOB	WEARS_EYEGLASSES	PERSON_BLIND	WEAR_HEARING_AID	IS_DEAF	PERSON_WEIGHT	TOTALEXP	AMOUNT_PAID_MEDICARE	AMOUNT_PAID_MEDICAID	NUMB_VISITS	DENTAL_CHECKUP	CHOLEST_LST_CHCK	LAST_CHECKUP	LAST_FLUSHOT	LOST_ALL_TEETH	LAST_PSA	LAST_PAP_SMEAR	LAST_BREAST_EXAM	LAST_MAMMOGRAM	BLD_STOOL_TST	SIGMOIDOSCOPY_COLONOSCOPY	WEAR_SEAT_BELT	HIGH_BLOOD_PRESSURE_DIAG	HEART_DISEASE_DIAG	ANGINA_DIAGNOSIS	HEART_ATTACK	OTHER_HEART_DISEASE	STROKE_DIAGNOSIS	EMPHYSEMA_DIAGNOSIS	JOINT_PAIN	CURRENTLY_SMOKE	ASTHMA_DIAGNOSIS	CHILD_BMI	ADULT_BMI	DIABETES_DIAG_BINARY]
#colmns=["ID number", "Diagnosis" ,"radius", "texture","perimeter","area","smoothness","compactness","concavity","concave points","symmetry","fractal dimension"]


print("\n\t is file=",os.path.isfile(dataPath3))
dataset=pd.read_csv(dataPath3)
print(dataset.shape)
#print("\n\t columns=",dataset.columns)


def process_house_attributes(df):
	# initialize the column names of the continuous data
    continuous = ["AGE","TOTAL_INCOME","PERSON_WEIGHT","TOTALEXP","AMOUNT_PAID_MEDICARE","AMOUNT_PAID_MEDICAID","NUMB_VISITS","CHILD_BMI","ADULT_BMI"]

    cs = MinMaxScaler()
    trainContinuous = cs.fit_transform(df[continuous])


    categorical=["SEX","CENSUS_REGION","MARITAL_STATUS","YEARS_EDUC","HIGHEST_DEGREE","SERVED_ARMED_FORCES","FOODSTAMPS_PURCHASE","MORE_THAN_ONE_JOB","WEARS_EYEGLASSES","PERSON_BLIND","WEAR_HEARING_AID","IS_DEAF",
                  "DENTAL_CHECKUP","CHOLEST_LST_CHCK","LAST_CHECKUP","LAST_FLUSHOT","LOST_ALL_TEETH","LAST_PSA","LAST_PAP_SMEAR","LAST_BREAST_EXAM",
                  "LAST_MAMMOGRAM","BLD_STOOL_TST","SIGMOIDOSCOPY_COLONOSCOPY","WEAR_SEAT_BELT","HIGH_BLOOD_PRESSURE_DIAG","HEART_DISEASE_DIAG",
                  "ANGINA_DIAGNOSIS","HEART_ATTACK","OTHER_HEART_DISEASE","STROKE_DIAGNOSIS","EMPHYSEMA_DIAGNOSIS","JOINT_PAIN","CURRENTLY_SMOKE","ASTHMA_DIAGNOSIS","DIABETES_DIAG_BINARY"]


    #categorical=["SEX","CENSUS_REGION"]

    # performin min-max scaling each continuous feature column to
	# the range [0, 1]

    zipBinarizer = LabelBinarizer().fit(df[categorical[0]])
    # zipBinarizer = LabelBinarizer().fit(df[categorical])

    trainX = zipBinarizer.transform(df[categorical[0]])
    #print("\n\t trainX1=", trainX)

    for indx,feature in enumerate(categorical):

        if indx==0:
            continue

        print("\n\t indx=",indx,"\t feature=",feature)
        zipBinarizer = LabelBinarizer().fit(df[categorical[indx]])
        #zipBinarizer = LabelBinarizer().fit(df[categorical])

        trainCategorical = zipBinarizer.transform(df[categorical[indx]])
        trainX = np.hstack([trainX,trainCategorical])

    trainX = np.hstack([ trainContinuous,trainX])
    trainX=pd.DataFrame(data=trainX)
    print("\n\t trainX=",trainX.shape)
    #print("\n\t columns=", trainX.shape)
    return trainX

trainX=process_house_attributes(dataset)

print("\n\t trainX=",trainX.columns)

X= trainX.iloc[:,0:171]
y= trainX.iloc[:,172]


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print("\n\t X_test--->",type(X_test))
with open("/home/kapitsa/PycharmProjects/upwork/splitSave//test.csv","w") as fout:
    np.savetxt(fout,X_test,delimiter=",")

with open("/home/kapitsa/PycharmProjects/upwork/splitSave//labels.csv","w") as fout:
    np.savetxt(fout,y_test,delimiter=",")


from keras import Sequential
from keras.layers import Dense


classifier = Sequential()
#First Hidden Layer
classifier.add(Dense(256, activation='relu', kernel_initializer='random_normal', input_dim=X.shape[1]))

#Second  Hidden Layer
classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal'))

#Output Layer
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))


#Compiling the neural network
#classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

classifier.compile(loss='binary_crossentropy',
          optimizer= "adam",
          metrics=[f1])

classifier.fit(X_train,y_train, batch_size=1, epochs=300)

eval_model=classifier.evaluate(X_train, y_train)
print("\n\t eval_model=",eval_model)

y_pred=classifier.predict(X_test)


#print("\n\t y_test=",type(y_test),"\t y_pred=",type(y_pred))
y_pred =(y_pred>0.5)
#y_pred=pd.Series(y_pred)

'''
    saving model
'''

model_json=classifier.to_json()

with open("/home/kapitsa/PycharmProjects/upwork/models//model.json","w") as json_file:
    json_file.write(model_json)

classifier.save_weights("/home/kapitsa/PycharmProjects/upwork/models//model.h5")

'''
'''

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("2.cm=",cm)

TP=cm[0][0]
TN=cm[1][1]
FP=cm[0][1]
FN=cm[1][0]
print("\n\t TP=",cm[0][0])
print("\n\t TN=",cm[1][1])
print("\n\t FP=",cm[0][1])
print("\n\t FN=",cm[1][0])

print("\n\t Accuracy=",(TP+TN)/(TP+TN+FP+FN))
#accuracy=(cm[0][0]+cm[3])/(cm[0]+cm[3]+cm[1]+cm[2])

print("\n\t cm=",cm[0])
print("\n\t cm=",cm[1])
print("\n\t y_test=",type(y_test),"\t y_pred=",type(y_pred))
print("\n\t shape y_test=",y_test.shape,"\t y_pred=",y_pred.shape)

temp=np.hstack([y_test.to_numpy(),y_pred])
ans=pd.DataFrame(data=temp)

ans.to_csv("/home/kapitsa/PycharmProjects/upwork/results//result.csv")













