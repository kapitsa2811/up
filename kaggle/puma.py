import pandas as pd
basePath="/home/kapitsa/PycharmProjects/upwork/kaggle/dataSet//"
dataPath3=basePath+"pima-indians-diabetes.data.csv"

columns=["Number_of_times pregnant","Plasma_glucose","Diastolic_blood","Triceps_skinfold","2_Hour_serum","Body_mass","Diabetes_pedigree",
         "Age","Class"]

#colmns=["ID number", "Diagnosis" ,"radius", "texture","perimeter","area","smoothness","compactness","concavity","concave points","symmetry","fractal dimension"]

dataset=pd.read_csv(dataPath3,names=columns)
print(dataset.head())

X= dataset.iloc[:,0:8]
y= dataset.iloc[:,8]


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


from keras import Sequential
from keras.layers import Dense


classifier = Sequential()
#First Hidden Layer
classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal', input_dim=8))

#Second  Hidden Layer
classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal'))

#Output Layer
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))


#Compiling the neural network
classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])

classifier.fit(X_train,y_train, batch_size=10, epochs=100)

eval_model=classifier.evaluate(X_train, y_train)
print("\n\t eval_model=",eval_model)

y_pred=classifier.predict(X_test)
y_pred =(y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)













