import pandas as pd
import os

cwd=os.getcwd()
dataFolder=os.path.join(cwd,"dataset")
fileName=["diabetic_data.csv","SAS Shootout 2010.csv"]

filePath1=os.path.join(dataFolder,fileName[0])
filePath2=os.path.join(dataFolder,fileName[1])

df1=pd.read_csv(filePath1)
#df2=pd.read_csv(filePath2)

print("\n\t df1=",df1.shape)
#print("\n\t df2=",df2.shape)

print("\n\t features=",df1.columns)

'''
    set the data type
'''

numType=['race','gender','age','weight','time_in_hospital','num_lab_procedures','num_procedures','num_medications','number_outpatient','number_emergency',
,'number_inpatient','diag_1','diag_2','diag_3','number_diagnoses']
catType=['encounter_id','patient_nbr','admission_type_id','discharge_disposition_id','admission_source_id','payer_code','medical_specialty',
         'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
         'acetohexamide', 'glipizide', 'glyburide''tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol',
         'troglitazone', 'tolazamide', 'examide', 'citoglipton''insulin', 'glyburide-metformin', 'glipizide-metformin',
         'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone',
         'change	readmitted', 'diabetesMed']

for indx,ele in enumerate(df1.columns):
    print("\n\t indx:",indx,"\t ele:",ele,"\t type=",df1[df1.columns[indx]].dtype)







