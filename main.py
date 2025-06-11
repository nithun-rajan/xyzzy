import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,classification_report
from sklearn.metrics import confusion_matrix
dataframe = pd.read_csv('Churn_Data.csv')
columns_pallete={"Germany": "#716D31", "France": "#0A5CC7", "Spain": "#F17105"}

#sns.displot(data=churners, x="Age", hue='Geography', kde=True, height=7, palette=palette)
#sns.displot(data=nonchurners, x="Age", hue='Geography', kde=True, height=10, palette=palette)
#sns.displot(df_churn, x="EstimatedSalary", hue="Exited", multiple="stack", height=10)
#sns.displot(churners, x="CreditScore", hue="Geography", multiple="stack", height=10, palette=palette)


dataframe_palette=pd.DataFrame(columns_pallete.items(), columns=['Geography', 'Color'])


dataframe_balanced = pd.DataFrame(dataframe['Exited'].value_counts(normalize=False))

dataframe_balanced=pd.DataFrame(dataframe['Exited'].value_counts(normalize=False))##taking some excited column columns
dataframe_balanced=dataframe_balanced.reset_index().rename(columns = {'index':'Exited','Exited':'Count'})#creating the balanced dataframe
dataframe_balanced
 
label = LabelEncoder()#creating the label encoder
dataframe['Geography']=label.fit_transform(dataframe['Geography'])

label_encoder = dict(zip(label.classes_, label.transform(label.classes_)))
print(label_encoder)

#XGBOOST

x_dataset = dataframe.drop(["Exited","Gender","Surname"], axis=1)
y_datset = dataframe['Exited']

x_train , x_test , y_train , y_test = train_test_split(x_dataset,y_datset,test_size=0.25 , stratify=y_datset, random_state=42)

xgb_model= xgb.XGBClassifier(gamma= 1.0,learning_rate= 0.15,max_depth= 7, n_estimators=100)

xgb_model.fit(x_train,y_train)

pred_values1= xgb_model.predict(x_test)
xgb_accuracy = accuracy_score(y_test, pred_values1) 
print("The accuracy score for XGboost is: ",xgb_accuracy )
classification= classification_report(y_test,pred_values1)
print(classification)
matrix=confusion_matrix(y_test,pred_values1)
print(matrix)

#RANDOM FOREST


random_forest= RandomForestClassifier(criterion ='gini', n_estimators=100,random_state = 10)#creating the model using sklearn
random_forest.fit(x_train, y_train)#fitting the x and y value
 
pred_values2= random_forest.predict(x_test)# Evaluating on Training set
 
accuracy=accuracy_score(y_test, pred_values2) # Display accuracy score
print("The accuracy score RANDOM FOREST is: ", accuracy)
classification= classification_report(y_test,pred_values2)
print(classification)
matrix=confusion_matrix(y_test,pred_values2)
print(matrix)

