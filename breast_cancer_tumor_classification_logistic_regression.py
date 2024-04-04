#!/usr/bin/env python
# coding: utf-8

# Classification of breast tumor cancer with logistic regression model.

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


df_data = pd.read_csv('data_breast_cancer_logistic_regression.csv')
df_data.head(5)


# In[4]:


df_data.info()


# In[5]:


df_data.isnull().sum()


# In[6]:


df_data.drop('Unnamed: 32', axis=1,inplace=True)


# In[7]:


df_data['diagnosis'].unique()


# In[8]:


df_data['diagnosis'].replace(to_replace='M',value = 1, inplace=True)
df_data['diagnosis'].replace(to_replace='B',value = 0, inplace=True)


# In[9]:


df_data['diagnosis'].unique()


# In[23]:


df_data['diagnosis'].count()


# In[24]:


# Contar la cantidad de ceros y unos en la columna "diagnosis"
diagnostico_ceros = df_data["diagnosis"].value_counts()[0]
diagnostico_unos = df_data["diagnosis"].value_counts()[1]

# Imprimir los resultados
print(f"Cantidad de diagnósticos con valor 0: {diagnostico_ceros}")
print(f"Cantidad de diagnósticos con valor 1: {diagnostico_unos}")


# The diagnosis variable has 569 rows composed by 0's and 1's. We have 357 ceros and 212 number one. This is two check that owr two clases are in balance and they are. 

# In[10]:


df_data_processing = df_data.copy()
df_data_processing


# Analisis de correlación

# In[11]:


import matplotlib.pyplot as plt


# In[12]:


fig = plt.figure(figsize=(15,9))
df_data_processing.corr()['diagnosis'].sort_values(ascending=True).plot(kind='bar')
plt.show()


# Escalando nuestros datos

# In[13]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_data_processing_scaled = scaler.fit_transform(df_data_processing)


# In[14]:


df_data_processing_scaled = pd.DataFrame(df_data_processing_scaled)
df_data_processing_scaled.head(5)


# In[15]:


df_data_processing_scaled.columns = df_data_processing.columns
df_data_processing_scaled.head(5)


# In[16]:


import seaborn as sns


# In[ ]:


fig = plt.figure(figsize=(10,10))
sns.pairplot(data=df_data,hue='diagnosis')
plt.show()


# Entrenamiento del modelo de regresion logística binomial

# In[17]:


X = df_data_processing_scaled.drop('diagnosis', axis=1)
y = df_data_processing_scaled['diagnosis'].values


# In[18]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3,random_state=42)


# In[19]:


X_train.shape


# In[20]:


X_test.shape


# In[21]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
result = model.fit(X_train,y_train)


# In[22]:


from sklearn import metrics 
prediction_test = model.predict(X_test)
print(metrics.accuracy_score(y_test,prediction_test))


# We obtain 0.97 accuracy score, so our model performes good. 

# Model evaluation

# In[25]:


model.predict_proba(X_test)


# In[26]:


model.coef_


# In[27]:


model.feature_names_in_


# In[28]:


weights = pd.Series(model.coef_[0], index = X.columns.values)
print(weights.sort_values(ascending=False)[:10].plot(kind='bar'))


# The variable with more weight in our model is concave points_worst.

# In[29]:


print(weights.sort_values(ascending=False)[-10:].plot(kind='bar'))


# In[30]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# In[33]:


cm = confusion_matrix(y_test,prediction_test, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix = cm , display_labels=model.classes_)
disp.plot(cmap='pink')
plt.show()


# In this project, we developed a classification model for breast tumors using logistic regression. We trained the model on a dataset of 398 records, achieving an accuracy score of 0.97. Subsequently, the model was evaluated on a separate testing set of 171 records. Of these, the model correctly predicted 108 records as negative (0/Bening) and 59 records as positive (1/Malign).
# 

# In[ ]:




