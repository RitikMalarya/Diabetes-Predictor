#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# In[2]:


data = pd.read_csv("dataset.csv")


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data.info()


# In[6]:


data.dtypes


# In[7]:


data.describe().T


# In[8]:


data.isnull().sum()


# In[9]:


new_data = data.copy(deep=True)


# In[10]:


# Replacing zero values with NaN
new_data[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] = new_data[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].replace(0, np.NaN)


# In[11]:


# count of NAN
new_data.isnull().sum()


# In[12]:


# replacing Nan value with mean values.dataset_new["Glucose"].fillna(dataset_new["Glucose"].mean(), inplace = True)
new_data["BloodPressure"].fillna(new_data["BloodPressure"].mean(), inplace = True)
new_data["SkinThickness"].fillna(new_data["SkinThickness"].mean(), inplace = True)
new_data["Insulin"].fillna(new_data["Insulin"].mean(), inplace = True)
new_data["BMI"].fillna(new_data["BMI"].mean(), inplace = True)


# In[13]:


#new_data = new_data.reset_index()


# In[14]:


new_data.head()


# In[15]:


# Selecting features  which are responsibwe for Diabetes- [Glucose,BloodPressure, BMI, Age]
x = data.iloc[:, [1,2, 5, 7]].values
y = data.iloc[:, 8].values


# In[16]:


from sklearn.model_selection import train_test_split


# In[17]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


# In[18]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[19]:


# Logistic Regression Algorithm
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(random_state = 42)
logreg.fit(x_train, y_train)


# In[20]:


import pickle
filename = 'logreg-model.pkl'
pickle.dump(logreg,open(filename,'wb'))


# In[ ]:




