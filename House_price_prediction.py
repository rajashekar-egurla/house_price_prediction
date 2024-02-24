#!/usr/bin/env python
# coding: utf-8

# In[102]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[103]:


df=pd.read_csv("House_price.csv")


# In[104]:


df.head()


# In[105]:


df.isnull().sum()


# In[106]:


df.duplicated().sum()


# In[107]:


from sklearn.preprocessing import MinMaxScaler
scaling=MinMaxScaler()
scaling.fit_transform(df[['price','area']])


# In[108]:


df['area'], df['price']


# In[109]:


plt.figure(figsize=(12,8))
plt.scatter(df['area'], df['price'])
plt.show()


# In[110]:


plt.figure(figsize=(15,10))
plt.scatter(df['bedrooms'],df['price'],color='green',label='bedrooms')
plt.ylabel='price'
plt.show()


# In[111]:


plt.scatter(df['bathrooms'],df['price'],color='r',label='bathrooms')
plt.ylabel='price'
plt.show()


# In[112]:


df['area'].corr(df['price'])


# In[113]:


df.corr()


# In[ ]:



    


# In[ ]:





# In[114]:


from sklearn.preprocessing import LabelEncoder


# In[115]:


df


# In[116]:


lb=LabelEncoder()


# In[117]:


df.dtypes


# In[118]:


ob_data=['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea','furnishingstatus']
for i in ob_data:
    df[i]=lb.fit_transform(df[i])


# In[119]:


df.head()


# In[120]:


df.describe()


# In[121]:


import seaborn as sns


# In[122]:


sns.boxplot(df['bedrooms'],df['price'])


# In[123]:


sns.boxplot(df['bathrooms'],df['price'])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[124]:


def eval(model):
    print("The traing score is ",model.score(x_train,y_train),end='\n')
    print("The test score is ",model.score(x_test,y_test))


# In[125]:


def metric_score(y_pred,y_test):
    print("The mean absolute error is ",mean_absolute_error(y_pred,y_test))
    print("The mean square error id ",mean_squared_error(y_pred,y_test))
    print("the R2 score is ",r2_score(y_pred,y_test))


# splitting the data

# In[126]:


x=df.drop('price',axis=1)
y=df['price']


# In[127]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error,confusion_matrix,r2_score


# In[128]:


x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


# In[129]:


print(x_train.shape,"",type(x_train))
print(y_train.shape,"\t ",type(y_train))
print(x_test.shape,"",type(x_test))
print(y_test.shape,"\t ",type(y_test))


# In[130]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()


# In[131]:


x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)


# In[132]:


y_train=scaler.fit_transform(y_train)
y_test=scaler.transform(y_test)


# In[ ]:





# In[ ]:


from sklearn.tree import DecisionTreeRegressor
DTmodel=DecisionTreeRegressor(criterion='squared_error',
    splitter='best',
    max_depth=10,
    min_samples_split=12,
    random_state=42)


# In[133]:


DTmodel.fit(x_train,y_train)


# In[134]:


y_pred=DTmodel.predict(x_test)


# In[135]:


y_pred


# In[136]:


eval(DTmodel)


# In[137]:


metric_score(y_pred,y_test)


# In[138]:


from sklearn.ensemble import RandomForestRegressor


# In[139]:


RFmodel=RandomForestRegressor(n_estimators=100,
    criterion='squared_error',
    max_depth=10,
    min_samples_split=12,random_state=42)


# In[140]:


RFmodel.fit(x_train,y_train)


# In[141]:


eval(RFmodel)


# In[142]:


y_pred=RFmodel.predict(x_test)


# In[143]:


y_pred


# In[144]:


y_test


# In[145]:


metric_score(y_pred,y_test)


# In[146]:


from sklearn.linear_model import LinearRegression


# In[147]:


LRmodel=LinearRegression()


# In[148]:


LRmodel.fit(x_train,y_train)


# In[149]:


eval(LRmodel)


# In[150]:


y_pred=LRmodel.predict(x_test)


# In[151]:


y_pred


# In[152]:


y_test


# In[153]:


metric_score(y_pred,y_test)


# In[154]:


df


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




