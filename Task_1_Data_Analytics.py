#!/usr/bin/env python
# coding: utf-8

# # THE SPARKS FOUNDATION
# 
# # VIVISHA SINGH
# # Data Science and Business Analytics
# # Task-1
# # Prediction using Supervised ML
# # Predict the percentage of an student based on the no. of study hours.

# In[ ]:



#importing all libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[9]:



mydataset=pd.read_csv("http://bit.ly/w-data") #reading the csv file or dataset


# In[10]:


mydataset


# In[11]:



mydataset.head() #viewing top 5 rows of data


# In[12]:


# checking for NULL values
mydataset.isnull().sum()


# In[14]:



#checking for linearity
plt.scatter(mydataset['Hours'],mydataset['Scores'])
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# In[15]:


#correlation matrix
cor=mydataset.corr()
cor   # variables positively /highly co-related


# # Training the Model
# 1) Splitting the Data

# In[20]:



# Defining X and y from the Data
X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values

# Spliting the Data in two
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)


# # 2) Fitting the Data into the model

# In[21]:



regression = LinearRegression()
regression.fit(train_X, train_y)
print("---------Model Trained---------")


# # Predicting the Percentage of Marks

# In[22]:



pred_y = regression.predict(val_X)
prediction = pd.DataFrame({'Hours': [i[0] for i in val_X], 'Predicted Marks': [k for k in pred_y]})
prediction


# # Comparing the Predicted Marks with the Actual Marks

# In[23]:


compare_scores = pd.DataFrame({'Actual Marks': val_y, 'Predicted Marks': pred_y})
compare_scores


# # Visually Comparing the Predicted Marks with the Actual Marks

# In[24]:



plt.scatter(x=val_X, y=val_y, color='blue')
plt.plot(val_X, pred_y, color='Black')
plt.title('Actual vs Predicted', size=20)
plt.ylabel('Marks Percentage', size=12)
plt.xlabel('Hours Studied', size=12)
plt.show()


# # CHECKING ACCURACY OF THE MODEL

# In[28]:


from sklearn import metrics
from sklearn.metrics import mean_absolute_error


# In[29]:


print('Mean absolute error: ',metrics.mean_absolute_error(val_y,pred_y)) #less error


# # What will be the predicted score of a student if he/she studies for 9.25 hrs/ day?

# In[30]:


hours = [9.25]
answer = regression.predict([hours])
print("Score = {}".format(round(answer[0],3)))


# # According to the regression model if a student studies for 9.25 hours a day he/she is likely to score 93.89 marks.
