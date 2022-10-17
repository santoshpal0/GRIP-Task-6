#!/usr/bin/env python
# coding: utf-8

# ### Prediction using Decision Tree Algorithm
# ● Create the Decision Tree classifier and visualize it graphically.
# 
# ● The purpose is if we feed any new data to this classifier, it would be able to predict the right class accordingly.

# #### Import Libraries 

# In[22]:


#importing all the required libraries
import numpy as np
import pandas as pd
import sklearn.metrics as sm
import seaborn as sns
import matplotlib.pyplot as mt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report


# #### Iris Data Set:
# Load the csv file and define the input and output variables

# In[25]:


#load the csv file into a new pandas dataframe
iris = pd.read_csv("C:\\Users\\user\\Downloads\\Iris.csv")


# In[26]:


iris.head()


# In[27]:


iris.info()


# In[28]:


iris.describe()


# ### Input data visualization

# In[29]:


sns.pairplot(iris, hue='Species')


# ### Finding the correlation matrix

# In[31]:


iris.corr()


# In[33]:


#In next step, using heatmap to visulaize data
sns.heatmap(iris.corr())


# #### We observed that: (i)Petal length is highly related to petal width (ii)Sepal length is not related to sepal width

# ###  Data preprocessing

# In[34]:


target=iris['Species']
df=iris.copy()
df=df.drop('Species', axis=1)
df.shape


# In[37]:


#defingi the attributes and labels
X=iris.iloc[:, [0,1,2,3]].values
le=LabelEncoder()
iris['Species']=le.fit_transform(iris['Species'])
y=iris['Species'].values
iris.shape


# ###  Trainig the model
# We will now split the data into test and train.

# In[38]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print("Traingin split:",X_train.shape)
print("Testin spllit:",X_test.shape)


# In[39]:


#Defining Decision Tree Algorithm
dtree=DecisionTreeClassifier()
dtree.fit(X_train,y_train)
print("Decision Tree Classifier created!")


# ###  Classification Report and Confusion Matrix

# In[40]:


y_pred=dtree.predict(X_test)
print("Classification report:\n",classification_report(y_test,y_pred))


# In[41]:


print("Accuracy:",sm.accuracy_score(y_test,y_pred))


# In[43]:


#The accuracy is 1 or 100% since i took all the 4 features of the iris dataset.


# In[44]:


#confusion matrix
cm=confusion_matrix(y_test,y_pred)
cm


# ###  Visualization of trained model

# In[45]:


#visualizing the graph
mt.figure(figsize=(20,10))
tree=plot_tree(dtree,feature_names=df.columns,precision=2,rounded=True,filled=True,class_names=target.values)


# #### The Descision Tree Classifier is created and is visaulized graphically. Also the prediction was calculated using decision tree algorithm and accuracy of the model was evaluated.

# In[ ]:




