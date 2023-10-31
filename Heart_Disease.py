#!/usr/bin/env python
# coding: utf-8

# In[25]:


#!pip install graphviz
#!pip install pydotplus
#!pip install six
#!pip install --upgrade scikit-learn==0.20.3
import pandas as pd
import pydotplus
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier ,export_graphviz
from sklearn.preprocessing import StandardScaler
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.metrics import accuracy_score ,confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data=pd.read_csv("dataset.csv")


# In[3]:


data.head()


# In[4]:


data.describe()


# In[5]:


countFemale = len(data[data.sex == 0])
print("Female Count:",countFemale)
countMale = len(data[data.sex == 1])
print("Male Count:",countMale)


# In[6]:


data.shape


# In[7]:


corr=data.corr()
corr


# # Visualize the number of patients having a heart disease and not having a heart disease.

# In[8]:


sns.countplot(data.target, palette=["green","red"])
plt.title("[0] Do not have Herat Disease [1] Does have Heart Disease ")


# # Visualize the age and weather patient has disease or not

# In[9]:


plt.figure(figsize=(18,10))
sns.countplot(x="age",hue="target",data=data,palette=["green","red"])
plt.legend(["Does not have heart disease","Have heart disease"])
plt.title("Heart Disease for ages of patient")
plt.xlabel("Age")
plt.ylabel("Frequancy")
plt.plot()


# # Visualize correlation between all features using a heat map

# In[10]:


plt.figure(figsize=(18,10))
sns.heatmap(corr,annot=True)
plt.pot()


# # a. Build a simple logistic regression model
# i. Divide the dataset in 70:30 ratio
# ii. Build the model on train set and predict the values on test set
# iii. Build the confusion matrix and get the accu racy score

# In[11]:


x=data.iloc[:,:-1]
x


# In[12]:


y=data.iloc[:,-1]
y


# In[13]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3,random_state=0)


# In[9]:


x_train


# In[10]:


x_test


# In[11]:


y_train


# In[12]:


y_test


# In[14]:


clf=LogisticRegression()
#clf=StandardScaler()


# In[15]:


clf.fit(x_train,y_train)


# In[16]:


y_pred=clf.predict(x_test)


# In[17]:


x_test


# In[18]:


y_pred


# In[19]:


log_score=accuracy_score(y_test,y_pred)


# In[20]:


log_score


# In[21]:


log_cm=confusion_matrix(y_test,y_pred)


# In[22]:


log_cm


# # Decision Tree:
# a. Build a decision tree model
# i. Divide the dataset in 70:30 ratio
# ii. Build the model on train set and predict the values on test set
# iii. Build the confusion matrix and calculate the accuracy
# iv. Visualize the decision tree using the graphviz package

# In[26]:


clf=DecisionTreeClassifier(criterion='entropy',random_state=0)


# In[27]:


clf.fit(x_train,y_train)


# In[28]:


y_pred=clf.predict(x_test)


# In[29]:


y_pred


# In[31]:


acc=accuracy_score(y_pred,y_test)
#accuracies['Decision Tree'] = acc
print("Decision Tree Test Accuracy {:.2f}%".format(acc))


# In[32]:


cm=confusion_matrix(y_test,y_pred)
cm


# In[ ]:




