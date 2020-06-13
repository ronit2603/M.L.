
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


churn = pd.read_csv("churn.csv")
churn.head()


# In[3]:


churn.describe()


# In[4]:


list(churn.columns)


# In[5]:


churn.info()


# In[6]:


churn1=pd.get_dummies(churn[['Int\'l Plan','VMail Plan','Churn?']])
churn_new=pd.concat([churn1,churn],axis=1)
churn_new.head()


# In[7]:


col=['State','Area Code','Phone','Int\'l Plan','VMail Plan','Churn?']
churn_new= churn_new.drop(col,axis=1)
churn_new.head()


# In[8]:


from sklearn.model_selection import train_test_split


# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
churn_new.plot(kind='box',subplots=True,layout=(6,5),sharex=False,sharey=False)
plt.show()


# In[11]:


X=churn_new[['Day Mins','Day Calls','Day Charge','Eve Mins','Eve Calls','Eve Charge','Night Mins']]
Y=churn_new[['Churn?_False.']]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y)
X_test


# In[32]:


X.hist(figsize=(8,7))
plt.show()


# In[12]:


churn.plot(kind='scatter',x='Intl Calls',y='Intl Charge')
plt.plot()


# In[13]:


import seaborn as sns
df = sns.heatmap(X)


# In[15]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit_transform(X)


# In[16]:


from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors=3) 


# In[17]:


knn.fit(X_train,Y_train)


# In[18]:


knn.score(X_train,Y_train)


# In[19]:


knn.score(X_test,Y_test)


# In[20]:


knn.score(X,Y)


# In[21]:


from sklearn.ensemble  import RandomForestClassifier
from sklearn.metrics import accuracy_score
rfc = RandomForestClassifier()
rfc.fit(X_train,Y_train)
y_pred = rfc.predict(X_test)


# In[22]:


acc = accuracy_score(Y_test,y_pred)
print(acc)


# In[23]:


rfc.score(X_train,Y_train)


# In[24]:


rfc.score(X_test,Y_test)


# In[25]:


y=churn_new['Churn?_False.']
y


# In[29]:


feat_importances = pd.Series(rfc.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')

