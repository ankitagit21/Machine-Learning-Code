#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


digits = load_digits()


# In[3]:


print("Image Data shape ", digits.data.shape)
print("Label Data shape ", digits.target.shape)


# # Plotting Graph

# In[6]:


plt.figure(figsize=(20,4))
for index, (image,label) in enumerate (zip(digits.data[0:5], digits.target[0:5])):
    plt.subplot(1, 5, index+1)
    plt.imshow(np.reshape(image,(8,8)), cmap= plt.cm.gray)
    plt.title('Training :%i\n' %label, fontsize = 20)


# # Model Training

# In[7]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target,test_size=0.23,random_state=2)


# In[8]:


print (x_train.shape)


# In[9]:


print (x_test.shape)


# In[11]:


print (y_train.shape)


# In[12]:


print (y_test.shape)


# In[14]:


from sklearn.linear_model import LogisticRegression


# In[15]:


logisticreg = LogisticRegression()


# In[17]:


logisticreg.fit(x_train, y_train)


# In[19]:


print (logisticreg.predict(x_test[0:10]))


# In[20]:


predictions = logisticreg.predict(x_test)


# In[21]:


score = logisticreg.score(x_test, y_test)
print(score)


# # Creating Confusion Metrics for the Model

# In[22]:


cm = metrics.confusion_matrix(y_test,predictions)
print(cm)


# # Creating Graphical Representation for the Model

# In[26]:


plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True,fmt='.3f', linewidths=.5, square= True, cmap='Blues_r')
plt.ylabel('Actual Label')
plt.xlabel('predicted Label')
all_sample_title = 'Accuracy Score:{0}'.format(score)
plt.title(all_sample_title, size =15)


# # Model Prediction with new data 

# In[27]:


index = 0
classifiedIndex = []
for pred, actual in zip(predictions, y_test):
    if pred == actual:
        classifiedIndex.append(index)
    index+=1
plt.figure(figsize=(20,23))
for plotindex, wrong in enumerate(classifiedIndex[0:4]):
    plt.subplot(1,4,plotindex+1)
    plt.imshow(np.reshape(x_test[wrong],(8,8)),cmap=plt.cm.gray)
    plt.title("Predicted {} :: Actual {}".format(predictions[wrong],y_test[wrong]),fontsize = 20)


# In[ ]:




