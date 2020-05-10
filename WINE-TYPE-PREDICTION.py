#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns


# # DATA COLLECTION

# In[2]:


train_data=pd.read_csv('winetrain.csv')


# In[3]:


train_data


# In[4]:


train_data.isnull().any()
#checking weather any null value is present column wise or not


# In[5]:


train_data.duplicated().any()


# In[6]:


unique_data=train_data[train_data.duplicated('review_description',keep=False)]


# In[7]:


unique_data.describe()


# In[8]:


unique_data.isnull().any()


# # Pre Processing

# In[9]:


train_review=unique_data['review_description'].str.lower()


# In[10]:


train_review


# In[11]:


train_variety=unique_data['variety'].values


# In[12]:


train_variety


# # Label Encoding

# In[13]:


from sklearn.preprocessing import LabelEncoder


# In[14]:


label_variety=LabelEncoder()


# In[15]:


label_variety.fit(train_variety)


# In[16]:


train_variety_labelled=label_variety.transform(train_variety)


# In[17]:


train_variety_labelled


# # Processing reviews uisng NLP

# In[18]:


from nltk.corpus import stopwords


# In[19]:


from nltk.tokenize import word_tokenize


# In[20]:


stop_words=list(set(stopwords.words('english')))+[',','.','!',';',':','\'s','%']


# In[21]:


stop_words


# In[22]:


for i in range(len(train_review)):
    train_review.iloc[i]=word_tokenize(train_review.iloc[i])
    train_review.iloc[i]=[w for w in train_review.iloc[i] if not w in stop_words]
    train_review.iloc[i]=" ".join(train_review.iloc[i])


# In[23]:


train_review


# # Vectorization

# In[24]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[25]:


vect=TfidfVectorizer()


# In[26]:


vect.fit(train_review)


# In[27]:


train_vector=vect.transform(train_review)


# In[28]:


train_vector=train_vector.toarray()


# # Splitting data into dependent and independent variables

# In[29]:


x=train_vector


# In[30]:


x


# In[31]:


y=train_variety_labelled


# In[32]:


y


# # Train Test Split

# In[33]:


from sklearn.model_selection import train_test_split


# In[34]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=0)


# # Model Training

# In[35]:


from sklearn.ensemble import RandomForestClassifier


# In[36]:


cls=RandomForestClassifier()


# In[37]:


cls.fit(x_train,y_train)


# In[38]:


prediction=cls.predict(x_test)


# # Model Validation

# In[39]:


from sklearn.metrics import classification_report


# In[40]:


print(classification_report(y_test,prediction))


# In[41]:


from sklearn.metrics import confusion_matrix


# In[42]:


import matplotlib.pyplot as plt


# In[43]:


plt.subplots(figsize=(25,20))
sns.heatmap(confusion_matrix(y_test,prediction),annot=True,linewidths=0.0001)


# # Model Prediction

# In[44]:


pred_data=pd.read_csv('winetest.csv')


# In[45]:


pred_data.isnull().any()


# In[52]:


test_review=pred_data['review_description'].str.lower()


# In[54]:


test_review


# In[55]:


for i in range(len(test_review)):
    test_review.iloc[i]=word_tokenize(test_review.iloc[i])
    test_review.iloc[i]=[w for w in test_review.iloc[i] if not w in stop_words]
    test_review.iloc[i]=" ".join(test_review.iloc[i])


# In[56]:


test_review


# In[57]:


test_vector=vect.transform(test_review)


# In[58]:


test_vector=test_vector.toarray()


# In[61]:


result_labelled=cls.predict(test_vector)


# In[62]:


result_labelled


# In[64]:


result=label_variety.inverse_transform(result_labelled)


# In[65]:


result


# In[66]:


result_doc=pd.DataFrame(zip(pred_data['review_description'],result),columns=['review_description','variety'])


# In[67]:


print(result_doc)


# In[68]:


result_doc.to_csv('resultdoc.csv')

