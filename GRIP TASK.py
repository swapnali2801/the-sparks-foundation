#!/usr/bin/env python
# coding: utf-8

# # GRIP : The Sparks Foundation

# # Swapnali Dattatraya Patil

# ## TASK: Prediction using Supervised ML 

# ## Linear Regression with Python Scikit Learn
# In this section we will see how the Python Scikit-Learn library for machine learning canbe used to
# implement regression functions. We will start with simple linear regression involving two variables.

# ## Problem statement
# In this regression task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied. This is a simple linear regression task as it involves just two variables.

# ## To predict:
# What will be predicted score if a student studies for 9.25 hrs/ day?

# #### given dataset : http://bit.ly/w-data

# In[7]:


# Importing all libraries required in this notebook
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


# Reading data from remote link
url="https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
s_data = pd.read_csv(url)
print("Data imported successfully")
s_data.head(10)


# In[9]:


# Plotting the distribution of scores
s_data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# #### From the graph above, we can clearly see that there is a positive linear relation between the number of hours studied and percentage of score.

# ### Preparing the data
# The next step is to divide the data into "attributes" (inputs) and "labels" (outputs).

# In[11]:


X = s_data.iloc[:, :-1].values  
y = s_data.iloc[:, 1].values


# Now that we have our attributes and labels, the next step is to split this data into training and test sets. We'll do this by using Scikit-Learn's built-in train_test_split() method:

# In[17]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0)


# #### Training the Algorithm
# We have split our data into training and testing sets, and now is finally the time to train our algorithm.

# In[18]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training complete.")


# In[19]:


# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X,y)
plt.plot(X, line);
plt.show()


# ### Making Predictions
# Now that we have trained our algorithm, it's time to make some predictions.

# In[20]:


print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores


# In[21]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df


# In[22]:


# You can also test with your own data
hours = 9.25
own_pred = regressor.predict([[hours]])
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# In[23]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred))


# In[ ]:




