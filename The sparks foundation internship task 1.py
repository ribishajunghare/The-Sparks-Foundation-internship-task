#!/usr/bin/env python
# coding: utf-8

# ##### Author : Ribisha Junghare

# # Linear Regression with Python Scikit Learn
# 
# In this section we will see how the Python Scikit-Learn library for machine learning can be used to implement regression functions. We will start with simple linear regression involving two variables.
# 
# ## Simple Linear Regression
# In this regression task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied. This is a simple linear regression task as it involves just two variables.

# In[2]:


# Import libraries

import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Read Data

# In[3]:


data = pd.read_csv(r"C:\Users\Ribisha\Downloads\student_scores - student_scores.csv")


# In[4]:


## Print the first 10 rows of data

data.head(10)


# Plotting 2-D graph to see if there exist any relationship between the data. 

# In[5]:


# Plotting the distribution of scores

data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()

We can see that there is a +ve linear relationship between the number of hours studied by the student and their percentage score.
# ## Preparing the data
# The next step is to divide the data into "attributes" (inputs) and "labels" (outputs).

# In[6]:


X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values 


# Now that we have our attributes and labels, the next step is to split this data into training and test sets. We'll do this by using Scikit-Learn's built-in train_test_split() method:

# In[7]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 


# ## Training the Algorithm
# We have split our data into training and testing sets, and now is finally the time to train our algorithm. 

# In[8]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training complete.")


# In[9]:


# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# ## Making Predictions
# Now that we have trained our algorithm, it's time to make some predictions.

# In[10]:


print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores
print(y_pred)


# In[15]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# ## Evaluating the model
# The final step is to evaluate the performance of algorithm. This step is particularly important to compare how well different algorithms perform on a particular dataset. For simplicity here, we have chosen the mean square error. There are many such metrics.

# In[18]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 


# # Thank you!
