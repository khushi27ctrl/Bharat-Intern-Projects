#!/usr/bin/env python
# coding: utf-8

# In[49]:


import pandas as pd
import numpy as np


# In[50]:


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
IDtest = test["PassengerId"]


# In[51]:


test.head()


# In[52]:


# Fill empty and NaNs values with NaN
dataset = train.fillna(np.nan)

# Check for Null values
dataset.isnull().sum()


# In[53]:


train.info()


# In[54]:


### Summarize data

train.describe()


# #  Data analysis

# In[55]:


import seaborn as sns


# In[56]:


g = sns.heatmap(train[["Survived","SibSp","Parch","Age","Fare"]].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")


# In[57]:


# Explore SibSp feature vs Survived
g = sns.catplot(x="SibSp", y="Survived", data=train, kind="bar", height=5, palette="muted")
g.despine(left=True)
g.set_ylabels("Survival Probability")


# In[58]:


# Explore Age vs Survived
g = sns.FacetGrid(train, col='Survived')
g = g.map(sns.histplot, "Age")


# In[59]:


# Explore Age distibution 
g = sns.kdeplot(train["Age"][(train["Survived"] == 0) & (train["Age"].notnull())], color="Red", fill = True)
g = sns.kdeplot(train["Age"][(train["Survived"] == 1) & (train["Age"].notnull())], ax =g, color="Green", fill= True)
g.set_xlabel("Age")
g.set_ylabel("Frequency")
g = g.legend(["Not Survived","Survived"])


# In[60]:


# Explore Fare distribution 
g = sns.histplot(dataset["Fare"], label="Skewness : %.2f"%(dataset["Fare"].skew()))
g = g.legend(loc="best")


# In[61]:


# Apply log to Fare to reduce skewness distribution
dataset["Fare"] = dataset["Fare"].map(lambda i: np.log(i) if i > 0 else 0)

g = sns.histplot(dataset["Fare"], color="lightblue", label="Skewness : %.2f"%(dataset["Fare"].skew()))
g = g.legend(loc="best")


# # Gender survival  probablitty

# In[62]:


g = sns.barplot(x="Sex",y="Survived",data=train)
g = g.set_ylabel("Survival Probability")


# In[63]:


import matplotlib.pyplot as plt


# In[64]:


# Create a range of x values (predicted probabilities)
x = np.linspace(0.01, 0.99, 100)

# Calculate the cost for each scenario
cost_scenario_1 = -np.log(1 - x)  # When y = 0
cost_scenario_2 = -np.log(x)      # When y = 1

# Plot the cost function for both scenarios
plt.figure(figsize=(10, 5))
plt.plot(x, cost_scenario_1, label='Cost when y = 0', linestyle='-', color='blue')
plt.plot(x, cost_scenario_2, label='Cost when y = 1', linestyle='--', color='red')
plt.xlabel('Predicted Probability (y_pred)')
plt.ylabel('Cost (Error)')
plt.legend()
plt.title('Cost Function for Binary Classification')
plt.grid(True)
plt.show()


# In[65]:


from sklearn.linear_model import LogisticRegression
from mpl_toolkits.mplot3d import Axes3D

# Load the train data (replace file path as needed)
train = pd.read_csv("train.csv")

# Handle missing values in the "Age" column by filling with the median age
train['Age'].fillna(train['Age'].median(), inplace=True)

# Apply log transformation to "Fare" to reduce skewness
train["Fare"] = train["Fare"].map(lambda i: np.log(i) if i > 0 else 0)

# Extract features and target variable
X_age_fare = train[['Age', 'Fare']]
y = train['Survived'].values

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
ax.scatter(X_age_fare['Age'], X_age_fare['Fare'], y, c=y, cmap='coolwarm', marker='o')
ax.set_xlabel('Age')
ax.set_ylabel('Log(Fare)')
ax.set_zlabel('Survived')
ax.set_title('3D Scatter Plot of Age, Log(Fare), and Survived')
# Fit a logistic regression model
model = LogisticRegression()
model.fit(X_age_fare, y)

# Create a meshgrid for the logistic regression plane
age_range = np.linspace(X_age_fare['Age'].min(), X_age_fare['Age'].max(), 30)
fare_range = np.linspace(X_age_fare['Fare'].min(), X_age_fare['Fare'].max(), 30)
age_mesh, fare_mesh = np.meshgrid(age_range, fare_range)
X_test_3d = np.column_stack((age_mesh.ravel(), fare_mesh.ravel()))
y_pred = model.predict_proba(X_test_3d)[:, 1].reshape(age_mesh.shape)

# Plot the logistic regression plane
ax.plot_surface(age_mesh, fare_mesh, y_pred, cmap='viridis', alpha=0.5)
plt.show()


# In[66]:


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Load the Titanic dataset
train = pd.read_csv("train.csv")

# Handle missing values in the "Age" column by filling with the median age
train['Age'].fillna(train['Age'].median(), inplace=True)

# Apply log transformation to "Fare" to reduce skewness
train["Fare"] = train["Fare"].map(lambda i: np.log(i) if i > 0 else 0)

# Encode 'Sex' column to numeric values (0 for female, 1 for male)
encoder = LabelEncoder()
train['Sex'] = encoder.fit_transform(train['Sex'])

# Extract features and target variable
X = train[['Age', 'Fare', 'Sex']]
y = train['Survived']
model = LogisticRegression()
model.fit(X, y)

# Get coefficients and intercept
coefficients = model.coef_
intercept = model.intercept_

# Display coefficients and intercept
print("Coefficients:", coefficients)
print("Intercept:", intercept)

# Interpret coefficients
feature_names = ['Age', 'Log(Fare)', 'Sex']
for i, feature in enumerate(feature_names):
    print(f"Coefficient for {feature}: {coefficients[0][i]}")

# Interpret the intercept
print("Intercept:", intercept[0])


# In[67]:


dataset =  pd.concat(objs=[train, test], axis=0).reset_index(drop=True)


# In[68]:


dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]
dataset["Title"] = pd.Series(dataset_title)
dataset["Title"].head()


# In[69]:


# Convert to categorical values Title 
dataset["Title"] = dataset["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
dataset["Title"] = dataset["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
dataset["Title"] = dataset["Title"].astype(int)


# In[70]:


# Drop Name variable
dataset.drop(labels = ["Name"], axis = 1, inplace = True)


# In[71]:


# Create a family size descriptor from SibSp and Parch
dataset["Fsize"] = dataset["SibSp"] + dataset["Parch"] + 1


# In[72]:


# Create new feature of family size
dataset['Single'] = dataset['Fsize'].map(lambda s: 1 if s == 1 else 0)
dataset['SmallF'] = dataset['Fsize'].map(lambda s: 1 if  s == 2  else 0)
dataset['MedF'] = dataset['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
dataset['LargeF'] = dataset['Fsize'].map(lambda s: 1 if s >= 5 else 0)


# In[73]:


dataset = pd.get_dummies(dataset, columns = ["Title"])
dataset = pd.get_dummies(dataset, columns = ["Embarked"], prefix="Em")


# In[74]:


# Replace the Cabin number by the type of cabin 'X' if not
dataset["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dataset['Cabin'] ])


# In[75]:


dataset = pd.get_dummies(dataset, columns = ["Cabin"],prefix="Cabin")


# In[76]:


Ticket = []
for i in list(dataset.Ticket):
    if not i.isdigit() :
        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0]) #Take prefix
    else:
        Ticket.append("X")
        
dataset["Ticket"] = Ticket
dataset["Ticket"].head()


# In[77]:


dataset = pd.get_dummies(dataset, columns = ["Ticket"], prefix="T")


# In[78]:


# Create categorical values for Pclass
dataset["Pclass"] = dataset["Pclass"].astype("category")
dataset = pd.get_dummies(dataset, columns = ["Pclass"],prefix="Pc")


# In[79]:


# Drop useless variables 
dataset.drop(labels = ["PassengerId"], axis = 1, inplace = True)


# In[80]:


dataset["Sex"] = dataset["Sex"].map({"male": 0, "female":1})


# In[81]:


dataset['Age'].fillna(dataset['Age'].median(), inplace=True)


# In[82]:


dataset['Fare'].fillna(dataset['Fare'].median(), inplace=True)


# # Modelling

# In[83]:


train_len = len(train)
train = dataset[:train_len]
test = dataset[train_len:]
test.drop(labels=["Survived"],axis = 1,inplace=True)


# In[84]:


## Separate train features and label 

train["Survived"] = train["Survived"].astype(int)

Y_train = train["Survived"]

X_train = train.drop(labels = ["Survived"],axis = 1)


# In[85]:


from sklearn.model_selection import StratifiedKFold

# Cross validate model with Kfold stratified cross val
kfold = StratifiedKFold(n_splits=10)


# In[101]:


from sklearn.model_selection import cross_val_score

# Set the random seed for reproducibility
random_state = 2

# Create a list of classifiers with logistic regression
classifiers = []
classifiers.append(LogisticRegression(random_state=random_state, max_iter=1000))  # Increase max_iter

# Define the number of cross-validation folds (k)
kfold = StratifiedKFold(n_splits=10)

# Perform cross-validation and store results
cv_results = []
for classifier in classifiers:
    cv_results.append(cross_val_score(classifier, X_train, y=Y_train, scoring="accuracy", cv=kfold, n_jobs=4))
#Calculate means and standard deviations of cross-validation results
cv_means = [cv_result.mean() for cv_result in cv_results]
cv_std = [cv_result.std() for cv_result in cv_results]

# Create a DataFrame to store the results
cv_res = pd.DataFrame({"CrossValMeans": cv_means, "CrossValErrors": cv_std, "Algorithm": ["LogisticRegression"]})

cv_res


# In[ ]:






# In[103]:


test.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




