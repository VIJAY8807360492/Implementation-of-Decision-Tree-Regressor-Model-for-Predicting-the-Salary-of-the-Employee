# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.

2. Upload the dataset and check for any null values using .isnull() function.

3. Import LabelEncoder and encode the dataset.

4. Import DecisionTreeRegressor from sklearn and apply the model on the dataset.

5. Predict the values of arrays.

6. Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.

7. Predict the values of array.

8. Apply to new unknown values. 

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: vijay k
RegisterNumber: 24901153
 import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("/content/Employee.csv")

# Display the first few rows
print(data.head())

# Dataset information
data.info()

# Check for missing values
print(data.isnull().sum())

# Value counts for the "left" column
print(data["left"].value_counts())

# Encode categorical variables (salary)
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])

# Display the updated data
print(data.head())

# Define features and target
x = data[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours",
          "time_spend_company", "Work_accident", "promotion_last_5years", "salary"]]
y = data["left"]

# Display features
print(x.head())

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

# Initialize and fit the Decision Tree classifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train, y_train)

# Make predictions
y_pred = dt.predict(x_test)

# Evaluate the model
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Predict for a new sample
new_sample = [[0.5, 0.8, 9, 260, 6, 0, 1, 2]]
prediction = dt.predict(new_sample)
print("Prediction for new sample:", prediction)

# Plot the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(dt, feature_names=x.columns, class_names=['Not Left', 'Left'], filled=True)
plt.show()

*/
```

## Output:
![ex9](https://github.com/user-attachments/assets/4e8afe19-ee33-4ffc-a1ae-ad18d64534f3)




## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
