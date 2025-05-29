import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

# Data Exploration
# Loading the dataset into a dataframe
bc_data = datasets.load_breast_cancer(as_frame=True)
bc_data.data.describe()

# Data inspection
# Looking for null and duplicate values
print("Null values:", bc_data.data.isnull().sum().sum())
print("Duplicate values:", bc_data.data.duplicated().sum())

# Replaces outliers in numerical columns with median values using IQR method
pd.options.mode.copy_on_write = True

def replace_outliers_with_median(df):
    for column in df.select_dtypes(include=['number']).columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        median = df[column].median()
        df[column] = df[column].apply(lambda x: median if x < lower_bound or x > upper_bound else x)
    return df

bc_data_cleaned = replace_outliers_with_median(bc_data.data.copy())
bc_data_cleaned.describe()

# The dataset is clean, now we're creating histograms for each column in our dataframe
bc_data_cleaned.hist(figsize=(20,20),edgecolor='black', rwidth=0.8, color='green');

# Modeling Data
# Separating the dataset into X and y.

X = bc_data_cleaned
y = bc_data.target

# Printing the shapes to verify there's data for all of the features 
print("X Shape:", X.shape)
print("Y Shape: ", y.shape)

# Creating test and training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Creating the list names to provide all of the classifier names to be usedwe are going to use and compare
names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process", "Decision Tree", "Random Forest", "Neural Net", "AdaBoost", "Naive Bayes", "QDA"]

# Creating the list classifiers to supply all of the classifiers and their parameters
classifiers = [KNeighborsClassifier(n_neighbors=3),
              SVC(kernel="linear", C=0.025, random_state=42),
              SVC(gamma=2, C=1, random_state=42),
              GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
              DecisionTreeClassifier(max_depth=5, random_state=42),
              RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, random_state=42),
              MLPClassifier(alpha=1, max_iter=1000, random_state=42),
              AdaBoostClassifier(random_state=42),
              GaussianNB(),
              QuadraticDiscriminantAnalysis(reg_param = 0.1)]

# Initialize the figure
figure = plt.figure(figsize=(27, 9))

x_min, x_max = X.iloc[:, 0].min() - 0.5, X.iloc[:, 0].max() + 0.5
y_min, y_max = X.iloc[:, 1].min() - 0.5, X.iloc[:, 1].max() + 0.5

# Create loop that scales the data, trains the models, calculates the evaluation metrics, and plots the results
acc_scores = []
cvs = []

i=1
for name, model in zip(names, classifiers):
  ax = plt.subplot(1, len(classifiers) + 1, i)

  pipe = make_pipeline(StandardScaler(), model)
  pipe.fit(X_train, y_train)
  y_pred = pipe.predict(X_test)
  
  # Metrics
  acc = accuracy_score(y_test, y_pred)
  acc_scores.append(acc)

  cv = cross_val_score(pipe, X, y, cv=5)
  cvs.append(cv)

  print(classifiers[i-1], "\n", confusion_matrix(y_test, y_pred))

  #Plot the training points
  ax.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train, cmap=plt.colormaps['viridis'], edgecolors="k")
  # Plot the testing points
  ax.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_test, cmap=plt.colormaps['viridis'], edgecolors="k", alpha=0.5)
  # Setting up the parameters for the graphs
  ax.set_xlim(x_min, x_max)
  ax.set_ylim(y_min, y_max)
  ax.set_xticks(())
  ax.set_yticks(())
  ax.set_title(name)
  # Adding the score to the figures
  ax.text(x_max - 0.3, y_min + 0.3, ("Score: %.2f" % acc_scores[i-1]).lstrip("0"), size=15, verticalalignment='center', horizontalalignment="right")
  i+=1

plt.show()

"""It appears that most of the classifiers did well with the exceptions of the RBF Support Vector Machine and Gaussian Process classifiers.
The confusion matrix tells us how the classifier predicted the data points and the results for the RBG SVM and the Gaussian Process classifiers show that the low 
scores were due to predicting the opposite of the targets for the data hence the ~50% scores."""

# Visualizing how the classifiers performed by plotting the accuracy and cross validation scores
ax = plt.subplot()
bar_width = 0.2

x = np.arange(len(names))
mean_cvs = [np.mean(scores) for scores in cvs]

ax.barh(x + bar_width, acc_scores, height = 0.4, label = 'Accuracy', color = 'blue')
ax.barh(x + bar_width*3, mean_cvs, height = 0.4, label = 'Cross Validation', color = 'green')
ax.set_yticks(x, names)
ax.set_xlabel("Score")
ax.set_title("Classifier Comparison")
ax.grid(True, axis='x', linestyle='--', alpha=0.7)
ax.legend()

plt.tight_layout()
plt.show()

"""This next bit was just for fun to practice a little bit of formatting. The goal was to print the label prediction assigned to a specimen by the linear SVC 
classifier."""

# Using the last 10 samples, run the linear SVC classifier and print their predictions.
input = X[559:].values
SVC_model = SVC(kernel="linear", C=0.025, random_state=42)
SVC_model.fit(X_train, y_train)
SVC_predict = SVC_model.predict(input)
print(SVC_predict)

# Print the predictions as either benign or malignant
for i in range (len(SVC_predict)):
  if SVC_predict[i] == 0:
    print(f'Index {i}: ', "Benign")
  else:
    print(f'Index {i}: ', "Malignant")
  i += 1