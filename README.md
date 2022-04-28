Machine-Learning-Classifier
![](Resources/MachineLearning-Banner.jpg)

This project aids to predict credit risk with machine learning techniques.

In my analysis I evalutated data applying several machine learning models to predict credit risk using data you'd typically see from peer-to-peer lending services. Credit risk is an inherently imbalanced classification problem (the number of good loans is much larger than the number of at-risk loans), different techniques for training and evaluating models with imbalanced classes was necassary. I applied imbalanced-learn and Scikit-learn libraries to build and evaluate models using the two following techniques:

Resampling
Ensemble Learning

Files
Resampling Starter Notebook
Ensemble Starter Notebook
Lending Club Loans Data


Resampling
Used the imbalanced learn library to resample the LendingClub data and built and evaluate logistic regression classifiers using the resampled data.

Split the data into Training and Testing sets.

Scaled the training and testing data using the StandardScaler from sklearn.preprocessing.


Used the provided code to ran a Simple Logistic Regression:

Fit the logistic regression classifier.
Calculated the balanced accuracy score.
Displayed the confusion matrix.
Printed the imbalanced classification report.

Oversampled the data using the Naive Random Oversampler and SMOTE algorithms.

Undersampled the data using the Cluster Centroids algorithm.

Over- and undersampled using a combination SMOTEENN algorithm.


Trained a logistic regression classifier from sklearn.linear_model using the resampled data.


Calculated the balanced accuracy score from sklearn.metrics.


Displayed the confusion matrix from sklearn.metrics.


Printed the imbalanced classification report from imblearn.metrics.


Used the above to answer the following questions:

Which model had the best balanced accuracy score?




Which model had the best recall score?




Which model had the best geometric mean score?


Ensemble Learning
In this section, I trained and compared two different ensemble classifiers to predict loan risk and evaluate each model. I used the Balanced Random Forest Classifier and the Easy Ensemble Classifier. 

Split the data into training and testing sets.


Scaled the training and testing data using the StandardScaler from sklearn.preprocessing.



Trainned the model using the quarterly data from LendingClub provided in the Resource folder.


Calculated the balanced accuracy score from sklearn.metrics.


Displayed the confusion matrix from sklearn.metrics.


Generated a classification report using the imbalanced_classification_report from imbalanced learn.


For the balanced random forest classifier only, I printed the feature importance sorted in descending order (most important feature to least important) along with the feature score.


Used the above to answer the following questions:


Which model had the best balanced accuracy score?


Which model had the best recall score?


Which model had the best geometric mean score?


What are the top three features?

![](Resources/Data Science Courses.jpg)
