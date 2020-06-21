### Project Overview

 Till now you have seen that how to solve the linear regression and regularization problem. Now in this project, you are going to predict the Insurance claim using logistic regression. This dataset contains information on the insurance claim. each observation is different policyholder with various features like the age of the person, the gender of the policyholder, body mass index, providing an understanding of the body, number of children of the policyholder, smoking state of the policyholder and individual medical costs billed by health insurance.


### Learnings from the project

 After completing this project, you will have a better understanding of how to build a logistic regression model. In this project, you will apply the following concepts.
•	Train-test split
•	Correlation between the features 
•	Logistic Regression
•	Auc score
•	Roc AUC plot



### Approach taken to solve the problem

 Data loading and splitting

The first step - you know the drill by now - load the dataset and see how it looks like. Additionally, split it into train and test set. 
Instructions:
•	Load dataset using pandas read_csv api in variable df and give file path as path.
•	Display first 5 columns of dataframe df.
•	Store all the features(independent values) in a variable called X
•	Store the target variable (dependent value) in a variable called y
•	Split the dataframe into X_train,X_test,y_train,y_test using train_test_split() function. Use test_size = 0.2 and random_state = 6


Outlier Detection

Let's plot the box plot to check for the outlier. 
Instructions:
•	Plot the boxplot for X_train['bmi'].
•	Set quantile equal to 0.95for X_train['bmi']. and store it in variable q_value.
•	Check the value counts of the y_train


Correlation Check !

Let's check the pair_plot for feature vs feature. This tells us which features are highly correlated with the other feature and help us predict its better logistic regression model. 
Instructions:
•	Find the correlation between the features which are stored in 'X_train' and store the result in a variable called 'relation'.
•	print relation.
•	plot pairplot for X_train.


Predictor check!

Let's check the count_plot for different features vs target variable insuranceclaim. This tells us which features are highly correlated with the target variable insuranceclaim and help us predict it better. 
Instructions :
•	Create a list cols store the columns 'children','sex','region','smoker' in it.
•	Create subplot with (nrows = 2 , ncols = 2) and store it in variable's fig ,axes
•	Create for loop to iterate through row.
•	Create another for loop inside for to access column.
•	create variable col and pass cols[ i * 2 + j].
•	Using seaborn plot the countplot where x=X_train[col], hue=y_train, ax=axes[i,j]


Is my Insurance claim prediction right?

Now let's come to the actual task, using logistic regression to predict the insuranceclaim. We will select the best model by cross-validation using Grid Search. 
Instructions:
•	You are given a list of values for regularization parameters for the logistic regression model.
•	Instantiate a logistic regression model with LogisticRegression() and pass the parameter as random_state=9 and save it to a variable called 'lr'.
•	Inside GridSearchCV() pass estimator as the logistic model, param_grid=parameters. to do grid search on the logistic regression model store the result in variable grid. 
•	Fit the model on the training data X_train and y_train.
•	Make predictions on the X_test features and save the results in a variable called 'y_pred'.
•	Calculate accuracy for grid and store the result in the variable accuracy
•	print accuracy.


Performance of a classifier !

Now let's visualize the performance of a binary classifier. Check the performance of the classifier using roc auc curve. 
Instructions:
•	Calculate the roc_auc_score and store the result in variable score.
•	Predict the probability using grid.predict_proba on X_test and take the second column and store the result in y_pred_proba.
•	Use metrics.roc_curve to calculate the fpr and tpr and store the result in variables fpr, tpr, _.
•	Calculate the roc_auc score of y_test and y_pred_proba and store it in variable called roc_auc. 
•	Plot auc curve of 'roc_auc' using the line plt.plot(fpr,tpr,label="Logistic model, auc="+str(roc_auc)).



