# Supervised Machine Learning Homework - Predicting Credit Risk

## Background and Objectuve

LendingClub is a peer-to-peer lending services company that allows individual investors to partially fund personal loans as well as buy and sell notes backing the loans on a secondary market. LendingClub offers their previous data through an API.

To build a machine learning model that attempts to predict whether a loan from LendingClub will become high risk or not. Specifically, the model should compare the Logistic Regression model and Random Forest Classifier.

## Key Artifact

[Machine Learning Model using Python](Credit_Risk_Evaluator.ipynb)

## Model Development Process

### Retrieve the data

Using the [GenerateData.ipynb](/Resources/Generator/GenerateData.ipynb) notebook in the In the `Generator` folder in `Resources`, downloaded data from LendingClub and output two CSVs: 

* `2019loans.csv`
* `2020Q1loans.csv`

Using an entire year's worth of data (2019) the goal was to predict the credit risk of loans from the first quarter of the next year (2020).

Note: these two CSVs have been undersampled to give an even number of high risk and low risk loans. In the original dataset, only 2.2% of loans are categorized as high risk. To get a truly accurate model, special techniques need to be used on imbalanced data. Undersampling is one of those techniques. Oversampling and SMOTE (Synthetic Minority Over-sampling Technique) are other techniques that are also used.

## Preprocessing: Convert categorical data to numeric

Created a training set from the 2019 loans by reading it into a Dataframe and using `pd.get_dummies()` convert the categorical data to numeric columns. 

Also, created a testing set from the 2020 loans by reading it into a Dataframe and using `pd.get_dummies()`. 

Additionally there were some category columns that were in the 2019 loans that did not exist in the testing set. Using Python code, filled in the missing categories in the testing set. 

## Consider the models

Created and compared two models: LogisticRegression, and a RandomForestClassifier. Before, creating, fitting, and scoring the models, my prediction was that the RandomForestClassifierr would do better than the LogisticRegression because it is bit more feature rich and generally found to be more accurate.

## Fit a LogisticRegression model and RandomForestClassifier model

Created a LogisticRegression model, fit it to the data, and printed the model's score using Unscaled Data.

### Logisitc Regression - Unscaled Data
------------------------------------
* Training Data Score: 0.65311986863711
* Testing Data Score: 0.5072309655465759

Did the same for a RandomForestClassifier. For this I used hyperparameters such as n_estimators and random_state. Also fit the data and printed the results

### Random Forest Classifier - Unscaled Data
------------------------------------
* Training Data Score: 1.0
* Testing Data Score: 0.646958740961293

RandomForestClassifier appeared to have slightly better accuracy.

## Revisit the Preprocessing: Scale the data

Using `StandardScaler` to scale the training and testing sets. Based on past experience, I was expecting better scores using the scaled data on both the LogisticRegression and RandomForestClassifier.

Ran both the models on the scaled data and slightly better results. In fact scaled data resulted in the LogisiticRegression model having a better Testing Score than the same for RandomForestClassifier.

### Logisitc Regression - Scaled Data
------------------------------------
* Training Data Score: 0.7108374384236453
* Testing Data Score: 0.7598894087622289

----------- AND ---------------------

### Random Forest Classifier - Scaled Data
------------------------------------
* Training Data Score: 1.0
* Testing Data Score: 0.6480221182475542

## Conclusion
* Both Models Performed better using Scaled Data as opposed to Unscaled Data.
* LogisticRegression using Scaled Data had a higher score than RandomForestClassifier also using Scaled Data.

### References

LendingClub (2019-2020) _Loan Stats_. Retrieved from: [https://resources.lendingclub.com/](https://resources.lendingclub.com/)

- - -

