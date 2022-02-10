# Supervised Machine Learning Homework - Predicting Credit Risk

## Background

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

You will be creating and comparing two models on this data: a logistic regression, and a random forests classifier. Before you create, fit, and score the models, make a prediction as to which model you think will perform better. You do not need to be correct! Write down (in markdown cells in your Jupyter Notebook or in a separate document) your prediction, and provide justification for your educated guess.

## Fit a LogisticRegression model and RandomForestClassifier model

Create a LogisticRegression model, fit it to the data, and print the model's score. Do the same for a RandomForestClassifier. You may choose any starting hyperparameters you like. Which model performed better? How does that compare to your prediction? Write down your results and thoughts.

## Revisit the Preprocessing: Scale the data

The data going into these models was never scaled, an important step in preprocessing. Use `StandardScaler` to scale the training and testing sets. Before re-fitting the LogisticRegression and RandomForestClassifier models on the scaled data, make another prediction about how you think scaling will affect the accuracy of the models. Write your predictions down and provide justification.

Fit and score the LogisticRegression and RandomForestClassifier models on the scaled data. How do the model scores compare to each other, and to the previous results on unscaled data? How does this compare to your prediction? Write down your results and thoughts.

## Conclusion


### References

LendingClub (2019-2020) _Loan Stats_. Retrieved from: [https://resources.lendingclub.com/](https://resources.lendingclub.com/)

- - -

