# Practical Assignment 17.1 :  Whether a Client with subscribe for a term deposit 

##  1.	Business Understanding
        The data is from a Portuguese banking institution and is a collection of the results of multiple marketing campaigns. 
        Solve binary classification problem, using predictive modeling, which would identify whether a client 
        will subscribe to a term deposit (y) based on various features.
##     	Business Goals and KPI’s
        The business goal is to recommend best performing model that would determine the category of clients who are likely to do term deposit
##  2. Data Understanding 
        Dataset :  Bank-additional-full.csv
##    Observation
      **Numerical Columns :age, campaign, pdays, previous, emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed]
      **Categorical Columns: job, marital, education, default, housing, loan, contact, month, day_of_week, poutcome, y
        No duplicates found
        No Null values found
      There are some NaN values
## 3. Data Preparation
    	 Dropped unnecessary columns, rows and columns with all NaN values
    	 Dropped the columns that has high cardinality 
       'education', 'month', 'day_of_week'
## 4. Data Exploration
      Performed Descriptive statistical analysis :
       Average age of the clients is 40 years
       There is minimum 1 contact*
       There are 56 number of max contacts
       75% of the data have an employment rate of 1.40 or lower
       999 maximum days of wait time
       25% of the data have an index of -42.70 or lower
       75% of the data have a number of employees of 5,228.10 or lower
                    age	          campaign	     pdays	         previous	     emp.var.rate	cons.price.idx	cons.conf.idx	 nr.employed
          count	  41188.00000	   41188.000000	  41188.000000	  41188.000000	 41188.000000	41188.000000  	41188.000000	41188.000000
          mean	  40.02406	     2.567593	      962.475454	    0.172963	     0.081886	    93.575664	      -40.502600	  5167.035911
          std	    10.42125	     2.770014	      186.910907	    0.494901	     1.570960	    0.578840	      4.628198	    72.251528
          min	    17.00000	     1.000000	      0.000000	      0.000000	     -3.400000    92.201000	      -50.800000	  4963.600000
          25%	    32.00000	     1.000000	      999.000000	    0.000000	     -1.800000    93.075000	      -42.700000	  5099.100000
          50%	    38.00000	     2.000000	      999.000000	    0.000000	     1.100000	    93.749000	      -41.800000	  5191.000000
          75%	    47.00000	     3.000000	      999.000000	    0.000000	     1.400000	    93.994000	      -36.400000	  5228.100000
          max	    98.00000	     56.000000	    999.000000	    7.000000	     1.400000	    94.767000	      -26.900000	  5228.100000 
##Graphical interpretation
![images/categorical_distributions](images/categorical_distributions.png)
** The one who have jobs 
** Married
** Who were contacted by cellular
** who did not take loans in the personal_loan_subscription_distribution
##By Age
Age_Distribution_by_Term_Deposit_Subscription.png
![Age_Distribution_by_Term_Deposit_Subscription](Age_Distribution_by_Term_Deposit_Subscription.png)
** Max count is shown for the once who have age around 30 yrs 
** The age distribution indicates there are very few older once
## Feature Engineering 
**Separate Features and target
**Define Categorical and Numerical Columns
**Preprocess Numerical Columns using SimpleImputer and StandardScalar
**Create PCA Pipeline, Extract PCA Components,  reduce the feature space to 2 principal components.
**Visualizations
images/PCA_Analysis.png
![images/PCA_Analysis](images/PCA_Analysis.png)
**Scatter plots shows that the dataset 
     **Includes more Clients that are likely to make Term deposit
     ** There are very few outliers
Explained_Var_Ratio_PCA.png
![Explained_Var_Ratio_PCA](Explained_Var_Ratio_PCA.png)
**PC1 is responsible for capturing a significant amount of the variability in the data.
**PC2 still contributes additional information that PC1 does not cover
**Total Explained Variance: 40.8% ~  two components collectively capture a moderate portion of the data's overall variability
**Created correlation matrix to show relationship between various numeric variables
Correlation_Matrix_Num_Var.png
![Correlation_Matrix_Num_Var](Correlation_Matrix_Num_Var.png)
** Diagonal 1's shows there is perfect correlation between variables
** Previous Campaigns have no or very less impact on clients (values shown -1 in the graph)
## 5 Process and prepare data for model training and evaluation
   **Scaling and Encoding 
	 Convert Categorical to numeric using Label encoding
	 Fit and Transform values (replace categorical labels with integers)
	 Encoded Column to the DataFrame allows for seamless integration into numerical analysis and machine learning models
	 Separate features and target
   Define a preprocessing pipeline for numerical features (Imputer: SimpleImputer,Scaler: StandardScaler)
   Scale and Impute Numerical columns 
	 Use one-hot encoded for categorical columns 
	 Use ColumnTransformer to transform 
	 **Idenfity feature importance
	 Based on the values received, below is the feature importance from this dataset
	 **High Importance: y_encoded is the most important feature, but this might be due to its direct representation of the target variable, making its importance somewhat expected.
   **Moderate Importance: Features like nr.employed, pdays, poutcome, and economic indicators have moderate importance, reflecting their role in influencing predictions.
   **Low Importance: Features such as job, previous, and various categorical indicators (marital, housing, loan, default) have very low importance, indicating they contribute minimally to the model’s predictions.
## 5	Modeling
Applied Cross-Validation logic to evaluate the model and
the final numbers showed same as predicted.

**Before hypertuning and using best parameters
Model								 Train Time (sec)	Train Accuracy	Test Accuracy	Cross Validation Mean Accuracy	Cross Validation Std Accuracy
Logistic Regression				0.6039				1.0000				1.0000						1.0000												0.0000
Decision Tree Classifier	0.0153				1.0000				1.0000						1.0000												0.0000
k-Nearest Neighbors				0.0800				0.09331				0.9053						0.6052												0.2492
Support Vector Machine		20.8662				0.9977				0.9972						0.9833												0.0242

**After hyper tuning and using best parameters 
Model 									Model  Train Time (s)	Train Accuracy	Test Accuracy	CV Mean Accuracy	CV Std Accuracy	Best Params
Logistic Regression			3.780599							0.542857					0.5						0.51							0.02						{'C': 0.01, 'solver': 'saga'}
Decision Tree						0.341747							1									0.566667			0.47							0.102956				{'max_depth': None, 'min_samples_split': 2}
k-Nearest Neighbors			0.111976							0.657143					0.466667			0.45							0.063246				{'n_neighbors': 5, 'weights': 'uniform'}
Support Vector Machine	0.180856							0.542857					0.5						0.53							0.024495				{'C': 0.1, 'gamma': 'scale', 'kernel': 'rbf'}

## 5	Evaluation
For each model, uses the corresponding parameter grid to find the best hyperparameter.
Decision Tree Classifier would be the best choice if computational efficiency is a priority due to its 
very fast training time and perfect performance metrics. However, if interpretability or consistency
 is more critical, Logistic Regression is also an excellent choice.

Best Overall							: Decision Tree Classifier (due to its fast training time and perfect accuracy)
Best for Interpretability	: Logistic Regression
k-Nearest Neighbors 			: low performance  
Support Vector Machine    : high training time

Accuracy:

The Decision Tree model has the highest test accuracy (56.7%) among the models, but it has a lower cross-validation mean accuracy (47%) and a high standard deviation (10.3%). This indicates that while it fits the training data very well, it may not generalize well across different subsets of data.
The Support Vector Machine has the highest cross-validation mean accuracy (53%) with a relatively low standard deviation (2.5%), indicating that it might be more stable and generalizable compared to other models.

Train Time:
The k-Nearest Neighbors model has the fastest train time (0.11 seconds), which is beneficial for applications requiring quick training. However, it has the lowest test and cross-validation accuracies.
Stability:

The Support Vector Machine has the best stability among the models with the lowest standard deviation in accuracy (2.5%). This suggests it is less prone to performance fluctuations across different data folds.


## 6	Deployment
     
Maintain Accuracy during Deployment to Production
Implement monitoring to track the model’s performance over time, including accuracy, 
 precision, recall, and other relevant metrics. Set up alerts for performance degradation
Ensure the model can handle the expected load and traffic.
Maintain Version Control
Include fallback mechanism
Include Error handling
Create API's that can be deployed both on-premise and in call_duration_distribution

##  Recommendations 
Given the performance metrics:

For Best Overall Accuracy and Stability: The Support Vector Machine model is recommended. 
It provides a good balance of accuracy and stability with a reasonable training time. 
Even though its test accuracy is similar to Logistic Regression and SVM, 
its cross-validation accuracy and stability make it a strong choice for deployment.

## Conclusion
In order to have a model that balances accuracy and stability for predicting 
whether a customer should be given a loan, then the Support Vector Machine
is the most suitable choice. Make sure to have enough  computational resources and 
powerful deployment environment when making the final decision.
