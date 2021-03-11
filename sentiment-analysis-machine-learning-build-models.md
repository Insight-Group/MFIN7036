# Machine learning approaches: Build NB, LR, SVM models

Machine learning is another feasible way for sentiment analysis. We build three machine learning models to predict the sentiment of tweets. We start from the twitter dataset of Fusun Pharma. After calculating the accuracy of these three models, we will use the most efficient model to predict the sentiment of tweets in other dataset. 

## 1. Generate Training Dataset

```python
def get_feature_vector(train_fit):
    vector = TfidfVectorizer(sublinear_tf=True)
    vector.fit(train_fit)
    return vector

def ML_train_dataset(df_text, df_target):
    # df_text and df_target are columns of a dataframe

    # tf_vector will be used for Testing sentiments on unseen trending data
    tf_vector = get_feature_vector(np.array(df_text.dropna()).ravel())
    X = tf_vector.transform(np.array(df_text.dropna()).ravel())
    #target
    y = np.array(df_target.dropna()).ravel()
    # Split dataset into Train, Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    return tf_vector, X_train, X_test, y_train, y_test
```

After Fitting the vectorizer to the training data and save the vectorizer to a variable, the variable output from `fit()` is transformed to validation data by `transform()`. The validation data and the target data will be splited into two parts. 

After training the train part by the following three machine learning models, the accuracy of each model will be calculated based on the prediction results of the test part. The `test_size` and `random_state` are set to be 0.33 and 42 respectively, and later will be tested with other input values.

The variable `tf_vector` will also further be used to predict unseen trending data from other dataset.


## 2. Machine Learning Models

### 2.1 Naive Bayes Model

Naïve Bayes is a generative model and assumes all the features to be conditionally independent. So, if some of the features are in fact dependent on each other, the prediction might be poor.(Ottesen, 2017)

```python
def Naive_Bayes_model(df_test_text, df_training_text, df_training_target):
        
    tf_vector, X_train, X_test, y_train, y_test = ML_train_dataset(df_training_text, df_training_target)
    
    # Training Naive Bayes model
    NB_model = MultinomialNB()
    NB_model.fit(X_train, y_train)
    
    # Accuracy of the trained data
    y_predict_nb = NB_model.predict(X_test)
    print('Training accuracy of Naive Bayes model is: ', accuracy_score(y_test, y_predict_nb))
        
    # Prediction on Real-time Feeds
    test_feature = tf_vector.transform(np.array(df_test_text).ravel())
    prediction_nb = NB_model.predict(test_feature)
    
    return prediction_nb

```

## 2.2 Logistics Regression Model

Logistic regression is a discriminative model which splits feature space linearly, and works reasonably well when some of the variables are correlated. Logistic regression can have different decision boundaries with different weights that are near the optimal point.(Ottesen, 2017)

```python
def Logistics_Regression_model(df_test_text, df_training_text, df_training_target):
    
    tf_vector, X_train, X_test, y_train, y_test = ML_train_dataset(df_training_text, df_training_target)
    
    # Training Logistics Regression model
    LR_model = LogisticRegression(solver='lbfgs')
    LR_model.fit(X_train, y_train)
    
    # Accuracy of the trained data
    y_predict_lr = LR_model.predict(X_test)
    print('Training accuracy of Logistics Regression model is: ', accuracy_score(y_test, y_predict_lr))
    
    # Prediction  
    test_feature = tf_vector.transform(np.array(df_test_text).ravel())
    prediction_lr = LR_model.predict(test_feature)
    
    return prediction_lr

```

## 2.3 SVM Model

SVM tries to finds the “best” margin (distance between the line and the support vectors) that separates the classes and this reduces the risk of error on the data.(Bassey, 2019)

```python
def Support_Vector_Machines(df_test_text, df_training_text, df_training_target):
    
    tf_vector, X_train, X_test, y_train, y_test = ML_train_dataset(df_training_text, df_training_target)

    # Training SVM model
    SVC_model = SVC()
    SVC_model.fit(X_train, y_train)
    
    # Accuracy of the trained data
    y_predict_svc = SVC_model.predict(X_test)
    print('Training accuracy of Support Vector Machines is:', accuracy_score(y_test, y_predict_svc))
    
    # Prediction 
    test_feature = tf_vector.transform(np.array(df_test_text).ravel())
    prediction_svc = SVC_model.predict(test_feature)
    
    return prediction_svc
```

# 3. Accuracy of Three Models







# References:

1. Ottesen, C. 2017, *Comparison between Naïve Bayes and Logistic Regression*, viewed 11 March 2011, <https://dataespresso.com/en/2017/10/24/comparison-between-naive-bayes-and-logistic-regression/>.

2. Bassey, P. 2019, *Logistic Regression Vs Support Vector Machines (SVM)*,  viewed 11 March 2011, <https://medium.com/axum-labs/logistic-regression-vs-support-vector-machines-svm-c335610a3d16>.