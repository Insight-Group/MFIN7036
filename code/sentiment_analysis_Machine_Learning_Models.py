# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 14:27:24 2021

@author: Sophie
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def get_feature_vector(train_fit):
    vector = TfidfVectorizer(sublinear_tf=True)    # TfidfVectorizer (stop_words= 'english', max_df=0.9)
    vector.fit(train_fit)
    return vector


def ML_train_dataset(df_text, df_target):
    # Same tf vector will be used for Testing sentiments on unseen trending data
    tf_vector = get_feature_vector(np.array(df_text.dropna()).ravel())
    X = tf_vector.transform(np.array(df_text.dropna()).ravel())
    
    #target
    y = np.array(df_target.dropna()).ravel()
    
    # Split dataset into Train, Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return tf_vector, X_train, X_test, y_train, y_test
    

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
    