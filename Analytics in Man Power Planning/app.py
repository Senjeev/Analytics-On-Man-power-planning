# Libraries
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
st.set_option('deprecation.showPyplotGlobalUse', False)
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics
# load dataset
data = pd.read_csv("train.csv")
#split dataset in features and target variable
feature_cols = ['etest_percentage','coding_score','degree_percentage',
           'Aptitude_Score', 'mba_percentage', 'Salary Expected']
X = data[feature_cols] # Features
y = data["status"] # Target variable
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) 
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()
# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
st.title("Analytics in Manpower Planning")
if st.sidebar.checkbox("Predict"):
    name=st.text_input("Name")
    desig=st.text_input("Designation")
    stre=st.text_input("Stream")
    special=st.text_input("Specialisation")
    inter=st.text_input("Interested In")
    ascore=st.number_input("Aptitude Score")
    cscore=st.number_input("Coding Score")
    dp=st.number_input("Degree Percentage")
    ep=st.number_input("ETest Percentage")
    mba=st.number_input("MBA Percentage")
    exsalary=st.number_input("Expected Salary")
    pre=[[ep,cscore,dp,ascore,mba,exsalary]]
    #Predict the response for test dataset
    y_pred = clf.predict(pre)
    if st.button("Analyze"):
        if y_pred == 1:
            st.success("Placed")
        else:
            st.error("Not Placed")
if st.sidebar.checkbox("Visualization"):
    st.write("Education Status")
        #Number of employess and their education status - Promotion
    plt.figure(figsize=(6,6))
    sns.barplot(x='degree_percentage',y='Aptitude_Score',hue='status',data=data)
    st.pyplot()
    st.write("Interested Stream")
    sns.countplot(x='interested_in',data=data,palette='Greens_d')
    st.pyplot()
    st.write("Decision Tree")
    plt.figure(figsize=(20, 10))
    tree.plot_tree(clf, filled=True, rounded=True,feature_names = feature_cols,class_names = ['0','1'])
    st.pyplot()
    st.write("Accuracy:")
    y_pred = clf.predict(X_test)
    st.write(metrics.accuracy_score(y_test, y_pred))
    st.write("Analyzing Designation Vs MBA Percentage")
    # Scatter plot with day against tip
    plt.scatter(data['Designation'], data['mba_percentage'])

    # Adding Title to the Plot
    plt.title("Scatter Plot")

    # Setting the X and Y labels
    plt.xlabel('Designation')
    plt.ylabel('mba_percentage')
    st.pyplot()
    st.write("Designation Vs Score")
    # The slice names of a population distribution pie chart
    pieLabels = 'Science', 'Commerce', 'Arts'
    # Population data
    populationShare     = [72, 20, 8]
    figureObject, axesObject = plt.subplots()
    # Draw the pie chart
    axesObject.pie(populationShare, labels=pieLabels, autopct='%1.2f', startangle=90)
    # Aspect ratio - equal means pie is a circle
    axesObject.axis('equal')
    st.pyplot()
    st.write("KNN")
    from sklearn.neighbors import KNeighborsClassifier
    knn3 = KNeighborsClassifier(n_neighbors=2)
    model3=knn3.fit(X_train, y_train)
    y_pred=model3.predict(X_test)
    st.write("Accuracy")
    st.write(model3.score(X_test, y_test))
    st.write("Naive Bayes")
    clf = RandomForestClassifier(n_estimators = 100) 
 
    # Training the model on the training dataset
    # fit function is used to train the model using the training sets as parameters
    clf.fit(X_train, y_train)

    # performing predictions on the test dataset
    y_pred = clf.predict(X_test)

    # using metrics module for accuracy calculation
    st.write("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))