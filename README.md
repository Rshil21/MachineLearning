Titanic Survival Prediction
📋 Project Overview
This project implements a machine learning solution to predict passenger survival on the Titanic. Using various classification algorithms and comprehensive data analysis, the system identifies key factors influencing survival and builds predictive models with detailed evaluation metrics.

Domain: Machine Learning / Binary Classification
Goal: Predict whether a passenger survived the Titanic disaster based on demographic and ticket information.

📊 Dataset Information
Source: Kaggle Titanic Dataset
Size: 891 passengers × 12 features
Target Variable: Survived (0 = Died, 1 = Survived)

Features:
PassengerId: Unique identifier for each passenger

Survived: Survival status (0 = No, 1 = Yes) - Target Variable

Pclass: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)

Name: Passenger name (includes title)

Sex: Gender

Age: Age in years

SibSp: Number of siblings/spouses aboard

Parch: Number of parents/children aboard

Ticket: Ticket number

Fare: Passenger fare

Cabin: Cabin number

Embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

🛠️ Technology Stack
Core Libraries
Pandas - Data manipulation and analysis

NumPy - Numerical computing

Scikit-learn - Machine learning algorithms and evaluation

Matplotlib & Seaborn - Data visualization

Machine Learning Models Implemented
Random Forest Classifier - Ensemble of decision trees

Gradient Boosting Classifier - Sequential boosting algorithm

Logistic Regression - Statistical classification model

Support Vector Machine (SVC) - Kernel-based classification

Key Techniques
Exploratory Data Analysis (EDA)

Missing value imputation

Feature engineering

Cross-validation

Hyperparameter tuning

Model evaluation with multiple metrics
