# Titanic Survival Prediction

## Project Overview

This project implements a machine learning solution to predict passenger survival on the Titanic. Using various classification algorithms and comprehensive data analysis, the system identifies key factors influencing survival and builds predictive models with detailed evaluation metrics.

**Domain:** Machine Learning / Binary Classification  
**Goal:** Predict whether a passenger survived the Titanic disaster based on demographic and ticket information.


##  Dataset Information

**Source:** [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic)  
**Size:** 891 passengers × 12 features  
**Target Variable:** `Survived` (0 = Died, 1 = Survived)

### Features:
- **PassengerId**: Unique identifier for each passenger
- **Survived**: Survival status (0 = No, 1 = Yes) - **Target Variable**
- **Pclass**: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
- **Name**: Passenger name (includes title)
- **Sex**: Gender
- **Age**: Age in years
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Ticket**: Ticket number
- **Fare**: Passenger fare
- **Cabin**: Cabin number
- **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## Tech Stack

### Core Libraries
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning algorithms and evaluation
- **Matplotlib & Seaborn** - Data visualization
- **Jupyter Notebook** - Interactive development environment

### Machine Learning Models Implemented
1. **Random Forest Classifier** - Ensemble of decision trees
2. **Gradient Boosting Classifier** - Sequential boosting algorithm
3. **Logistic Regression** - Statistical classification model
4. **Support Vector Machine (SVC)** - Kernel-based classification

### Key Techniques
- Exploratory Data Analysis (EDA)
- Missing value imputation
- Feature engineering
- Cross-validation
- Model evaluation with multiple metrics
- Data visualization


## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (optional, for cloning)

## Create and activate virtual environment

# For Windows
python -m venv venv
venv\Scripts\activate

# install libraries using PIP
pip install pandas numpy scikit-learn matplotlib seaborn jupyter notebook

# Download dataset

The dataset is included in the repository as Titanic-Dataset.csv
Or download from Kaggle and place in the project root directory

# Key Findings from Analysis
# Missing Values Analysis

Age: 177 missing values (19.9%) - Imputed with median (28 years)

Cabin: 687 missing values (77.1%) - Too many missing, excluded from initial analysis

Embarked: 2 missing values (0.2%) - Imputed with mode ('S')

Fare: 0 missing values after imputation

## Machine Learning Pipeline

1. Data Preprocessing Pipeline
Missing Value Treatment: Median imputation for Age and Fare

Categorical Encoding: Label Encoding for Sex and Embarked

Feature Scaling: StandardScaler for numerical features

Train-Test Split: 80-20 split with stratification

2. Model Training & Evaluation
Cross-Validation: 5-fold cross-validation for robust performance estimation

Multiple Algorithms: 4 different classification approaches

Comprehensive Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC

3. Performance Metrics Used
Accuracy: Overall prediction correctness

Precision: Proportion of correctly identified survivors among predicted survivors

Recall: Proportion of actual survivors correctly identified

F1-Score: Harmonic mean of precision and recall

ROC-AUC: Area under Receiver Operating Characteristic curve

Confusion Matrix: Detailed breakdown of predictions

## Expected Model Performance
# Based on typical Titanic prediction projects:

# Random Forest	
80-85%	Handles non-linear relationships, robust to outliers,Can overfit, less interpretable
# Gradient Boosting
82-87%	High predictive power, handles mixed data types	Computationally intensive, sensitive to overfitting
# Logistic Regression	
78-82%	Interpretable, fast, good baseline,Assumes linear relationships
# Support Vector Machine	
79-84%	Effective in high dimensions, versatile with kernels,Slow with large datasets, sensitive to parameters