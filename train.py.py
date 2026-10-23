# Yield Prediction Model Training 

# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from joblib import dump
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


# Step 2: Load the dataset

df = pd.read_csv('D:\\NareshIT Practicals\\11AM Batch\\smart-agriculture-mlops\\data\\yield_dataset.csv')
print("="*100)
print(df.head())

print("="*100)

# print the information about the dataset
print(df.info())
print("="*100)

# Step 3: Preprocess the data
print("Preprocessing the data...")

# check for missing values
print(df.isnull().sum())
print("="*100)

# drop the missing values
df = df.dropna()
print("Missing values dropped.")
print("="*100)

# Check the all unique values in the categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns

for col in categorical_columns:
    print(f"Unique values in {col}: {df[col].unique()}")
    print("="*100)

# handled the categorical columns name 
for col in categorical_columns:
    df[col] = df[col].astype('str').str.strip().str.title()
print("Categorical columns processed.")
print("="*100)

# handled the Planting Date

df['Planting Date'] = pd.to_datetime(df['Planting Date'],
                                    format='%d/%m/%Y', 
                                    errors='coerce'
                                    )

# Extract the year, month, and day from the Planting Date
df['Planting_Year'] = df['Planting Date'].dt.year
df['Planting_Month'] = df['Planting Date'].dt.month
df['Planting_Day'] = df['Planting Date'].dt.day
df['Planting_DayOfYear'] = df['Planting Date'].dt.dayofyear
df['Planting_Quarter'] = df['Planting Date'].dt.quarter

# Drop the original Planting Date column
df = df.drop(columns=['Planting Date'])

print("Planting Date processed.")
print("="*100)

# check the the invalid value Field Size (hectares)
print('Invalid values in Field Size (hectares):', df[df['Field Size (hectares)'] <= 0].shape[0])
print("="*100)

# check the the invalid value Yield_t_per_ha
print('Invalid values in Yield_t_per_ha:', df[df['Yield_t_per_ha'] <= 0].shape[0])
print("="*100)

# check the outlier in the Yield_t_per_ha

Q1 = df['Yield_t_per_ha'].quantile(0.25)
Q3 = df['Yield_t_per_ha'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['Yield_t_per_ha'] < lower_bound) | (df['Yield_t_per_ha'] > upper_bound)]
print(f"Number of outliers in Yield_t_per_ha: {outliers.shape[0]}")
print("="*100)

# Step 4: Split the data into features and target variable
X = df.drop(columns=['Yield_t_per_ha'])
y = df['Yield_t_per_ha']

print("Data split into features and target variable.")
print("="*100)

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split into training and testing sets.")
print("="*100)

# divide the columns into numerical and categorical columns

categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist
print("Categorical features:", categorical_features)
print("="*100)
print("Numerical features:", numerical_features)
print("="*100)

# handled the missing value 
numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# create a preprocessor that combines the numerical and categorical pipelines
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features)
])

print("Preprocessing pipelines created.")
print("="*100)

# model training using RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

# create a pipeline that combines the preprocessor and the model
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# Step 6: Train the model
print("Training the model...")
rf_pipeline.fit(X_train, y_train)

# predict the target variable for the test set
y_pred = rf_pipeline.predict(X_test)
print("Model training completed.")
print("="*100)

# Step 7: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print("="*100)

# Step 8: R2 score
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(f"R2 Score: {r2}")
print("="*100)
