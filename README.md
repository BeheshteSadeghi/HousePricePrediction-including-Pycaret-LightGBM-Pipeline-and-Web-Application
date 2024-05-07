## House Price Prediction 

### I worked on Tehran houses dataset which had been extracted from Divar site.

### The focus of this project is on Data Cleaning, Data Analysis, Model Building (based on an insight from Pycaret tool) and local Deployment of the final result (based on streamlit).

### After applying Pycaret over the cleaned data, top models are selected for hyper-parameter tuning:
#### 1. Linear Regression
#### 2. Ridge
#### 3. Lasso
#### 4. K Neighbors
#### 5. Decision Tree
#### 6. Random Forest
#### 7. XGBoost
#### 8. Light GBM

### Then the performance of these models are compared together based on R2Score and RMSE factors.
### The best model (LightGBM model) is then used to create a pipeline.
### The pipeline is saved using joblib module and used as backend for Streamlit web application.
