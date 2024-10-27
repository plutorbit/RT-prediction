import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import  r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

data = pd.read_csv('/Users/sarahackerman/Desktop/grouped_cystein5_feature_ex.csv')



X = data[['hydrophobicity','seq_length', 'molecular_weight', 'instability_index', 'isoelectric_point', 'cystein_count','aromaticity']]
y = data['retention_time']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Step 4: Train the Gradient Boosting Regressor
gbm = GradientBoostingRegressor(
    random_state=42,
    max_depth= 49,
    min_samples_leaf= 14,
    min_samples_split= 11,
    n_estimators= 46
    )


#fitting 
print("stage 2")
gbm.fit(X_train, y_train)

y_train_pred = gbm.predict(X_train)
y_pred_test = gbm.predict(X_test)


y_test_2D = np.array(y_test).reshape(1,-1)
y_pred_2D = np.array(y_pred_test).reshape(1,-1)

#metrics 
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_pred_test)
cosine_sim = cosine_similarity(y_test_2D, y_pred_2D)

train_median_absolute_error = median_absolute_error(y_train, y_train_pred)
test_median_absolute_error = median_absolute_error(y_test, y_pred_test)



print(f'Training Mean Absolute Error: {train_mae}')
print(f'Testing Mean Absolute Error: {test_mae}')
print(f'Training R^2 Score: {train_r2}')
print(f'Testing R^2 Score: {test_r2}')
print(f'Training Median Absolute Error: {train_median_absolute_error}')
print(f'Testing Median Absolute Error: {test_median_absolute_error}')
print(f"Cosine Similarity: {cosine_sim[0][0]}")

