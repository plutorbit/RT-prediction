import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from scipy.stats import randint
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

data = pd.read_csv('/Users/sarahackerman/Desktop/grouped_cystein5_feature_ex.csv')

data_sampled = data.sample(frac=0.2, random_state = 42)


X = data_sampled[['hydrophobicity','seq_length', 'molecular_weight', 'instability_index', 'isoelectric_point', 'cystein_count','aromaticity']]
y = data_sampled['retention_time']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


param_dist = {
    'n_estimators': randint(33, 45),
    'max_depth': randint(27,40),
    'min_samples_split': randint(16,30),
    'min_samples_leaf': randint(12,30),
    
}

gbm = GradientBoostingRegressor(random_state=42)

rand_search = RandomizedSearchCV(
    gbm,
    param_distributions=param_dist,
    n_iter=3,
    cv=2,
    random_state=42,
    n_jobs=-1,
    verbose= 2
)

#fitting
print("stage 2")
rand_search.fit(X_train, y_train)

print("Best parameters found:")
print(rand_search.best_params_)


best_gbm = rand_search.best_estimator_
y_pred = best_gbm.predict(X_test)

y_test_2D = np.array(y_test).reshape(1,-1)
y_pred_2D = np.array(y_pred).reshape(1,-1)


y_train_pred = best_gbm.predict(X_train)
y_test_pred = best_gbm.predict(X_test)

# metrics
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
cosine_sim = cosine_similarity(y_test_2D, y_pred_2D)

train_median_absolute_error = median_absolute_error(y_train, y_train_pred)
test_median_absolute_error = median_absolute_error(y_test, y_test_pred)



print(f'Training Mean Squared Error: {train_mae}')
print(f'Testing Mean Squared Error: {test_mae}')
print(f'Training R^2 Score: {train_r2}')
print(f'Testing R^2 Score: {test_r2}')
print(f'Training Median Absolute Error: {train_median_absolute_error}')
print(f'Testing Median Absolute Error: {test_median_absolute_error}')
print(f"Cosine Similarity: {cosine_sim}")

