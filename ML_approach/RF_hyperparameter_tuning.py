import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    median_absolute_error,
    r2_score,
    mean_absolute_error,
)
from scipy.stats import randint
import matplotlib.pyplot as plt


data = pd.read_csv('/Users/sarahackerman/Desktop/grouped_cystein5_feature_ex.csv')
print('stage 1')

data_sampled = data.sample(frac = 0.05, random_state=42)

X_sampled = data_sampled[['hydrophobicity', 'seq_length', 'molecular_weight', 'instability_index', 'isoelectric_point', 'cystein_count', 'aromaticity']]
y_sampled = data_sampled['retention_time']
print('stage 2')


X_train, X_test, y_train, y_test = train_test_split(X_sampled, y_sampled, test_size=0.2, random_state=42)

#regressor instead of classifier due to retention_time as targer variable
rf = RandomForestRegressor(random_state=42)
print('stage 3')
#hyperparameter distiction
param_dist = {
    'n_estimators': randint(10,30),
    'max_depth': randint (5,20), 
    'min_samples_split': randint (5,20),  
    'min_samples_leaf': randint (5, 20)     
}
print('stage 4')
#iterating w/ searchCV 
rand_search = RandomizedSearchCV(
    rf, 
    param_distributions= param_dist,
    n_iter=3,
    cv=2,
    random_state=42,
    n_jobs=-1,
    verbose= 2
)

#fitting searchCV to model 
rand_search.fit(X_train, y_train)
print('stage 5')
#best estimator from searchCV 
best_rf = rand_search.best_estimator_
print(f"Best hyperparameters: {rand_search.best_params_}")

# predict w/ best estimator 
y_pred = best_rf.predict(X_test)
print(y_pred)
print('stage final')

# evaluating  model -> medAE for blended prediction 
test_MAE = mean_absolute_error(y_test, y_pred)
test_MedAE = median_absolute_error(y_test, y_pred)
test_MSE = mean_squared_error(y_test, y_pred)
test_R2 = r2_score(y_test, y_pred)

print(f'MAE Test: {test_MAE}')
print(f'MedAE Test: {test_MedAE}')
print(f'MSE Test: {test_MSE}')
print(f'R^2 Test: {test_R2}')

#plotting pred against true 
#plt.scatter(y_test, y_pred, alpha=0.6)
#plt.xlabel('Actual Retention Time')
#plt.ylabel('Predicted Retention Time')
#plt.title('Actual vs Predicted Retention Time')
#plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Reference line
#plt.show()


