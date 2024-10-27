import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import median_absolute_error, mean_absolute_error
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np


data = pd.read_csv('/Users/sarahackerman/Desktop/grouped_cystein5_feature_ex.csv')
print('stage 1')

data_sampled = data.sample(frac = 0.3, random_state=42)

X = data_sampled[['hydrophobicity', 'seq_length', 'molecular_weight', 'instability_index', 'isoelectric_point', 'cystein_count', 'aromaticity']]
y = data_sampled['retention_time']
print('stage 2')


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#regressor statt classifier due to retention_time as targer variable
rf = RandomForestRegressor( # based on 20% of data set 
    random_state= 42,
    max_depth= 34,
    min_samples_split=25,
    min_samples_leaf= 11,
    n_estimators=63 
    )

gbm = GradientBoostingRegressor(
    random_state=42,
    max_depth= 39,
    min_samples_leaf= 18,
    min_samples_split= 25,
    n_estimators= 35
    )


print('stage 3')

rf.fit(X_train,y_train)
gbm.fit(X_train, y_train)
print('stage 4')

y_pred_test = gbm.predict(X_test)
y_pred_rf = rf.predict(X_test)


y_pred_blend = (y_pred_rf + y_pred_test)/2
print('stage final')


test_MAE_BLEND = mean_absolute_error(y_test, y_pred_blend)
test_MedAE_BLEND = median_absolute_error(y_test, y_pred_blend)

print(f'MAE Blend: {test_MAE_BLEND}')#
print(f'MedAE Blend: {test_MedAE_BLEND}')

