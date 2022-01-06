from matminer.datasets.convenience_loaders import load_elastic_tensor
from matminer.featurizers.conversions import StrToComposition, CompositionToOxidComposition
from matminer.featurizers.composition import ElementProperty, OxidationStates
from matminer.featurizers.structure import DensityFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

df = load_elastic_tensor()

print(df.head())
print(df.columns)

unwanted_columns = ["volume", "nsites", "compliance_tensor", "elastic_tensor", 
                    "elastic_tensor_original", "K_Voigt", "G_Voigt", "K_Reuss", "G_Reuss"]
df = df.drop(unwanted_columns, axis=1)
print(df.head())

print(df.describe())

df = StrToComposition().featurize_dataframe(df, "formula")
print(df.head())

ep_feat = ElementProperty.from_preset(preset_name="magpie")
df = ep_feat.featurize_dataframe(df, col_id="composition")  # input the "composition" column to the featurizer

print(df.head())

print(ep_feat.citations())

df = CompositionToOxidComposition().featurize_dataframe(df, "composition")

os_feat = OxidationStates()
df = os_feat.featurize_dataframe(df, "composition_oxid")
print(df.head())

df_feat = DensityFeatures()
df = df_feat.featurize_dataframe(df, "structure")  # input the structure column to the featurizer
print(df.head())

y = df['K_VRH'].values
excluded = ["G_VRH", "K_VRH", "elastic_anisotropy", "formula", "material_id", 
            "poisson_ratio", "structure", "composition", "composition_oxid"]
X = df.drop(excluded, axis=1)
print("There are {} possible descriptors:\n\n{}".format(X.shape[1], X.columns.values))

lr = LinearRegression()
lr.fit(X, y)

# get fit statistics
print('training R2 = ' + str(round(lr.score(X, y), 3)))
print('training RMSE = %.3f' % np.sqrt(mean_squared_error(y_true=y, y_pred=lr.predict(X))))

# Use 10-fold cross validation (90% training, 10% test)
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
r2_scores = cross_val_score(lr, X, y, scoring='r2', cv=crossvalidation, n_jobs=1)
scores = cross_val_score(lr, X, y, scoring='neg_mean_squared_error', cv=crossvalidation, n_jobs=1)
rmse_scores = [np.sqrt(abs(s)) for s in scores]

print('Cross-validation results:')
print('Folds: %i, mean R2: %.3f' % (len(scores), np.mean(np.abs(r2_scores))))
print('Folds: %i, mean RMSE: %.3f' % (len(scores), np.mean(np.abs(rmse_scores))))

y_pred = cross_val_predict(lr, X, y, cv=crossvalidation)

df["K_VRH LR predicted"] = y_pred
df["percentage_error LR"] = (df["K_VRH"] - df["K_VRH LR predicted"]).abs() / df["K_VRH"] * 100

reference_line = go.Scatter(
    x=[0, 400],
    y=[0, 400],
    line=dict(color="black", dash="dash"),
    mode="lines",
    showlegend=False
)

fig = px.scatter(
    df, 
    x="K_VRH", 
    y="K_VRH LR predicted", 
    hover_name="formula", 
    color="percentage_error LR", 
    color_continuous_scale=px.colors.sequential.Bluered,
)

fig.add_trace(reference_line)
#fig.show()
fig.write_image('bulk_modulus_linear_regression.png')

rf = RandomForestRegressor(n_estimators=50, random_state=1)

rf.fit(X, y)
print('training R2 = ' + str(round(rf.score(X, y), 3)))
print('training RMSE = %.3f' % np.sqrt(mean_squared_error(y_true=y, y_pred=rf.predict(X))))

# compute cross validation scores for random forest model
r2_scores = cross_val_score(rf, X, y, scoring='r2', cv=crossvalidation, n_jobs=-1)
scores = cross_val_score(rf, X, y, scoring='neg_mean_squared_error', cv=crossvalidation, n_jobs=-1)
rmse_scores = [np.sqrt(abs(s)) for s in scores]

print('Cross-validation results:')
print('Folds: %i, mean R2: %.3f' % (len(scores), np.mean(np.abs(r2_scores))))
print('Folds: %i, mean RMSE: %.3f' % (len(scores), np.mean(np.abs(rmse_scores))))

y_pred = cross_val_predict(lr, X, y, cv=crossvalidation)

df["K_VRH RF predicted"] = y_pred
df["percentage_error RF"] = (df["K_VRH"] - df["K_VRH RF predicted"]).abs() / df["K_VRH"] * 100

fig = px.scatter(
    df, 
    x="K_VRH", 
    y="K_VRH RF predicted", 
    hover_name="formula", 
    color="percentage_error RF", 
    color_continuous_scale=px.colors.sequential.Bluered,
)

fig.add_trace(reference_line)
#fig.show()
fig.write_image('bulk_modulus_random_forest_regression.png')

importances = rf.feature_importances_
included = X.columns.values
indices = np.argsort(importances)[::-1]

fig_bar = px.bar(
    x=included[indices][0:5], 
    y=importances[indices][0:5], 
    title="Feature Importances of Random Forest",
    labels={"x": "Feature", "y": "Importance"}
)
#fig_bar.show()
fig_bar.write_image('bulk_modulus_random_forest_regression_features.png')

df.to_csv('elastic_tensor_2015_mod.csv')
#df.to_json('elastic_tensor_2015_mod.json')
