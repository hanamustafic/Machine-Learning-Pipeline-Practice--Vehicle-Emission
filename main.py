#imports 
import pandas as pd 
import numpy as np 
import joblib 
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.model_selection import train_test_split #flashcards podjela podataka na test i train
from sklearn.pipeline import Pipeline  #spajanje vise koraka u obradi podataka i treniranju modela u jedan objekat 
from sklearn.compose import ColumnTransformer  #obrada razlicitih kolona razlicitih trans
from sklearn.preprocessing import StandardScaler, OneHotEncoder  #pretvaranje u 0 i 1
from sklearn.impute import SimpleImputer  #popunjava nedostajuce podatke 
from sklearn.ensemble import RandomForestRegressor  #algoritam ucenja 
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load the dataset from a CSV file
data= pd.read_csv("vehicle_emissions.csv")
print(data.head())
data.info()

# Separate features (X) and target variable (y)
x= data.drop(["CO2_Emissions"], axis=1)
y= data["CO2_Emissions"]

# Define numerical and categorical columns
numerical_cols = ['Engine_Size', 'Cylinders', 'Fuel_Consumption_in_City(L/100 km)',
                  'Fuel_Consumption_in_City_Hwy(L/100 km)', 'Fuel_Consumption_comb(L/100km)']
categorical_cols = ['Make', 'Model', 'Vehicle_Class', 'Transmission', 'Smog_Level']

# Pipeline for preprocessing numerical data
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="mean")), #ukoliko nedostaje data, popunjava to mjesto s prosjekom te kolone
    ('scaler', StandardScaler()) #standardizira datu da ima rang od 0 do 1
  ])

# Pipeline for preprocessing categorical data
categorical_pipeline= Pipeline([
    ('imputer',SimpleImputer(strategy="most_frequent")), #radi se o stringu pa ne mozemo uzeti prosjek vec koristimo najcesci string
    ('encoder', OneHotEncoder(handle_unknown='ignore'))  #konvertira stringove u binarne vrijednosti 
  ])

# Combine both pipelines using ColumnTransformer
preprocessor = ColumnTransformer([  #cleaning
    ('num', numerical_pipeline, numerical_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

# Final pipeline that includes preprocessing and the machine learning model
pipeline = Pipeline([
    ('preprocessor', preprocessor), #gotovo ciscenje
    ('model', RandomForestRegressor()) #model gdje ce data zavrsiti
])

# Split dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

#train and predict model 
pipeline.fit(x_train, y_train)

prediction= pipeline.predict(x_test)

#view the encoding that was done 
encoded_cols= pipeline.named_steps['preprocessor'].named_transformers_['cat']['encoder'].get_feature_names_out(categorical_cols)
print(encoded_cols)

#evaluate model accuracy
#metricke funkcije koje se koriste za regresivne modele poput randomforestregression
mse= mean_squared_error(y_test, prediction)
rmse= np.sqrt(mse) # Root mean squared error

r2 = r2_score(y_test, prediction) #koliko model objasnjava varijaciju ciljane varijable
mae = mean_absolute_error(y_test, prediction) #prosjecna apsolutna greska 

print(f'Model Performance:')
print(f'R2 score:{r2}')  # Higher is better; 1.0 indicates perfect prediction
print(f'Root Mean Square Error:{rmse}')   # Lower is better
print(f'Mean Absolute Error:{mae}')  # Lower is better

joblib.dump(pipeline, 'vehicle_emissions_pipeline.joblib')


metrics = ['R²', 'RMSE', 'MAE']
values = [r2, rmse, mae]

plt.bar(metrics, values, color=['blue', 'orange', 'green'])
plt.title('Model Performance Metrics')
plt.ylabel('Value')
for i, v in enumerate(values):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
plt.show()


plt.figure(figsize=(8, 6))
plt.scatter(y_test, prediction, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.title('Actual vs Predicted CO₂ Emissions')
plt.xlabel('Actual CO₂ Emissions')
plt.ylabel('Predicted CO₂ Emissions')
plt.show()


corr_matrix = data[numerical_cols + ['CO2_Emissions']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()