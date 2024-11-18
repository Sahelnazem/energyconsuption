import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from IPython.display import display

# Load data for Plants
plant_1_generation = pd.read_csv('dataset/Plant_1_Generation_Data.csv')
plant_1_weather = pd.read_csv('dataset/Plant_1_Weather_Sensor_Data.csv')
plant_2_generation = pd.read_csv('dataset/Plant_2_Generation_Data.csv')
plant_2_weather = pd.read_csv('dataset/Plant_2_Weather_Sensor_Data.csv')

# Convert DATE_TIME columns to datetime format
plant_1_generation['DATE_TIME'] = pd.to_datetime(plant_1_generation['DATE_TIME'])
plant_1_weather['DATE_TIME'] = pd.to_datetime(plant_1_weather['DATE_TIME'])
plant_2_generation['DATE_TIME'] = pd.to_datetime(plant_2_generation['DATE_TIME'])
plant_2_weather['DATE_TIME'] = pd.to_datetime(plant_2_weather['DATE_TIME'])

# Merge Plants data
plant_1_data = pd.merge(plant_1_generation, plant_1_weather, on='DATE_TIME')
print(plant_1_data.shape)
plant_2_data = pd.merge(plant_2_generation, plant_2_weather, on='DATE_TIME')
print(plant_2_data.shape)

# Concatenate data from plants
combined_data = pd.concat([plant_1_data, plant_2_data], ignore_index=True)
print(combined_data.shape)

# Check for missing values
print(combined_data.isnull().sum())

# Extract time-based features
combined_data['HOUR'] = combined_data['DATE_TIME'].dt.hour
combined_data['DAY_OF_WEEK'] = combined_data['DATE_TIME'].dt.dayofweek
combined_data['MONTH'] = combined_data['DATE_TIME'].dt.month
print(len(combined_data.columns))
print(combined_data.columns.tolist())

# Drop unnecessary columns
combined_data.drop(['PLANT_ID_x', 'SOURCE_KEY_x', 'PLANT_ID_y', 'SOURCE_KEY_y', 'DATE_TIME', 'DAILY_YIELD', 'TOTAL_YIELD'], axis=1, inplace=True)
print(len(combined_data.columns))
print(combined_data.columns.tolist())

# Columns to scale
cols_to_scale = ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 'AC_POWER']
combined_data[cols_to_scale] = StandardScaler().fit_transform(combined_data[cols_to_scale])

for column in ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 'AC_POWER']:
    print(combined_data[column].head(), '\n')

# Define feature matrix X and target variable y
X = combined_data.drop(['AC_POWER'], axis=1)
print(X.head(), '\n')
y = combined_data['AC_POWER']
print(y.describe(), '\n')

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.size, X_test.size, y_train.size, y_test.size)

# Define the models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
    "MLP Regressor": MLPRegressor(random_state=42)
}

# Store results for evaluation
results = {}

# Train and evaluate models
for name, model in models.items():
    print(f"Training model: {name}")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Store metrics
    results[name] = {
        "MAE": mae,
        "MSE": mse,
        "R2 Score": r2
    }
    
    print(f"{name} results - MAE: {mae}, MSE: {mse}, R2 Score: {r2}\n")

# Convert results to DataFrame for display
evaluation_metrics_df = pd.DataFrame(results).T
print("Model Evaluation Metrics:")
display(evaluation_metrics_df)