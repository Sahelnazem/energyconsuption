# energyconsuptionCode Explanation:

 1. Data Preprocessing:
 • Features (X) and target outputs (y) are separated.
 • The data is normalized to ensure all columns are on the same scale.
 2. Data Splitting:
 • The data is split into two sets:
 • Training set (80%): Used to train the model.
 • Test set (20%): Used to evaluate the model.
 3. Model Training:
 • The Linear Regression model is trained using the training data.
 4. Model Evaluation:
 • Metrics such as MAE, MSE, and R² are calculated for each output (Zone 1, Zone 2, Zone 3).
 • MAE (Mean Absolute Error): The average of the absolute differences between actual and predicted values.
 • MSE (Mean Squared Error): The average of the squared differences between actual and predicted values.
 • R² (Coefficient of Determination): Indicates how well the model explains the variance in the data (closer to 1 is better).
 5. Displaying Coefficients:
 • The model’s coefficients are displayed to determine which features have the most impact on predictions.
 6. Plotting a Comparison Chart:
 • A comparison chart is plotted to show the actual versus predicted values for Zone 1 Power Consumption. This visualizes the model’s accuracy and trend alignment.
