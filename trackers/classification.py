import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

# Generate more sample data for fuel consumption (liters/hour) and labels
np.random.seed(42)
fuel_consumption = np.random.uniform(3, 8, 100)  # Random values between 3 and 8 liters/hour
labels = (fuel_consumption <= 5).astype(int)  # Efficient if <= 5 liters/hour

# Train logistic regression model
fuel_consumption_reshaped = fuel_consumption.reshape(-1, 1)  # Reshape for sklearn
model = LogisticRegression()
model.fit(fuel_consumption_reshaped, labels)

# Generate points for the logistic regression line
x_values = np.linspace(3, 8, 500).reshape(-1, 1)  # Smooth line from 3 to 8 liters/hour
probabilities = model.predict_proba(x_values)[:, 1]  # Probability of being efficient

# Scatter plot of data
plt.scatter(fuel_consumption[labels == 1], labels[labels == 1], color='green', label='Efficient (1)', alpha=0.6)
plt.scatter(fuel_consumption[labels == 0], labels[labels == 0], color='red', label='Not Efficient (0)', alpha=0.6)

# Plot logistic regression curve
plt.plot(x_values, probabilities, color='blue', label='Logistic Regression Curve')

# Formatting the plot
plt.xlabel('Fuel Consumption (liters/hour)')
plt.ylabel('Efficiency Probability')
plt.title('Logistic Regression for Engine Efficiency')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
