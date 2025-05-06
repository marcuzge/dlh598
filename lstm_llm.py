# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# --- 1. Generate Sample Sequence Data ---
# Let's create a simple sequence prediction task.
# Given a sequence of numbers [t, t+1, t+2], predict the next number t+3.

def generate_data(sequence_length=3, n_samples=1000):
    """Generates sample sequences."""
    X, y = [], []
    for i in range(n_samples):
        # Generate a sequence starting at a random point
        start = np.random.rand() * 10 # Start between 0 and 10
        sequence = np.array([start + j for j in range(sequence_length + 1)])
        X.append(sequence[:-1]) # Input: [t, t+1, t+2]
        y.append(sequence[-1])  # Output: t+3
    return np.array(X), np.array(y)

sequence_length = 3
n_samples = 5000
X_train_raw, y_train = generate_data(sequence_length, n_samples)

# --- 2. Prepare Data for LSTM ---
# LSTM layers expect input in the shape: [samples, timesteps, features].
# In this case, each timestep has 1 feature (the number itself).
n_features = 1
X_train = X_train_raw.reshape((X_train_raw.shape[0], X_train_raw.shape[1], n_features))

print("Shape of training input (X_train):", X_train.shape) # Should be (n_samples, sequence_length, n_features)
print("Shape of training output (y_train):", y_train.shape)   # Should be (n_samples,)
print("\nSample Input Sequence (X):", X_train_raw[0])
print("Sample Target Output (y):", y_train[0])

# --- 3. Define the LSTM Model ---
print("\n--- Building LSTM Model ---")
model = Sequential(name="Simple_LSTM_Model")

# Add the LSTM layer
# - units: Number of LSTM units (dimensionality of the output space)
# - activation: Activation function for the LSTM layer (tanh is common)
# - input_shape: Shape of the input for one sample (timesteps, features)
model.add(LSTM(units=50, activation='relu', input_shape=(sequence_length, n_features)))

# Add a Dense output layer
# - units: 1, because we are predicting a single number
# - activation: None (linear activation) for regression tasks
model.add(Dense(units=1))

# Display the model's architecture
model.summary()

# --- 4. Compile the Model ---
# - optimizer: Algorithm to update the weights (Adam is a good default)
# - loss: Function to measure the error (Mean Squared Error for regression)
print("\n--- Compiling Model ---")
model.compile(optimizer='adam', loss='mean_squared_error')

# --- 5. Train the Model ---
# - epochs: Number of times to iterate over the entire dataset
# - batch_size: Number of samples per gradient update
# - validation_split: Fraction of training data to use for validation
# - verbose: How much information to print during training (1 = progress bar)
print("\n--- Training Model ---")
history = model.fit(X_train, y_train,
                    epochs=20,
                    batch_size=32,
                    validation_split=0.2, # Use 20% of data for validation
                    verbose=1)

# --- 6. Evaluate the Model (Optional: Plot Loss) ---
print("\n--- Plotting Training History ---")
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.show()


# --- 7. Make a Prediction ---
print("\n--- Making a Prediction ---")
# Create some new data to predict
X_new_raw = np.array([[10, 11, 12], [25, 26, 27], [5.5, 6.5, 7.5]])
print("New input sequences (raw):\n", X_new_raw)

# Reshape the new data for the LSTM model
X_new = X_new_raw.reshape((X_new_raw.shape[0], X_new_raw.shape[1], n_features))
print("New input sequences (reshaped for LSTM):\n", X_new)

# Predict using the trained model
predictions = model.predict(X_new)

print("\n--- Predictions ---")
for i in range(len(X_new_raw)):
    print(f"Input: {X_new_raw[i]} -> Predicted Output: {predictions[i][0]:.2f} (Expected: {X_new_raw[i][-1] + 1})")