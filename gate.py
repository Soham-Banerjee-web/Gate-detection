import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Define truth tables for different logic gates
truth_tables = {
    "AND":    [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]],
    "OR":     [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]],
    "XOR":    [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]],
    "NAND":   [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0]],
    "NOR":    [[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 0]],
}

# Prepare dataset (Flatten truth tables)
X = []
y = []
gate_labels = list(truth_tables.keys())  # ["AND", "OR", "XOR", "NAND", "NOR"]

for gate, table in truth_tables.items():
    flattened_inputs = np.array(table).flatten()  # Flatten to 1D array (8 features)
    X.append(flattened_inputs)
    y.append(gate_labels.index(gate))  # Class label

X = np.array(X)
y = to_categorical(y, num_classes=len(gate_labels))  # Convert labels to one-hot encoding

# Define Neural Network Model
model = Sequential([
    Dense(16, activation='relu', input_shape=(12,)),  # Increased neurons
    Dense(16, activation='relu'),
    Dense(len(gate_labels), activation='softmax')  # Output layer (5 classes)
])

# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(X, y, epochs=2000, verbose=0)  # Train for 2000 epochs for better accuracy

def predict_gate():
    """Get user input and predict which logic gate it is."""
    print("\nEnter truth table values (separate numbers with spaces):")
    
    user_truth_table = []
    for i in range(4):
        row = list(map(int, input(f"Input {i+1} (x1, x2, output): ").split()))
        if len(row) != 3:
            print("❌ Error: Please enter exactly 3 values (x1, x2, output). Try again.")
            return
        user_truth_table.append(row)

    # Flatten input into (1, 12) shape
    input_data = np.array(user_truth_table).flatten().reshape(1, -1)

    # Predict probabilities
    predictions = model.predict(input_data)
    predicted_gate = gate_labels[np.argmax(predictions)]  # Get the highest probability class
    
    print(f"\n✅ Predicted Logic Gate: {predicted_gate}")

# Run the function
predict_gate()
