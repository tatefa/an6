import numpy as np
import tensorflow as tf

# Define training data
training_data = {
    "0": np.array([[1, 1, 1],
                   [1, 0, 1],
                   [1, 0, 1],
                   [1, 0, 1],
                   [1, 1, 1]]),
    "1": np.array([[0, 1, 0],
                   [0, 1, 0],
                   [0, 1, 0],
                   [0, 1, 0],
                   [0, 1, 0]]),
    "2": np.array([[1, 1, 1],
                   [0, 0, 1],
                   [1, 1, 1],
                   [1, 0, 0],
                   [1, 1, 1]]),
    "39": np.array([[1, 1, 1],
                    [0, 0, 1],
                    [1, 1, 1],
                    [0, 0, 1],
                    [1, 1, 1]])
}

# Define labels
labels = {
    "0": [1, 0, 0, 0],
    "1": [0, 1, 0, 0],
    "2": [0, 0, 1, 0],
    "39": [0, 0, 0, 1]
}

# Convert data and labels to TensorFlow tensors
train_x = np.array([training_data[key].flatten() for key in training_data])
train_y = np.array([labels[key] for key in training_data])

# Define neural network architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, input_shape=(15,), activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

def train_model():
    # Train the model
    model.fit(train_x, train_y, epochs=1000)
    _, accuracy = model.evaluate(train_x, train_y)
    print(f"Model trained with accuracy: {accuracy*100:.2f}%")

def test_model(test_data):
    # Test the model
    test_x = np.array([test_data.flatten()])
    predictions = model.predict(test_x)

    # Convert predictions to human-readable labels
    predicted_number = np.argmax(predictions)
    print("Predicted number:", predicted_number)

def menu():
    while True:
        print("\nMenu:")
        print("1. Train Model")
        print("2. Test Model")
        print("3. Exit")

        choice = input("Enter your choice: ")

        if choice == "1":
            train_model()
            
        elif choice == "2":
            sample_test_data = np.array([[1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1]])
            test_model(sample_test_data)
        elif choice == "3":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter a valid option.")

if __name__ == "__main__":
    menu()
