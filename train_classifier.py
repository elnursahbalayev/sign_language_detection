# Import pickle module for data serialization and deserialization
import pickle
# Import RandomForestClassifier to build a random forest classification model
from sklearn.ensemble import RandomForestClassifier
# Import train_test_split to split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
# Import accuracy_score to calculate the accuracy of predictions
from sklearn.metrics import accuracy_score
# Import numpy for array handling

import numpy as np

# Load the data dictionary from a pickle file, which includes training data and labels
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Convert the data and labels to numpy arrays for easier processing
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split the dataset into 80% training and 20% testing sets
# Shuffle is set to True to randomize the data before splitting
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Create an instance of the RandomForestClassifier
model = RandomForestClassifier()

# Train the model using the training data
model.fit(x_train, y_train)

# Predict the labels for the test data
y_predict = model.predict(x_test)

# Calculate the accuracy of the predictions
score = accuracy_score(y_predict, y_test)

# Print the accuracy score formatted as a percentage
print(f'{score*100}% of samples were classified correctly!')

# Save the trained model to a file named 'model.p' using pickle
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

