import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
df = pd.read_csv("Dataset/diabetes.csv")

# Preprocess the data
X = df.drop(columns='Outcome', axis=1)
Y = df['Outcome']

scaler = StandardScaler()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Fit the scaler to the training data and transform both training and testing data
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler to a file
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

# Define the base models
estimators = [
    ('log_reg', LogisticRegression(C=41.702783448252696, max_iter=7231, penalty='l2', solver='liblinear')),
    ('svc', SVC(C=14.937087470469613, gamma=0.0001, probability=True)),
    ('gbc', GradientBoostingClassifier())
]

# Define the stacking classifier
stacking_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

# Train the stacking classifier
stacking_clf.fit(X_train, Y_train)

# Evaluate the model
train_acc = stacking_clf.score(X_train, Y_train)
test_acc = stacking_clf.score(X_test, Y_test)
Y_pred = stacking_clf.predict(X_test)

print(f"Stacking Classifier Training Accuracy: {train_acc:.4f}")
print(f"Stacking Classifier Testing Accuracy: {test_acc:.4f}")
print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))

# Save the model
filename = 'diabetes_model.pkl'
with open(filename, 'wb') as f:
    pickle.dump(stacking_clf, f)

# Load the model
with open('diabetes_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Example prediction
input_data = (5,166,72,19,175,25.8,0.587,51)
input_data_as_numpy_array = np.asarray(input_data).reshape(1,-1)
std_data = scaler.transform(input_data_as_numpy_array)

prediction = loaded_model.predict(std_data)

if prediction[0] == 0:
    print('The person is not diabetic')
else:
    print('The person is diabetic')
