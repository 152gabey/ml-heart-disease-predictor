import sys
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4
K_VAL = 1

def main():

    # Check command-line arguments
    if len(sys.argv) != 1:
        sys.exit("Usage: python heart_disease.py")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")

def load_data():
    """
    Load healtcare data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - gender: 0 if "Male", 1 if "Female", 2 if "Other"
        - age: age of the patient in integer form
        - hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
        - heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
        - ever_married: 0 if "No", 1 if "Yes"
        - work_type: Split into the following multi-column attributes: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
        - Residence_type: 0 if "Rural", 1 if "Urban"
        - avg_glucose_level: average glucose level in blood in float form
        - bmi: body mass index in float form. values with NaN are replaced with the mean bmi
        - smoking_status: Split into the following multi-column attributes: "formerly smoked", "never smoked", "smokes" or "Unknown"
        - stroke: 1 if the patient had a stroke or 0 if not

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    df = pd.read_csv("healthcare-dataset-stroke-data.csv")
    # Remove columns not useful for prediction
    df = df.drop(columns=['id'])

    # Map columns into binary values
    df['gender'] = df['gender'].map({'Male': 0, 'Female': 1, 'Other': 2})
    df['ever_married'] = df['ever_married'].map({'No': 0, 'Yes': 1})
    df['Residence_type'] = df['Residence_type'].map({'Rural': 0, 'Urban': 1})

    # Make multiple columns for multi-categorical attributes
    df = pd.get_dummies(df, columns=['work_type', 'smoking_status'], drop_first=True)

    # Replace all NaN fields with the average value of that column
    df['bmi'] = df['bmi'].fillna(df['bmi'].mean())

    evidence = df.drop(columns='stroke')
    labels = df['stroke']
    return (evidence, labels)

def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=K_VAL)
    X_training, X_testing, y_training, y_testing = train_test_split(evidence, labels, test_size=TEST_SIZE)
    model.fit(X_training, y_training)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    sensitivity = 0
    specificity = 0
    total_positive = 0
    total_negative = 0
    for actual, predicted in zip(labels, predictions):
        if actual == 1:
            total_positive += 1
            if actual == predicted:
                sensitivity += 1
        else:
            total_negative += 1
            if actual == predicted:
                specificity += 1
    return (sensitivity/total_positive , specificity/total_negative)

if __name__ == "__main__":
    main()
