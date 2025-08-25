import sys
import pandas as pd
import tensorflow.keras as tf

from sklearn.model_selection import train_test_split

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
    model = create_model()
    model.fit(X_train, y_train, epochs=20)
    model.evaluate(X_test, y_test, verbose=2)

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

    labels should be the corresponding list of labels, where each label
    is 1 if patient has a stroke, and 0 otherwise.
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
    return evidence, labels

def create_model():
    """
    Costructs a neural network with an 8 unit hidden layer
    outputs a value between 0 and 1 to determine stroke
    """
    model = tf.models.Sequential()
    model.add(tf.layers.Dense(8, input_shape=(15,), activation="relu"))
    model.add(tf.layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

if __name__ == "__main__":
    main()
