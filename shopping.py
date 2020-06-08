import csv
import sys
import calendar

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
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

def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    data = list()
    evidence = list()
    labels = list()

    # Dictionary for months name conversion to num
    abbr_to_num = {name: num for num, name in enumerate(calendar.month_abbr) if num}
    abbr_to_num["June"] = abbr_to_num.pop("Jun")

    # Get data from file
    with open(filename, newline='', encoding='utf-8') as csvfile:
        datareader = csv.reader(csvfile)
        for row in datareader:
            data.append(row)
    
    # Prepare data for dataset creation
    field_names = data[0]
    num_fields = len(field_names)
    data = data[1:]

    # Populate 'evidence' and 'labels' with data
    j = 0
    for row in data:

        # Initialize a record for 'evidence'
        record = list()

        # Conform to the standards in the description and populate the record
        for i in range(num_fields - 1):

            if i == field_names.index("Month"):
                if row[i] != None:
                    record.append(abbr_to_num[row[i]])
                else:
                    print("None", i, j, row[i])

            elif i == field_names.index("VisitorType"):
                if row[i] != None and row[i] == "Returning_Visitor":
                    record.append(1)
                elif row[i] != None and (row[i] == "New_Visitor" or row[i] == "Other"):
                    record.append(0)
                else:
                    print("None", i, j, row[i])

            elif i == field_names.index("Weekend"):
                if row[i] != None and row[i] == "TRUE":
                    record.append(1)
                elif row[i] != None and row[i] == "FALSE":
                    record.append(0)
                else:
                    print("None", i, j, row[i])

            else:
                if row[i] != None:
                    try:
                        record.append(int(row[i]))
                    except ValueError:
                        try:
                            record.append(float(row[i]))
                        except ValueError:
                            print("Fail")
                else:
                    print("None:", i, j, row[i])
        j+=1
        # Append the record to 'evidence' list
        evidence.append(record)

        # Make 'labels' list
        if row[num_fields - 1] != None and row[num_fields - 1] == "TRUE":
            labels.append(1)
        elif row[num_fields - 1] != None and row[num_fields - 1] == "FALSE":
            labels.append(0)

    # Return the dataset
    return evidence, labels


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    trained_model = model.fit(evidence, labels)
    return trained_model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    sensitivity = 0
    true_label = 0
    specificity = 0
    false_label = 0
    number_of_predictions = len(predictions)
    for i in range(number_of_predictions):
        if labels[i] == 1:
            true_label += 1
            if predictions[i] == labels[i]:
                sensitivity += 1
        elif labels[i] == 0:
            false_label += 1
            if predictions[i] == labels[i]:
                specificity += 1
    sensitivity /= true_label
    specificity /= false_label
    return sensitivity, specificity

if __name__ == "__main__":
    main()
