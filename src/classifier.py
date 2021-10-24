import csv

from sklearn.linear_model import LogisticRegressionCV
import pandas as pd
import preprocessor

def main():
    df = pd.read_csv('train.csv')
    df, labels, num_columns = preprocessor.preprocessing_data(df)
    iterations = 2000

    trained_musical_classifier = LogisticRegressionCV(max_iter=iterations).fit(df.values, labels.values)
    development_set = pd.read_csv('dev.csv')
    development_features, development_labels, num_columns = preprocessor.preprocessing_data(development_set)
    development_predicted_labels = []
    for index in range(len(development_features)):
        development_predicted_labels.append(trained_musical_classifier.predict(development_features.iloc[index-1].values.reshape(1,-1)))

    correct_count = 0
    for prediction, label in zip(development_predicted_labels, development_labels):
        if prediction[0] == label:
            correct_count = correct_count + 1

    print(f'Accuracy count for LinearRegressionCV {iterations} iterations is: {(correct_count / len(labels)) * 100}%')

    with open('output_classifier_dev.csv', 'w') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['prediction', 'label'])
        for label, prediction in zip(development_predicted_labels, development_labels):
            csv_writer.writerow([prediction, label[0]])

    '''
    TEST SET
    '''
    test_set = pd.read_csv('test.csv')
    test_features, test_labels, num_columns = preprocessor.preprocessing_data(test_set)
    test_predicted_labels = []
    for index in range(len(test_features)):
        info_for_prediction = test_features.iloc[index-1].values.reshape(1,-1)

        prediction = trained_musical_classifier.predict(info_for_prediction)
        test_predicted_labels.append(prediction)

    correct_count = 0
    for prediction, label in zip(test_predicted_labels, labels):
        if prediction == label:
            correct_count = correct_count + 1

    print(f'Accuracy count for Linear Regression {iterations} iterations on TEST set is: {(correct_count / len(labels)) * 100}%')

    with open('output_classifier_test.csv', 'w') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['prediction', 'label'])
        for prediction, label in zip(test_predicted_labels, labels):
            csv_writer.writerow([prediction, label])


if __name__ == "__main__":
    main()