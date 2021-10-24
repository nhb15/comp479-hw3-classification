import csv

from sklearn import svm
import pandas as pd
import preprocessor

def main():
    df = pd.read_csv('train.csv')
    df, labels, num_columns = preprocessor.preprocessing_data(df)
    iterations = 2000

    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(df.values, labels.values)

    '''
    DEV SET
    '''

    development_set = pd.read_csv('dev.csv')
    development_features, development_labels, num_columns = preprocessor.preprocessing_data(development_set)
    development_predicted_labels = []
    for index in range(len(development_features)):
        info_for_prediction = development_features.iloc[index-1].values.reshape(1,-1)

        prediction = clf.predict(info_for_prediction)
        development_predicted_labels.append(prediction)

    correct_count = 0
    for prediction, label in zip(development_predicted_labels, labels):
        if prediction == label:
            correct_count = correct_count + 1

    print(f'Accuracy count for SVM {iterations} iterations is: {(correct_count / len(labels)) * 100}%')

    with open('output_svm_dev.csv', 'w') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['prediction', 'label'])
        for prediction, label in zip(development_predicted_labels, labels):
            csv_writer.writerow([prediction, label])

    '''
    TEST SET
    '''
    test_set = pd.read_csv('test.csv')
    test_features, test_labels, num_columns = preprocessor.preprocessing_data(test_set)
    test_predicted_labels = []
    for index in range(len(test_features)):
        info_for_prediction = test_features.iloc[index-1].values.reshape(1,-1)

        prediction = clf.predict(info_for_prediction)
        test_predicted_labels.append(prediction)

    correct_count = 0
    for prediction, label in zip(test_predicted_labels, labels):
        if prediction == label:
            correct_count = correct_count + 1

    print(f'Accuracy count for SVM {iterations} iterations on TEST set is: {(correct_count / len(labels)) * 100}%')

    with open('output_svm_test.csv', 'w') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['prediction', 'label'])
        for prediction, label in zip(test_predicted_labels, labels):
            csv_writer.writerow([prediction, label])

if __name__ == "__main__":
    main()