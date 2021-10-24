import math

import pandas as pd
import preprocessor
import csv

def calculate_distance(first_row, second_row):
    distance = 0
    for index, data_point in enumerate(first_row):
        distance = distance + math.pow(data_point - second_row[index], 2)
    return math.sqrt(distance)

def create_and_sort_all_points_by_distance_to_current(row_to_predict, all_data, k):
    tuple_list = []
    for index, _ in enumerate(all_data):
        distance_score = calculate_distance(row_to_predict, all_data.iloc[index, :])
        tuple_list.append((all_data.iloc[index, :], distance_score))

    sorted_tuple_list = sorted(tuple_list, key=lambda x: x[1])
    return sorted_tuple_list[1:k]

def predict_zone_from_neighbors(neighbors):
    class_frequency_dict = {}

    for row in neighbors:
        if row[0]['Class'] in class_frequency_dict:
            class_frequency_dict[row[0]['Class']] = class_frequency_dict[row[0]['Class']] + 1
        else:
            class_frequency_dict[row[0]['Class']] = 1
    return max(class_frequency_dict, key=class_frequency_dict.get)

def main():
    df = pd.read_csv('train.csv')
    k_value = 100
    normalized_features, labels, num_columns = preprocessor.preprocessing_data(df)

    # for knn purposes, we want the labels for predicting the zone.
    labels.astype(int,errors='ignore')
    normalized_features['Class'] = labels

    '''
    DEV SET
    '''
    development_set = pd.read_csv('dev.csv')
    development_features, development_labels, num_columns = preprocessor.preprocessing_data(development_set)
    development_predicted_labels = []

    for x in range(len(development_features)):
        # print(f'normalized features iloc: {normalized_features.iloc[x]}')
        k_many_neighbors = create_and_sort_all_points_by_distance_to_current(development_features.iloc[x], normalized_features, k_value)
        development_predicted_labels.append(predict_zone_from_neighbors(k_many_neighbors))

    correct_count = 0
    for prediction, label in zip(development_predicted_labels, development_labels):
        if prediction == label:
            correct_count = correct_count + 1

    print(f'Accuracy count for development set KNN is: {(correct_count / len(labels)) * 100}%')

    with open('output_knn_dev.csv', 'w') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['prediction', 'label'])
        for label, prediction in zip(development_predicted_labels, development_labels):
            csv_writer.writerow([prediction, label])

    '''
    TEST SET
    '''
    test_set = pd.read_csv('test.csv')
    test_features, test_labels, num_columns = preprocessor.preprocessing_data(test_set)
    test_predicted_labels = []

    for x in range(len(test_features)):
        # print(f'normalized features iloc: {normalized_features.iloc[x]}')
        k_many_neighbors = create_and_sort_all_points_by_distance_to_current(test_features.iloc[x], normalized_features, k_value)
        test_predicted_labels.append(predict_zone_from_neighbors(k_many_neighbors))

    correct_count = 0
    for prediction, label in zip(test_predicted_labels, test_labels):
        if prediction == label:
            correct_count = correct_count + 1

    print(f'Accuracy count for test set KNN is: {(correct_count / len(labels)) * 100}%')

    with open('output_knn_test.csv', 'w') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['prediction', 'label'])
        for label, prediction in zip(test_predicted_labels, test_labels):
            csv_writer.writerow([prediction, label])


if __name__ == "__main__":
    main()