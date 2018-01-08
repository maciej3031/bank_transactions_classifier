import os

import pandas as pd

from model import NeuralNetwork
from utils import split_data, plot_confusion_matrix, plot_loss, get_under_sample_dataset

if __name__ == '__main__':
    dataset = pd.read_csv(os.path.join('data', 'creditcard.csv'))
    assert not dataset.isnull().values.any()
    dataset = dataset.drop(['Time', 'Amount'], axis=1)

    # Split dataset on train_and_validation dataset and test dataset
    train_and_validation, test = split_data(dataset, train_size=0.8)

    # Convert test data to numpyarray and split them.
    test = test.values
    x_test = test[:, :-1]
    y_test = test[:, -1:]

    # Create balanced, under sample train and validation dataset
    under_sample_dataset = get_under_sample_dataset(dataset, train_and_validation)

    # Convert training and validation dataset to numpy array
    under_sample_dataset = under_sample_dataset.values

    train, validation = split_data(under_sample_dataset, train_size=0.8)

    x_train = train[:, :-1]
    y_train = train[:, -1:]

    x_validation = validation[:, :-1]
    y_validation = validation[:, -1:]

    # 653
    model = NeuralNetwork(learning_rate=0.002, batch_size=653, epochs=100, loss='mse', regular_lambda=0.1)
    model.add_layer(input_dim=x_test.shape[1], neurons_number=512, activation='relu')
    model.add_layer(input_dim=512, neurons_number=1, activation='sigmoid')
    model.fit(x_train, y_train, x_validation, y_validation)

    print("\nValidation dataset evaluation:")
    model.evaluate(x_validation, y_validation)
    plot_loss(model.epochs, model.training_history, model.validation_history)

    print("\nTest dataset evaluation:")
    x_train_and_validation = under_sample_dataset[:, :-1]
    y_train_and_validation = under_sample_dataset[:, -1:]

    model.evaluate(x_test, y_test)
    plot_confusion_matrix(model, x_test, y_test)
    # print(model.layers[0].weights)

    # ROC, confusion matrix, learning curves.
