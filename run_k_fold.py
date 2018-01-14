import os

import numpy as np
import pandas as pd

from model import NeuralNetwork
from utils import k_fold_split_data, split_data, plot_confusion_matrix, plot_loss, get_under_sample_dataset

if __name__ == '__main__':
    dataset = pd.read_csv(os.path.join( 'creditcard.csv'))
    assert not dataset.isnull().values.any()
    dataset = dataset.drop(['Time', 'Amount'], axis=1)

    # Split dataset on train_and_validation dataset and test dataset
    train_and_validation, test = split_data(dataset, train_size=0.8)

    # Convert test data to numpyarray and split them.
    test = test.values
    x_test = test[:, :-1]
    y_test = test[:, -1:]

    #Split good and bad transactions
    fraud_indices = np.array(train_and_validation[train_and_validation.Class == 1].index)
    normal_indices = np.array(train_and_validation[train_and_validation.Class == 0].index)

    # Create balanced, under sample train and validation dataset
    under_sample_dataset = get_under_sample_dataset(dataset, train_and_validation)

    # Convert training and validation dataset to numpy array
    under_sample_dataset = under_sample_dataset.values

    # k-fold validation with k=5
    k_fold_split_array = k_fold_split_data(fraud_indices, 5)
    models = []
    for n, validation in enumerate(k_fold_split_array):

        #losowanie dobrych tranzakcji
        random_normal_indices = np.array(np.random.choice(normal_indices, fraud_indices.shape[0], replace=False))
        np.random.shuffle(random_normal_indices)
        normal_indices_split_array = np.array_split(random_normal_indices, 5)

        validation_normal_indices = normal_indices_split_array.pop(n)

        train = k_fold_split_array.copy()
        train.pop(n)
        train = np.concatenate([train,normal_indices_split_array])
        np.random.shuffle(train)

        x_train = train[:, :-1]
        y_train = train[:, -1:]

        validation = np.concatenate([validation,validation_normal_indices])
        np.random.shuffle(validation)

        x_validation = validation[:, :-1]
        y_validation = validation[:, -1:]

        # 653
        model = NeuralNetwork(learning_rate=0.002, batch_size=653, epochs=80, loss='mse', regular_lambda=0.1)
        model.add_layer(input_dim=x_test.shape[1], neurons_number=512, activation='relu')
        model.add_layer(input_dim=512, neurons_number=1, activation='sigmoid')
        model.fit(x_train, y_train, x_validation, y_validation)

        print("\nValidation dataset evaluation:")
        model.evaluate(x_validation, y_validation)
        print("\n")
        plot_loss(model.epochs, model.training_history, model.validation_history)
        models.append(model)

    training_history = np.average([mdl.training_history for mdl in models], axis=0)
    validation_history = np.average([mdl.validation_history for mdl in models], axis=0)

    plot_loss(80, training_history, validation_history)

    print("\nTest dataset evaluation:")
    x_train_and_validation = under_sample_dataset[:, :-1]
    y_train_and_validation = under_sample_dataset[:, -1:]

    model = NeuralNetwork(learning_rate=0.002, batch_size=653, epochs=80, loss='mse', regular_lambda=0.1)
    model.add_layer(input_dim=x_test.shape[1], neurons_number=512, activation='relu')
    model.add_layer(input_dim=512, neurons_number=1, activation='sigmoid')
    model.fit(x_train_and_validation, y_train_and_validation)

    model.evaluate(x_test, y_test)
    plot_confusion_matrix(model, x_test, y_test)
    # print(model.layers[0].weights)

    # ROC, confusion matrix, learning curves.
