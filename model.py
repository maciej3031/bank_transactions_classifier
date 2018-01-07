import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt


def sigmoid(x):
    return np.multiply(0.5, (1 + np.tanh(np.multiply(0.5, x))))


def sigmoid_derivative(x):
    return np.multiply(x, (1 - x))


class Layer:
    def __init__(self, input_dim, neurons_number, learning_rate, activation):
        self.activation = activation
        self.learning_rate = learning_rate

        self.output_layer = True

        self.input = np.asmatrix(np.zeros((input_dim + 1, 1)))
        self.output = np.asmatrix(np.zeros((neurons_number, 1)))
        self.weights = np.asmatrix(np.random.uniform(low=-1, high=1, size=(input_dim + 1, neurons_number)))
        self.deltas = np.asmatrix(np.zeros((neurons_number, 1)))
        self.cumulative_gradient = np.asmatrix(np.zeros((input_dim + 1, neurons_number)))

    def _activate(self, matrix):
        if self.activation == 'sigmoid':
            return sigmoid(matrix)

    def _get_gradient(self):
        return np.matmul(self.input, self.deltas.transpose())

    def forward_step(self, input_data):
        self.input = np.concatenate([[[1]], input_data])  # Add bias
        self.output = self._activate(np.matmul(self.weights.transpose(), self.input))
        return self.output

    def backward_step(self, next_weights=None, next_deltas=None, output_delta=None):
        if self.output_layer:
            self.deltas = output_delta
        else:
            derivative_of_activation = sigmoid_derivative(self.output)
            self.deltas = np.multiply(np.matmul(np.delete(next_weights, 0, 0), next_deltas),
                                      derivative_of_activation)  # Exclude bias row from weights

        self.cumulative_gradient = self.cumulative_gradient + self._get_gradient()

    def get_deltas(self):
        return self.deltas

    def get_weights(self):
        return self.weights


class NeuralNetwork:
    def __init__(self, learning_rate, batch_size=50, epochs=20, loss='mse', regular_lambda=0.1):
        self.layers = []
        self.lerning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss = loss
        self.regular_lambda = regular_lambda

        self.training_history = []
        self.validation_history = []

    def _global_forward_step(self, x_train_batch):
        y_predicted_batch = []
        for x_train_record in x_train_batch:
            x_train_record = x_train_record.reshape((-1, 1))
            for layer in self.layers:
                x_train_record = layer.forward_step(x_train_record)
            y_predicted_batch.append(x_train_record)

        return np.concatenate(y_predicted_batch)

    def _global_backward_step(self, y_predicted_record, y_train_record):

        output_delta = self._count_output_delta(y_predicted_record, y_train_record)

        output_layer = self.layers[-1]
        output_layer.backward_step(output_delta=output_delta)

        next_deltas = output_layer.get_deltas()
        next_weights = output_layer.get_weights()

        for layer in reversed(self.layers[:-1]):
            layer.backward_step(next_weights=next_weights, next_deltas=next_deltas)

    def _count_output_delta(self, y_predicted, y_actual):
        if self.loss == 'mse':
            return np.sum(np.multiply((y_predicted - y_actual), sigmoid_derivative(y_predicted)), axis=0)

    def _count_loss(self, y_predicted, y_actual):
        if self.loss == 'mse':
            return mean_squared_error(y_actual, y_predicted)

    def _save_loss(self, x_train, y_train, x_validation, y_validation):
        y_predicted_train = self._global_forward_step(x_train)
        y_predicted_validation = self._global_forward_step(x_validation)

        train_loss = self._count_loss(y_predicted_train, y_train)
        validation_loss = self._count_loss(y_predicted_validation, y_validation)

        self.training_history.append(train_loss)
        self.validation_history.append(validation_loss)

        print("Loss: ", train_loss)
        print("Accuracy: ", accuracy_score(y_train, np.round(y_predicted_train)))

    def add_layer(self, input_dim, neurons_number, activation='sigmoid'):
        layer = Layer(input_dim, neurons_number, learning_rate=self.lerning_rate, activation=activation)
        if self.layers:
            self.layers[-1].output_layer = False
        self.layers.append(layer)

    def fit(self, x_train, y_train, x_validation, y_validation):
        for i in range(self.epochs):
            for idx in range(0, x_train.shape[0], self.batch_size):
                x_train_batch = x_train[idx:idx + self.batch_size]
                y_train_batch = y_train[idx:idx + self.batch_size]

                for n, x_train_record in enumerate(x_train_batch):
                    x_train_record = x_train_record.reshape(1, -1)
                    y_predicted_record = self._global_forward_step(x_train_record)
                    self._global_backward_step(y_predicted_record, y_train_batch[n])

                for lyr in self.layers:
                    gradient = lyr.cumulative_gradient / x_train_batch.shape[0] + self.regular_lambda * lyr.weights
                    lyr.weights = lyr.weights - np.multiply(self.lerning_rate, gradient)
                    lyr.BIG_DELTAS = np.asmatrix(np.zeros(lyr.cumulative_gradient.shape))
            self._save_loss(x_train, y_train, x_validation, y_validation)

    def predict(self, x):
        return self._global_forward_step(x)

    def evaluate(self, x, y):
        y_predicted = self._global_forward_step(x)
        loss = self._count_loss(y_predicted, y)

        print("Loss: ", loss)
        print("Accuracy: ", accuracy_score(y, np.round(y_predicted)))

    def plot_loss(self):
        x_axis = range(0, self.epochs)
        fig, ax = plt.subplots()
        ax.plot(x_axis, self.training_history, label='train_loss')
        ax.plot(x_axis, self.validation_history, label='val_loss')
        ax.legend()
        plt.ylabel('MSE')
        plt.xlabel('epoch number')
        plt.title('loss vs epoch number')
        plt.show()

    def plot_confusion_matrix(self, x, y):
        predicted = np.round(model.predict(x_test)).reshape(-1, )
        actual = y_test.reshape(-1, )

        TP = np.count_nonzero(np.multiply(predicted, actual))
        TN = np.count_nonzero(np.multiply(predicted - 1, actual - 1))
        FP = np.count_nonzero(np.multiply(predicted, actual - 1))
        FN = np.count_nonzero(np.multiply(predicted - 1, actual))

        confusion_matrix_dict = {'actual 1': [TP, FN], 'actual 0': [FP, TN]}
        confusion_matrix = pd.DataFrame(data=confusion_matrix_dict, columns=['actual 1', 'actual 0'],
                                        index=['predicted 1', 'predicted 0'])
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)

        print('\nPrecision: {}'.format(precision))
        print('Recall: {}'.format(recall))
        print('F-score: {}'.format(f1))
        print(confusion_matrix)


if __name__ == '__main__':
    dataset = pd.read_csv('creditcard.csv')
    assert not dataset.isnull().values.any()
    dataset = dataset.drop(['Time', 'Amount'], axis=1)
    NUMBER_OF_FEATURES = dataset.shape[1] - 1
    NUMBER_OF_OK_TRANSACTIONS_IN_TRAIN_VALIDATION_DATASET = 400

    # Split dataset on train_and_validation dataset and test dataset
    train_and_validation, test = train_test_split(dataset, test_size=0.3, random_state=0)

    # Convert test data to numpyarray and split them.
    test = test.values
    x_test = test[:, :-1]
    y_test = test[:, -1]

    # Create balanced, under sample train and validation dataset
    fraud_indices = np.array(train_and_validation[train_and_validation.Class == 1].index)
    normal_indices = np.array(train_and_validation[train_and_validation.Class == 0].index)
    random_normal_indices = np.random.choice(normal_indices, NUMBER_OF_OK_TRANSACTIONS_IN_TRAIN_VALIDATION_DATASET,
                                             replace=False)
    random_normal_indices = np.array(random_normal_indices)
    under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])
    under_sample_dataset = dataset.iloc[under_sample_indices, :]
    # Shuffle train and validation dataset
    under_sample_dataset = under_sample_dataset.sample(frac=1)
    # Convert training and validation dataset to numpy array
    under_sample_dataset = under_sample_dataset.values

    train, validation = train_test_split(under_sample_dataset, test_size=0.3, random_state=0)

    x_train = train[:, :-1]
    y_train = train[:, -1]

    x_validation = validation[:, :-1]
    y_validation = validation[:, -1:]

    model = NeuralNetwork(learning_rate=0.005, batch_size=50, epochs=120, loss='mse', regular_lambda=0.005)
    model.add_layer(input_dim=x_train.shape[1], neurons_number=512, activation='sigmoid')
    model.add_layer(input_dim=512, neurons_number=1, activation='sigmoid')

    model.fit(x_train, y_train, x_validation, y_validation)

    print("\nValidation dataset evaluation:")
    model.evaluate(x_validation, y_validation)
    model.plot_loss()

    print("\nTest dataset evaluation:")
    model.evaluate(x_test, y_test)
    model.plot_confusion_matrix(x_test, y_test)
    # print(model.layers[0].weights)
