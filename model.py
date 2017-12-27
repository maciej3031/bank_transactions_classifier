import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class ActivationFunctions:
    @staticmethod
    def relu(x):
        return x.clip(min=0)

    @staticmethod
    def sigmoid(x):
        return np.multiply(.5, (1 + np.tanh(np.multiply(.5, x))))


class ActivationDerivatives:
    @staticmethod
    def relu(x):
        x[x > 0] = 1
        return x

    @staticmethod
    def sigmoid(x):
        return np.multiply(x, (1-x))


# TODO add regularization
class Layer:
    def __init__(self, input_dim, neurons_number, learning_rate, activation, output_layer):
        self.activation = activation
        self.learning_rate = learning_rate

        self.output_layer = output_layer

        self.input = np.asmatrix(np.zeros((input_dim + 1, 1)))
        self.output = np.asmatrix(np.zeros((neurons_number, 1)))
        self.weights = np.asmatrix(np.random.uniform(low=-0.1, high=0.1, size=( input_dim + 1, neurons_number)))
        self.deltas = None

    def forward_prop(self, input):
        self.set_input(input)
        self.set_output()
        return self.output

    def set_input(self, input):
        self.input = np.concatenate([[[1]], input])  # Add bias

    def set_output(self):
        self.output = self.activate(np.matmul(self.weights.transpose(), self.input))

    def activate(self, x):
        """ x must be numpy matrix """
        if self.activation == 'relu':
            return ActivationFunctions.relu(x)
        elif self.activation == 'sigmoid':
            return ActivationFunctions.sigmoid(x)

    def back_prop(self, next_weights=None, next_deltas=None, output_delta=None, y_predicted_batch=None):
        if self.output_layer:
            self.set_deltas_output_layer(output_delta)
        else:
            self.set_deltas(next_weights, next_deltas, y_predicted_batch)

        gradient = self.get_gradient()

        # Gradient Descent
        self.weights = self.weights - np.multiply(self.learning_rate, gradient)

    def set_deltas(self, next_weights, next_deltas, y_predicted_batch):
        self.deltas = np.multiply(np.matmul(np.delete(next_weights, 0, 0), next_deltas), np.sum(ActivationDerivatives.relu(y_predicted_batch), axis=0))  # Exclude bias row from weights

    def set_deltas_output_layer(self, output_delta):
        self.deltas = output_delta

    def get_gradient(self):
        return np.matmul(self.input, self.deltas.transpose())

    def get_deltas(self):
        return self.deltas

    def get_weights(self):
        return self.weights


# TODO add logloss loss function
class NeuralNetwork:
    def __init__(self, learning_rate, batch_size=50, epochs=20, loss='mse'):
        self.layers = []
        self.lerning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss = loss

    def add_layer(self, input_dim, neurons_number, activation='sigmoid', output_layer=False):
        layer = Layer(input_dim, neurons_number, learning_rate=self.lerning_rate, activation=activation, output_layer=output_layer)
        self.layers.append(layer)

    def forward_prop(self, x_train_batch):
        y_predicted_batch = []
        for x_train_record in x_train_batch:
            x_train_record = np.reshape(x_train_record, (x_train_record.shape[0], 1))
            for layer in self.layers:
                x_train_record = layer.forward_prop(x_train_record)
            y_predicted_batch.append(x_train_record)

        return np.concatenate(y_predicted_batch)

    def back_prop(self, y_predicted_batch, y_train_batch):

        output_delta = self.count_output_delta(y_predicted_batch, y_train_batch)

        output_layer = self.layers[-1]
        output_layer.back_prop(output_delta=output_delta, y_predicted_batch=y_predicted_batch)

        next_deltas = output_layer.get_deltas()
        next_weights = output_layer.get_weights()

        for layer in reversed(self.layers[:-1]):
            layer.back_prop(next_weights=next_weights, next_deltas=next_deltas, y_predicted_batch=y_predicted_batch)

    def count_output_delta(self, y_predicted_batch, y_train_batch):
        if self.loss == 'mse':
            return np.sum(np.multiply((y_predicted_batch - y_train_batch), ActivationDerivatives.sigmoid(y_predicted_batch)), axis=0)
        elif self.loss == 'logloss':
            return np.sum((y_predicted_batch - y_train_batch) / (np.finfo(float).eps + y_predicted_batch - np.square(y_predicted_batch)), axis=0)

    def count_loss(self, y_predicted, y_train):
        if self.loss == 'mse':
            return np.sum(0.5 * (np.square(y_predicted - y_train)))
        elif self.loss == 'logloss':
            # from sklearn.metrics import log_loss
            # return - np.sum(y_train*(np.log(y_predicted+np.finfo(float).eps)) + (1-y_train)*np.log(1+np.finfo(float).eps-y_predicted))
            # return log_loss(y_train, y_predicted, eps=1e-5)
            cost = -np.sum(np.multiply(y_train,np.log(y_predicted)) + np.multiply((1-y_train),np.log(1-y_predicted)))
            return (cost / y_train.shape[0])

    def fit(self, x_train, y_train):
        for i in range(self.epochs):
            for idx in range(0, x_train.shape[0], self.batch_size):
                x_train_batch = x_train[idx:idx+self.batch_size]
                y_train_batch = y_train[idx:idx+self.batch_size]

                y_predicted_batch = self.forward_prop(x_train_batch)
                print(idx)
                self.show_loss(x_train, y_train)
                self.back_prop(y_predicted_batch, y_train_batch)


    def show_loss(self, x_train, y_train):
        y_predicted = self.forward_prop(x_train)
        loss = self.count_loss(y_predicted, y_train)

        y_predicted_labels = np.round(y_predicted)

        from sklearn.metrics import accuracy_score

        print("Loss: ", loss)
        print("Accuracy: ", accuracy_score(y_train, y_predicted_labels))

    def evaluate(self):
        pass

    def predict(self, x_train):
        return self.forward_prop(x_train)


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
    print(fraud_indices, normal_indices)
    random_normal_indices = np.random.choice(normal_indices, NUMBER_OF_OK_TRANSACTIONS_IN_TRAIN_VALIDATION_DATASET, replace=False)
    random_normal_indices = np.array(random_normal_indices)
    print(random_normal_indices)
    under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])
    under_sample_dataset = dataset.iloc[under_sample_indices, :]
    print(under_sample_dataset.size)
    # Shuffle train and validation dataset
    under_sample_dataset = under_sample_dataset.sample(frac=1)
    print(under_sample_dataset.size)
    # Convert training and validation dataset to numpy array
    under_sample_dataset = under_sample_dataset.values

    train, validation = train_test_split(under_sample_dataset, test_size=0.3, random_state=0)

    x_train = train[:, :-1]
    y_train = train[:, -1]

    x_validation = validation[:, :-1]
    y_validation = validation[:, -1:]

    model = NeuralNetwork(learning_rate=0.00002, batch_size=1, epochs=2, loss='logloss')
    model.add_layer(input_dim=x_train.shape[1], neurons_number=1024, activation='sigmoid', output_layer=False)
    model.add_layer(input_dim=1024, neurons_number=1, activation='sigmoid', output_layer=True)

    #x_train = np.asarray([[0,0], [0,1], [1,0], [1,1]])
    #y_train = np.asarray([[0], [0], [0], [1]])

    model.fit(x_train, y_train)
    model.evaluate()
    print(model.layers[0].weights)
    print(model.predict(x_train))