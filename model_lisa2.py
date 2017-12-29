import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from decimal import *

#do zmiany na funkcje
class ActivationFunctions:
    @staticmethod
    def relu(x):
        return x.clip(min=0)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))


class ActivationDerivatives:
    @staticmethod
    def relu(x):
        x[x > 0] = 1
        return x

    @staticmethod
    def sigmoid(x):
        return np.dot(x, (1-x))


# TODO add regularization
class Layer:
    def __init__(self, nn, input_dim, neurons_number, learning_rate, activation):
        self.activation = activation
        self.learning_rate = learning_rate
        self.output_layer = True
        self.input = np.asmatrix(np.zeros((input_dim + 1, 1))) #[bias = 0, x0 = 0, x1 = 0, ...]
        self.z = np.asmatrix(np.zeros((neurons_number, 1))) #[z0 = 0, z1 = 0, ...]
        self.output = np.asmatrix(np.zeros((neurons_number, 1))) #[a0 = 0, a1 = 0, ...]
        eps = 2.45 / np.sqrt(input_dim+neurons_number) # evaluate perfect epsilon for input & output dimentions
        self.theta = np.asmatrix(np.random.uniform(low=-eps, high=eps, size=( input_dim + 1, neurons_number))) # begin with uniformly randomized theta in range [-eps, +eps]
        #self.deltas = None
        self.delta = np.asmatrix(np.zeros(( input_dim + 1, 1)))
        self.accumulated_gradient = 0
        self.gradient = 0
        self.nn = nn


    def forward_step(self, input):  # input: array of neurons
        self.input = np.concatenate([[[1]], input])  # add bias neuron
        self.z = np.matmul(self.theta.transpose(), self.input)
        self.output = self.activate(self.z) # theta' * X and activate in one step
        return self.output

    def activate(self, matrix): #matrix: matrix of neurons to activate
        if self.activation == 'relu':
            return ActivationFunctions.relu(matrix)
        elif self.activation == 'sigmoid':
            return ActivationFunctions.sigmoid(matrix)

    def cost_of_theta(self, input, th):  # input: array of neurons
        self.input = np.concatenate([[[1]], input])  # add bias neuron
        self.z = np.matmul(self.theta.transpose(), self.input)
        self.output = self.activate(self.z) # theta' * X and activate in one step
        return self.output
"""
    def backward_step(self, next_theta=None, next_deltas=None, output_delta=None, h_batch=None):
        if self.output_layer:
            self.deltas = output_delta
        else:
            self.deltas = np.dot(next_theta.transposed() @ next_deltas, ActivationDerivatives.sigmoid(self.z))
            #output_delta = next_theta'*next_deltas .* activationdevariatives.sigmoid
            #self.set_deltas(next_theta, next_deltas, h_batch)
            self.deltas = self.deltas[1:]



        # Gradient Descent
        #self.theta = self.theta - np.multiply(self.learning_rate, gradient)

    def set_deltas(self, next_theta, next_deltas, h_batch):
        self.deltas = np.multiply(np.delete(next_theta, 0, 0), (next_deltas @ np.sum(ActivationDerivatives.relu(h_batch), axis=0)))  # Exclude bias row from theta

    def set_deltas_output_layer(self, output_delta):
        self.deltas = output_delta

    def get_gradient(self):
        return np.matmul(self.input,self.deltas.transpose())
"""
class NeuralNetwork:
    def __init__(self, learning_rate, batch_size, epochs, loss):
        self.layers = []
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_of_epochs = epochs
        self.loss_func = loss
        self.loss_over_steps = []

    def add_layer(self, input_dim, neurons_number, activation):
        layer = Layer(nn=self, input_dim=input_dim, neurons_number=neurons_number, learning_rate=self.learning_rate, activation=activation)
        # auto switch output layer
        if self.layers:
            self.layers[-1].output_layer = False
        self.layers.append(layer)

    def forward_step(self, X_batch):
        h_batch = []
        for X_record in X_batch:
            X_record = np.reshape(X_record, (X_record.shape[0], 1))
            for layer in self.layers:
                X_record = layer.forward_step(X_record)
            h_batch.append(X_record)

        return np.concatenate(h_batch)

    def backward_step(self, X, h, y):
        #output_delta = self.count_output_delta(h, y)

        output_delta = h-y
        self.layers[-1].delta = output_delta
        """
        # przez brak wartwy zerowej Delta i delta sa na tym samym poziomie, a nie powinny
        #TODO check if delta[0] isnt going to gradient
        for id, layer in reversed(list(enumerate(self.layers))):
            if id == 0:
                continue
            if id != len(self.layers):
                layer.delta = np.dot(np.matmul(layer.theta[1:,:], self.layers[id-1].delta), np.dot(layer.output, 1-layer.output))
                layer.delta=layer.delta[1:]
            layer.accumulated_gradient += np.matmul(self.layers[id-1].delta, layer.output.transpose())

            #layer.backward_step(next_theta=next_theta, next_deltas=next_deltas, h_batch=h)
        """
        #delta2
        self.layers[-2].delta = (self.layers[-1].theta[1:,:]*self.layers[-1].delta).transpose() * self.layers[-2].output * 1-self.layers[-2].output
        #Delta2

        self.layers[-1].accumulated_gradient += np.matmul(self.layers[-1].delta, np.concatenate([[[1.0]], self.layers[-2].output] ).transpose())

        self.layers[-2].accumulated_gradient += np.matmul(self.layers[-2].delta, np.concatenate([[[1.0]], X.transpose()]).transpose())

    def count_output_delta(self, h, y):
        #output_delta = self.count_loss_vector(h_batch,y_batch)
        output_delta = h-y
        return output_delta

    def count_loss_vector(self, h, y):
        if self.loss_func== 'mse':
            return np.sum(0.5 * (np.square(h - y)))
        # \/ ???
        #return np.sum(np.multiply((h_batch - y_batch), ActivationDerivatives.sigmoid(h_batch)), axis=0)
        elif self.loss_func == 'logloss':
            print(y,h)
            loss = -(np.multiply(y,np.log(h)) + np.multiply((1-y),np.log(1-h)))
            #loss1 = log_loss(y_true=y_batch, y_pred=h_batch, eps=1e-15, normalize=True, sample_weight=None, labels=[0,1])
            #loss2 = np.asarray([[-np.average(np.multiply(y_batch,np.log(h_batch)) + np.multiply((1-y_batch),np.log(1-h_batch)))]])
            return loss

    def fit(self, X, y):
        for i in range(self.num_of_epochs):
            print(i)
            for id,layer in reversed(list(enumerate(self.layers))):
                layer.accumulated_gradient = np.asmatrix(np.zeros((layer.output.shape[0],self.layers[id].delta.shape[0])))
                #if id < len(self.layers)-1: #exclude bias
                #    layer.accumulated_gradient = layer.accumulated_gradient[1:]
                print(id, "grad.shape:", layer.accumulated_gradient.shape)
            for idx in range(0, X.shape[0], self.batch_size):
                X_batch = X[idx:idx+self.batch_size]
                y_batch = y[idx:idx+self.batch_size]

                h_batch = self.predict(X_batch)
                print("fit: h", h_batch)

                # calculate and save COST ( = loss)
                J_avg = np.average(self.count_loss_vector(h_batch, y_batch))
                self.loss_over_steps.append(J_avg)
                #self.show_loss(X, y)
                self.backward_step(X_batch, h_batch, y_batch)



            #gradient checking
            for layer in self.layers:
                print("m: ", X.shape[0])
                layer.gradient = layer.accumulated_gradient / X.shape[0]
            self.gradient_checking(X, y)

        x = np.arange(len(self.loss_over_steps))
        y = self.loss_over_steps
        plt.plot(x,y,'ro')
        plt.title('plot loss over epochs')
        #plt.axis([0, len(self.loss_over_steps), 0, max(self.loss_over_steps)])
        plt.show()


    def predict(self, input):
        return self.forward_step(input)

    def show_loss(self, X, y):
        h = self.forward_step(X)
        loss = np.average(self.count_loss_vector(h, y))

        h_rounded = np.round(h)

        from sklearn.metrics import accuracy_score

        print("loss: ", loss)
        print("Accuracy: ", accuracy_score(y, h_rounded))

    def gradient_checking(self, X, y):
        eps = 0.000001
        gradApprox = []
        for i, layer in enumerate(self.layers):
            print(i, layer.theta)

        #flattening thetas #TODO: make it pretttier
        thetas_raw = [layer.theta.flatten().tolist() for layer in self.layers]
        thetas_semiraw = [item for sublist in thetas_raw for item in sublist]
        thetas = [item for sublist in thetas_semiraw for item in sublist]

        print(thetas)

        for i in range(len(thetas)):
            thetasPlus = list(thetas)
            thetasPlus[i] += eps
            thetasMinus = list(thetas)
            thetasMinus[i] -= eps
            gradApprox.append((self.cost_of_theta(thetasPlus, X, y) - self.cost_of_theta(thetasMinus, X, y))/(2*eps))
        print("approx:", len(gradApprox), gradApprox)
        for layer in self.layers:
            print("grad:", layer.gradient)

    def cost_of_theta(self, th, X, y):
        #TODO: generalize it (pack th into ndarrays)
        print("cost of theta")
        theta1 = np.matrix(th[:15]).reshape((3,5))
        theta2 = np.matrix(th[15:]).reshape((6,1))
        self.layers[0].theta = theta1
        self.layers[1].theta = theta2

        for i in range(self.num_of_epochs):
            for idx in range(0, X.shape[0], self.batch_size):
                X_batch = X[idx:idx+self.batch_size]
                y_batch = y[idx:idx+self.batch_size]

                h_batch = self.predict(X_batch)
                # calculate and save COST ( = loss)
                J_avg = np.average(self.count_loss_vector(h_batch, y_batch))
        return J_avg

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

    # logical AND
    x_train = np.asarray([[0,0], [0,1], [1,0], [1,1]])
    y_train = np.asarray([[0], [0], [0], [1]])

    model = NeuralNetwork(learning_rate=0.0001, batch_size=1, epochs=1, loss='logloss')
    model.add_layer(input_dim=x_train.shape[1], neurons_number=5, activation='sigmoid')
    model.add_layer(input_dim=5, neurons_number=1, activation='sigmoid')


    w_begin = model.layers[0].theta
    print("x_train:",x_train, "y_train:",y_train)
    model.fit(x_train, y_train)
    #model.evaluate()
    print("w_begin ", w_begin)
    print("w_end", model.layers[0].theta)

    #print(model.layers[1].theta)
    print(model.forward_step(x_train))