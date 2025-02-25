# :one: :two: :three: MNIST
MNIST is a beginner-friendly project that focuses on introducing students to machine learning and artificial intelligence. Students are tasked with implementing the algorithms from scratch in Python and NumPy without the help of frameworks. The project is split in different sections that tentatively span an entire semester at McGill (Jan-May). The sections are as follows:

- Linear Regression
- Logistic & Softmax Regression
- Neural Networks and Convolutional Neural Networks (+ Generalization, Regularization, and Miscellaneous)
- SVM, Kernel Methods, Random Forests, Gradient Boosting (+ Optimizer Algorithms)
- Unsupervised Learning (K-Means Clustering and Autoencoders)

A PDF document is given to provide the necessary theory to complete each section. (see WIP document [here](https://drive.google.com/drive/folders/1LEVz3-lgaS-LHKwZ-UXADz4q46_xsVBF?usp=sharing))

In the context of the McGill AI Lab, this project is helping McGill students bridge their first contact with ML. This is done through regular assignments to be submitted as Jupyter Notebooks, from which the "best" submissions will be chosen to be displayed in this repository.

# Section 0: Linear Regression
## :book: Theory
Linear regression is a staple of machine learning. Being one of the most popular algorithms even nowadays, it is important that one must begin with a foundational knowledge of this before moving any further. The task of linear regression is to fit a straight line ($\hat{y} = \vec{\theta}^\top \cdot \vec{x}$) to data points so as to minimize the *Sum of Squared Errors* (SSE) $\frac{1}{2} \sum_{i=1}^{n} {(y^{(i)} - \hat{y}^{(i)})^2}$. The derived update formulae for the 2D linear regression are:

$\theta_0 = \theta_0 - \alpha \sum_{i=1}^n {(\hat{y}^{(i)} - y^{(i)})}$

$\theta_1 = \theta_1 - \beta \sum_{i=1}^n {(\hat{y}^{(i)} - y^{(i)})x^{(i)}}$

## :computer: Implementation
In a Jupyter notebook, start by importing the necessary libraries:
```python
import numpy as np
import pandas as pd
import random
```

Then, parse the *housing.csv* dataset from [here](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset).
```python
housing = pd.DataFrame(pd.read_csv("Housing.csv"))

listOfExamples = list(zip((housing.area).astype(int) / 1000, (housing.price).astype(int) / 1000000))
```

Then, initialize the parameters:
```python
# Initialize parameters.
parameters = {"theta0": 0, "theta1": 0}
```

Then, define relevant functions:
```python
# defining the model
def model(parameters, x):
  return parameters["theta0"] + parameters["theta1"] * x

# definining the functions to compute the derivatives
def derivative0(parameters, x, y):
  return y - model(parameters, x)

def derivative1(parameters, x, y):
    return (y - model(parameters, x)) * x;


# Defining loss. Output loss as normalized by how many data points we have
def loss(parameters, listOfExamples):
  loss = 0;
  for x, y in listOfExamples:
    loss += (y - model(parameters, x)) ** 2

  return loss / len(listOfExamples)
```

Then, train!
```python
# hyperparameters
a = 0.01
b = 0.01
epochs = 1000

# shuffling list
random.shuffle(listOfExamples)

# defining the training - using stochastic GD
def train(parameters, listOfExamples):
  for i in range(epochs):
    for x, y in listOfExamples:
      parameters["theta0"] = parameters["theta0"] + a * derivative0(parameters, x, y) / (2**(epochs // 250))
      parameters["theta1"] = parameters["theta1"] + b * derivative1(parameters, x, y) / (2**(epochs // 250))

    if i % 100 == 0:
      print(loss(parameters, listOfExamples))

    if i % 10 == 0:
      pastParams.append((parameters["theta0"], parameters["theta1"]))

train(parameters, listOfExamples)
```

And that's about it! This is the first step into the world of machine learning! Below are the results obtained from an example run. Dashed line is the theoretical best line.

| Line fit to dataset | Parameter progression |
| ------------------- | --------------------- |
| ![regression](https://github.com/user-attachments/assets/e18e1cf3-e276-4b07-8830-6821af9e0a55) | ![regression_parameters](https://github.com/user-attachments/assets/6d13d384-e379-4830-af96-9c902c620162) |


# Section 1: Logistic & Softmax Regression
## Logistic Regression
### :book: Theory
Logistic regression is an extension of linear regression. Unlike it, however, the task of logistic regression is to classify between two binary options - often "yes" and "no", or 0 and 1. Here, a linear regression model is sent through a logistic function $\sigma (x) = \frac{1}{1 + e^{-x}}$, which *bends* the line into an 'S' shape. From there, a decision boundary of 0.5 is chosen, which allows us to classify the input into either classes. This algorithm has a similar update rule and way of quantifying its performance (i.e. SSE). Finally, the derived update rule for the parameter vector $\vec{\theta}$ ends up being:

$\vec{\theta} = \vec{\theta} + \alpha (y - \sigma(\vec{\theta}^\top \cdot \vec{x}))\vec{x}$

Which you can see is very similar to that of linear regression.

The logic that is used for this project, however, is that 10 different logistic regression models will be trained, one for each digit. This means that a prediction is now being made based off of the model with the highest probability that the digit it is presented is the right one.

### :computer: Implementation
Start, as usual, by importing the necessary libraries:
```python
import numpy as np
import keras
```

Then, import the MNIST dataset (normalize it too):
```python
dataset = keras.datasets.mnist.load_data(path="mnist.npz")
(x_train, y_train), (x_test, y_test) = dataset

x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)
```

Then, define the necessary functions:
```python
class LogisticRegression:
    def __init__(self, lr=0.01, epochs=1000):
      # Initialize required fields
       # lr is the learning rate, or the amount by which we will nudge each
       # parameter every iteration.
       # Epochs are the number of iterations.
       # Weights and bias are the vectors which we will update as we train.
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def train(self, x, y):
        # Initialize parameters
        # n_samples is the number of data samples i.e images.
        # Features are the labels
        n_samples, n_features = x.shape
        # All weights and bias are initially 0.
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent
        for _ in range(self.epochs):
            # Linear model
            linear_model = np.dot(x, self.weights) + self.bias
            # Sigmoid function
            y_predicted = self.sigmoid(linear_model)

            # Derivatives
            dw = (1 / n_samples) * np.dot(x.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update weights by the learning rate x the respective derivative
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, x):
      # Returns wether the model thinks the image IS the number (1) or not (0)
      # Model must be trained for this to work, or else all params are 0.
        linear_model = np.dot(x, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_class = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_class)

    def sigmoid(self, x):
      # Returns output of sigmoid function given input x.
        return 1 / (1 + np.exp(-x))
```

Then, train:
```python
def create_binary_labels(y, digit):
  # Returns wether the output matches the digit.
    return (y == digit).astype(int)

# Train 10 models
# Initialize a list, where each index will contain a model.
# Model for digit 0 is index 0, digit 1 is index 1, so on.
models = []
for digit in range(10):
    print(f"Training model for digit {digit}...")
    # Create comparator
    y_train_binary = create_binary_labels(y_train, digit)
    # Initialize model (call constructor)
    model = LogisticRegression(lr=0.01, epochs=250)
    # Train
    model.train(x_train, y_train_binary)

    models.append(model)
```

Finally, obtain the accuracy of the overall algorithm:
```python
# Predict function for multi-class classification
# Basically we get the probabilities of each model and return the highest.
def predict_multi(models, x):
    # Get probabilities for all models
    probabilities = [model.sigmoid(np.dot(x, model.weights) + model.bias) for model in models]
    # Convert to numpy array for easier handling
    probabilities = np.array(probabilities)
    # Find the digit with the highest probability for each sample
    predictions = np.argmax(probabilities, axis=0)
    return predictions

def accuracy_score(y_true, y_pred):
  # Returns the accuracy of the models
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    return correct / total

# Evaluate on test data
y_pred = predict_multi(models, x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")
```

With an accuracy of 82% in an example run, we can observe that this simple linear algorithm works remarkably well in this relatively complicated task.
Here's the SSE values calculated at every 100 training iteration:

![logistic](https://github.com/user-attachments/assets/29a3aa7e-c9ac-41d5-9bc1-9f5fbef2acf2)

## Softmax Regression
### :book: Theory
Softmax regression is a generalization of logistic regression. It involves the softmax function: $softmax_i(\vec{x}) = \frac{e^{x_i}}{\sum_{j=1}^n {e^{x_j}}}$, which is a generalization of the logistic function to multiple dimensions. The rationale here is that since softmax already outputs a vector, we can directly use that vector as a probability distribution of the model's confidence in its answer for a given input. This idea actually bridges what's done in neural networks to softmax regression, as NNs commonly use the softmax function to produce its output. In our case, however, softmax regression is also a linear algorithm, so it should not be a surprise that the update rule is also very similar to the two previous algorithms: $\vec{\theta} = \vec{\theta} + \alpha (y - \sigma(\vec{\theta}^\top \cdot \vec{x}))\vec{x}$.

### :computer: Implementation
Start as usual with the imports:
```python
import numpy as np
import matplotlib.pyplot as plt
import keras
import random
```

Then, import the dataset:
```python
# import the mnist dataset
keras.datasets.mnist.load_data(path="mnist.npz")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# normalize the pixel values and flatten it
x_train = (x_train / 255).reshape(len(x_train), 784)
x_test = (x_test / 255).reshape(len(x_test), 784)

# One-hot encode the labels
def one_hot(label):
    one_hot_vector = np.zeros(10)
    one_hot_vector[label] = 1
    return one_hot_vector

y_train = np.array([one_hot(label) for label in y_train])
y_test  = np.array([one_hot(label) for label in y_test])

# pair the label and image for training
listOfTraining = list(zip(y_train, x_train))
listOfTesting = list(zip(y_test, x_test))
```

Then, define the model:
```python
# initalize the weight matrix for each digit
digit_theta={}

for i in range (10):
    digit_theta[i] = np.random.normal(0.0, 0.01, 784)
```

Then, define the necessary functions:
```python
def softmax(logits):
    # we subtract the maximum to avoid computational challenges or overflow eg: exponenet very large
    shift = logits - np.max(logits)
    exps = np.exp(shift)
    return exps / np.sum(exps)

def gradient(x, yPred, yTrue):
    gradients = {}
    for d in range(10):
        gradients[d] = (yPred[d]- yTrue[d]) * x 
    return gradients
    
def cross_entropy(yPred, yTrue):
    # since label 0*log((yPred) makes no change and may cause Nan
    # we can just find the true index
    trueClassIndex = np.where(yTrue == 1)  
    return -1 * np.log(yPred[trueClassIndex])  

def model(parameters, x):
    logits = np.array([np.dot(parameters[d], x) for d in range(10)])
    return softmax(logits)

def getTestAcc(parameters, listOfTesting):
  totalWrongs = 0
  for label, x in listOfTesting:
    yPred = model(parameters, x)
    if(np.argmax(yPred) != np.where(label == 1)):
      totalWrongs += 1
  return 1 - (totalWrongs / len(listOfTesting))

def train(listOfTraining, parameters, learningRate, epochs):
    training_log = []
    testing_log  = []
    for epoch in range(epochs):
        epoch_loss = 0
        
        for label, x in listOfTraining:
            # predict
            yPred = model(parameters, x)

            # Compute loss
            loss = cross_entropy(yPred, label)
            epoch_loss += loss

            # compute and update gradients
            gradients = gradient(x, yPred, label)
            for d in range(10):
                parameters[d] -= learningRate * gradients[d]/(epoch+1)

        training_log.append((epoch + 1, epoch_loss/len(listOfTraining)))
        testing_log.append((epoch + 1, getTestAcc(parameters, listOfTesting)))
        print("Epoch:", epoch + 1, "Loss:", epoch_loss/len(listOfTraining))

        random.shuffle(listOfTraining) 
    return (training_log, testing_log)
```

And finally, train!
```python
training_log, testing_log = train(listOfTraining, digit_theta, 0.5, 80)
```

And that's it! Here is a loss curse associated to this example run, which yielded an accuracy of about 91%:

![softmax](https://github.com/user-attachments/assets/a88e8eeb-454b-45e4-8bcc-3d6cd61e7929)

# Section 2: NNs and CNNs
## Neural Networks
### :book: Theory
Neural networks are building blocks to so many modern AI solutions. It is therefore very important to understand them, as it opens doors to LLMs, RL, etc. The very core of neural networks is the perceptron, a linear algorithm that is simply a dot product between its inputs and a weight vector. Additionally, a bias value may be added. Then, the result is sent through an activation function, which introduces nonlinearity into the overall algorithm. Putting a couple of these perceptrons together yields a neural network! A neural network functions in 2 stages: forward and backward passes. The forward pass is responsible for passing the input through the network, through a bunch of dot products. Thereafter, the model is updated by backpropagating the output through and using an algorithm known as backpropagation to effectively train the weights and biases. The backpropagation algorithm is out of the scope of this README (refer to the WIP document). Finally, predictions are made by simply passing foward new inputs and looking at the output layer.

### :computer: Implementation (OOP version)
Start as usual:
```python
from numpy.typing import NDArray
from typing import Dict
import numpy as np
import keras
import matplotlib.pyplot as plt
from tqdm import tqdm
```

Import the data:
```python
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(path="mnist.npz")

x_train = np.reshape(x_train, (x_train.shape[0], -1))
m = x_train.shape[0]

x_test = np.reshape(x_test, (x_test.shape[0], -1))
m_t = x_test.shape[0]

def one_hot_encoding(size, Y):
    ny = np.zeros(size)
    for i in range(size[0]):
        ny[i, Y[i]] = 1
    return ny

y_train_encoded = (one_hot_encoding((m, 10), y_train)).T
y_test = (y_test.reshape(1, -1))
x_train = x_train.T
x_test = x_test.T
```

The main code body:
```python
class Scalar:

    @staticmethod
    def z_score(data: NDArray[np.float64]) -> NDArray[np.float64]:
        mean: int = np.mean(data, axis=0)
        std: int = np.std(data, axis=0)
        std[std == 0] = 1
        return (data - mean) / std

class Layer:
    __supported_activations: list[str] = ["linear","relu","softmax"]

    def __init__(self, size: int, activation: str = "linear") -> None:
        if activation not in Layer.__supported_activations:
            raise ValueError(f"Parameter 'activation' not supported, please enter one of the following: {', '.join(Layer.__supported_activations)}")
        self.__size: int = size
        self.__activation: str = activation
        self.__parent: Layer = None
        self.__child: Layer = None
        self.__W: NDArray[np.float64] = None
        self.__B: NDArray[np.float64] = None
        self.__A: NDArray[np.float64] = None
        self.__dZ: NDArray[np.float64] = None

    def __str__(self) -> str:
        return f"ID: {self.__id} | Neurons: {self.__size} | Activation: {self.__activation}"

    def set_parent(self, parent: "Layer") -> None:
        self.__parent = parent

    def get_parent(self) -> "Layer":
        return self.__parent

    def set_child(self, child: "Layer") -> None:
        self.__child = child

    def get_child(self) -> "Layer":
        return self.__child

    def set_id(self, id: int) -> None:
        self.__id = id

    def get_id(self) -> int:
        return self.__id

    def get_size(self) -> int:
        return self.__size

    def get_activation(self) -> str:
        return self.__activation

    def get_dz(self) -> NDArray[np.float64]:
        return self.__dZ

    def set_dz(self, dZ: NDArray[np.float64]) -> None:
        self.__dZ: NDArray[np.float64] = dZ

    # New layers
    def set_neurons(self) -> None:
        if self.__parent:
            parent_size: int = self.__parent.get_size()
            self.__W: NDArray[np.float64] = np.random.normal(0, 0.05, size=(self.__size, parent_size))
            self.__B: NDArray[np.float64] = np.zeros((self.__size, 1))
            self.__A: NDArray[np.float64] = np.zeros((self.__size, 1))

    def get_value(self) -> NDArray[np.float64]:
        return self.__A

    def get_parameters(self) -> list[NDArray[np.float64]]:
        return self.__W, self.__B

    def set_parameters(self, W: NDArray[np.float64], B: NDArray[np.float64]) -> None:
        self.__W = W
        self.__B = B

    def update_parameters(self, learning_rate: float, dW: NDArray[np.float64], dB: NDArray[np.float64]):
        self.__W -= learning_rate * dW
        self.__B -= learning_rate * dB

    def set_input(self, input: NDArray[np.float64]) -> None:
        self.__A  = input

    def activation_function(self, Z: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.__activation == "relu":
            return np.maximum(Z, 0)
        elif self.__activation == "softmax":
            exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
        else:
            return Z

    def forward(self) -> None:
        if self.__parent:
            Z = np.matmul(self.__W, self.__parent.get_value()) + self.__B
            self.__A = self.activation_function(Z)
class Loss:
    __supported_losses: list[str] = ["CCEL"]
    __supported_regularizations: list[str] = ["none", "L2"]

    def __init__(self, loss: str = "CCEL", learning_rate: float = 1e-4, regularization: str = "none", lambda_: float = 0.01, batch_size: int = 32) -> None:
        if loss not in Loss.__supported_losses:
            raise ValueError(f"Parameter 'loss' not supported, please enter one of the following: {', '.join(Loss.__supported_losses)}")
        if regularization not in Loss.__supported_regularizations:
            raise ValueError(f"Parameter 'regularization' not supported, please enter one of the following: {', '.join(Loss.__supported_regularizations)}")

        self.__loss: str = loss
        self.__learning_rate: float = learning_rate
        self.__regularization: str = regularization
        self.__lambda: float = lambda_
        self.__batch_size: int = batch_size

    def get_loss(self) -> str:
        return self.__loss

    def get_learning_rate(self) -> float:
        return self.__learning_rate

    def get_regularization(self) -> str:
        return self.__regularization

    def get_lambda(self) -> float:
        return self.__lambda

    def get_batch_size(self) -> int:
        return self.__batch_size

    def loss_computation(self, layers: list[Layer], y_batch: NDArray[np.float64]) -> float:
        output_layer: Layer = layers[-1]
        predictions: NDArray[np.float64] = output_layer.get_value()

        # Categorical Cross-Entropy Loss Computation
        if self.__loss == "CCEL":
            batch_size = y_batch.shape[1]
            loss: float = -np.sum(y_batch * np.log(predictions + 1e-9)) / batch_size

        # L2 Regularization
        if self.__regularization == "L2":
            reg_loss = 0.5 * self.__lambda * sum(np.sum(layer.get_parameters()[0] ** 2)
                                                 for layer in layers if layer.get_parameters()[0] is not None)
            loss += reg_loss / batch_size

        return loss
class NeuralNetwork:

    def __init__(self) -> None:
        self.__layers: list[Layer] = []

    def __str__(self) -> str:
        layers_str = "\n".join(f"\t{layer}" for layer in self.__layers)
        return f"Neural Network:\n{layers_str}\nTotal of Parameters: {self.get_num_of_parameters()}"

    def get_num_of_parameters(self) -> int:
        num: int = 0
        for layer in self.__layers:
            W, B = layer.get_parameters()
            if W is not None and B is not None:
                num += W.shape[0] * W.shape[1] + B.shape[0]
        return num

    # Hidden layers can only be linear or ReLU
    def build(self, layers: list[Layer]) -> None:
        self.__layers: list[Layer] = layers
        # Updating layers' relationships
        layers[0].set_id(0)
        for i in range(1, len(layers)):
            layers[i].set_parent(layers[i - 1])
            layers[i - 1].set_child(layers[i])
            layers[i].set_neurons()
            layers[i].set_id(i)

    def fit_data(self, train: tuple[NDArray[np.float64]], test: tuple[NDArray[np.float64]]) -> None:
        input, target = train
        test_input, test_target = test
        # An input with P features and Q examples should have the shape (P,Q)
        # A target should have the shape (K,Q)
        if input.shape[0] != self.__layers[0].get_size():
            raise ValueError(f"At the input layer, supported size is: ({self.__layers[0].get_size()},n?), received: {input.shape}")
        if target.shape[0] != self.__layers[-1].get_size():
            raise ValueError(f"At the target layer, supported size is: ({self.__layers[-1].get_size()},n?), received: {target.shape}")

        self.__input: NDArray[np.float64] = input
        self.__target: NDArray[np.float64] = target
        self.__test_input: NDArray[np.float64] = test_input
        self.__test_target: NDArray[np.float64] = test_target

        # Creating weight matrix W, bias vector B, and value vector A for every layer
        # For a specifc layer containing K neurons, having a parent layer containing N neurons:
        # W shape: (K,N) | B shape: (K,1) | A shape: (K,1)
        for layer in self.__layers:
            layer.set_neurons()

    def forward_propagation(self, x_batch: NDArray[np.float64]) -> None:
        self.__layers[0].set_input(x_batch)
        for layer in self.__layers[1:]:
            layer.forward()

    def back_propagation(self, y_batch: NDArray[np.float64]) -> None:
        if self.__loss.get_loss() == "CCEL":
            # Output layer gradient for CCEL with Softmax
            output_layer = self.__layers[-1]
            output_layer.set_dz(output_layer.get_value() - y_batch)

            # Backpropagate through hidden layers
            for i in range(len(self.__layers) - 2, -1, -1):
                layer: Layer = self.__layers[i]
                child_layer: Layer = self.__layers[i + 1]

                # dA (Gradient of loss with respect to layer's activations)
                dA: NDArray[np.float64] = np.matmul(child_layer.get_parameters()[0].T, child_layer.get_dz())

                if layer.get_activation() == "relu":
                    dZ: NDArray[np.float64] = dA * (layer.get_value() > 0)
                else:
                    dZ: NDArray[np.float64] = dA

                layer.set_dz(dZ)

            # Updating parameters
            learning_rate: float = self.__loss.get_learning_rate()
            for layer in self.__layers[1:]:
                dW: NDArray[np.float64] = np.matmul(layer.get_dz(), layer.get_parent().get_value().T) / y_batch.shape[1]
                dB: NDArray[np.float64] = np.sum(layer.get_dz(), axis=1, keepdims=True) / y_batch.shape[1]

                if self.__loss.get_regularization() == "L2":
                    dW += self.__loss.get_lambda() * layer.get_parameters()[0]

                layer.update_parameters(learning_rate, dW, dB)

    def train(self, epochs: int, loss: Loss = Loss()) -> tuple[list[float]]:
        self.__loss: Loss = loss
        batch_size = self.__loss.get_batch_size()
        loss_hist: list[float] = []
        test_accuracy: list[float] = []
        num_samples = self.__input.shape[1]

        for epoch in tqdm(range(epochs), leave=True, position=0):
            test_accuracy.append(self.test(self.__test_input, self.__test_target))
            indices = np.random.permutation(num_samples)  # Shuffle dataset
            for batch_start in range(0, num_samples, batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                x_batch = self.__input[:, batch_indices]
                y_batch = self.__target[:, batch_indices]
                self.forward_propagation(x_batch)
                loss_hist.append(self.__loss.loss_computation(self.__layers, y_batch))
                self.back_propagation(y_batch)

        return (loss_hist, test_accuracy)

    def get_results(self) -> NDArray[np.float64]:
        return self.__layers[-1].get_value()

    def get_parameters(self) -> Dict[int, list[NDArray[np.float64]]]:
      parameters = {}
      for i, layer in enumerate(self.__layers):
          parameters[i] = layer.get_parameters()
      return parameters

    def set_nn_parameters(self, parameters: Dict[int, list[NDArray[np.float64]]]) -> None:
      for i, (W, B) in parameters.items():
          self.__layers[i].set_parameters(W, B)

    def test(self, test_input: NDArray[np.float64], test_target: NDArray[np.float64]) -> float:
        test_predictions: NDArray[np.int64] = self.predict(test_input)  # Shape: (1, m)
        if test_target.shape[0] > 1:  # If shape is (10, m), convert it
            test_target = np.argmax(test_target, axis=0).reshape(1, -1)  # Shape: (1, m)
        correct_predictions: int = np.sum(test_predictions == test_target)
        total_samples: int = test_target.shape[1]
        accuracy: float = correct_predictions / total_samples
        return accuracy


    def predict(self, input: NDArray[np.float64]) -> NDArray[np.int64]:
        self.__layers[0].set_input(input)
        for layer in self.__layers[1:]:
            layer.forward()
        probabilities: NDArray[np.float64] = self.get_results()
        predictions: NDArray[np.int64] = np.argmax(probabilities, axis=0).reshape(1, -1)
        return predictions
```

Defining the neural network:
```python
model = NeuralNetwork()
model.build([
    Layer(784),
    Layer(128, "relu"),
    Layer(128, "relu"),
    Layer(10, "softmax")
])

scaled_x = Scalar().z_score(x_train)
scaled_x_test = Scalar().z_score(x_test)
model.fit_data((scaled_x, y_train_encoded), (scaled_x_test, y_test))

# Uncomment to load weights
# with open('nn_parameters.pkl', 'rb') as f:
#     parameters: Dict[int, list[NDArray[np.float64]]] = pickle.load(f)
# model.set_nn_parameters(parameters)
```

Training the NN:
```python
epochs = 100
loss_hist = []
test_accuracy_hist = []

# Uncomment to load histories
# with open('nn_hist.pkl', 'rb') as f:
#     imported_loss_hist, imported_test_accuracy_hist = pickle.load(f)
# loss_hist += imported_loss_hist
# test_accuracy_hist += imported_test_accuracy_hist


epochs_loss_hist, epochs_test_accuracy_hist = model.train(epochs, Loss(learning_rate=5e-2, regularization="L2", lambda_=1e-3, batch_size=500))

loss_hist += epochs_loss_hist
test_accuracy_hist += epochs_test_accuracy_hist
```

With an accuracy of 98% for an example run, we can see that neural networks are incredibly powerful solutions that can be used for many tasks. The loss plots are shown below:

![NN](https://github.com/user-attachments/assets/71ce7e41-ecd3-4b21-88aa-bf00f932bd21)
