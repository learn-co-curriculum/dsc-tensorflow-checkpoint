# Neural Network Regularization


```python
# scikit-learn imports
import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.datasets import make_gaussian_quantiles, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# neural network imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import RandomNormal
tf.logging.set_verbosity(tf.logging.ERROR)
```

You're going to train full neural networks on a _small_ set of generated data. It is a binary classification problem in which you need to identify whether a dot will belong to the teal or orange class.


```python
# generate 2d classification dataset
X, y = make_circles(n_samples=450, noise=0.12)
df = pd.DataFrame(dict(x=X[:,0], y=X[:,1], label=y))

# plot the generated dataset
colors = {0:'teal', 1:'orange'}
fig, ax = pyplot.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    if key != 2:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
pyplot.show()
```

In the two cells below, the set of data has been split into a training and testing set and then fit to a neural network with two hidden layers. Run the two cells below to see how well the model performs.


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
```


```python
# CREATE CLASSIFIER
classifier = Sequential()

# add hidden layer
classifier.add(Dense(
    32, 
    activation='relu', 
    input_dim=2,
    kernel_initializer='random_normal',
))

# add hidden layer
classifier.add(Dense(
    32,
    activation='relu', 
    input_dim=2,
    kernel_initializer='random_normal',
))

# add output layer
classifier.add(Dense(
    1, 
    activation='sigmoid',
    kernel_initializer='random_uniform',
))

classifier.compile(optimizer ='adam',loss="binary_crossentropy",metrics =['accuracy'])

# TRAIN
classifier.fit(X_train, y_train, epochs=25, verbose=0, batch_size=10, shuffle=False)
predicted_vals_train = classifier.predict_classes(X_train)
print("Accuracy on training data:")
print(accuracy_score(y_train,predicted_vals_train))

# TEST
predicted_vals_test = classifier.predict_classes(X_test)
print("Accuracy on test data:")
print(accuracy_score(y_test,predicted_vals_test))
```

##### 1) Modify the code below to use L2 regularization


The model appears to be overfitting. To deal with this overfitting, modify the code below to include L2 regularization in the model. 

Hint: these might be helpful

 - [`Dense` layer documentation](https://keras.io/layers/core/)
 - [`regularizers` documentation](https://keras.io/regularizers/)


```python
# CREATE CLASSIFIER
classifier2 = Sequential()

# add hidden layer
classifier2.add(Dense(
    32, 
    activation='relu', 
    input_dim=2,
    kernel_initializer='random_normal'

))

# add hidden layer
classifier2.add(Dense(
    32,
    activation='relu', 
    input_dim=2,
    kernel_initializer='random_normal'

))

# add output layer
classifier2.add(Dense(
    1, 
    activation='sigmoid',
    kernel_initializer='random_uniform',
))

classifier2.compile(optimizer ='adam',loss="binary_crossentropy",metrics =['accuracy'])

# TRAIN
classifier2.fit(X_train, y_train, epochs=25, verbose=0, batch_size=10, shuffle=False)
predicted_vals_train = classifier2.predict_classes(X_train)
print("Accuracy on training data:")
print(accuracy_score(y_train,predicted_vals_train))

# TEST
predicted_vals_test = classifier2.predict_classes(X_test)
print("Accuracy on test data:")
print(accuracy_score(y_test,predicted_vals_test))

```

Did the regularization you performed prevent overfitting?


```python
# Your answer here
```

### 2) Explain how regularization is related to the bias/variance tradeoff within Neural Networks and how it's related to the results you just achieved in the training and test accuracies of the previous models. What does regularization change in the training process (be specific to what is being regularized and how it is regularizing)?



```python
# Your answer here
```

### 3) How might L1  and dropout regularization change a neural network's architecture?


```python
# Your answer here
```
