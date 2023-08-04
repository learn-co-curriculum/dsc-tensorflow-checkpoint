# TensorFlow Checkpoint

This assessment covers building and training a `tf.keras` `Sequential` model, then applying regularization.  The dataset comes from a ["don't overfit" Kaggle competition](https://www.kaggle.com/c/dont-overfit-ii).  There are 300 features labeled 0-299, and a binary target called "target".  There are only 250 records total, meaning this is a very small dataset to be used with a neural network. 

_You can assume that the dataset has already been scaled._

N.B. You may get comments from keras/ternsorflow regarding your kernel and runtime. These are completely normal and are informative comments, rather than warnings.


```python
# Run this cell without changes

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
```


```python
# __SOLUTION__

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
```

## 1) Prepare Data for Modeling

* Using `pandas`, open the file `data.csv` as a DataFrame
* Drop the `"id"` column, since this is a unique identifier and not a feature
* Separate the data into `X` (a DataFrame with all columns except `"target"`) and `y` (a Series with just the `"target"` column)
* The train-test split should work as-is once you create these variables


```python
# Replace None with appropriate code

# Read in the data
df = None

# Drop the "id" column
None

# Separate into X and y
X = None
y = None

# Test/train split (set the random state to 2021) and check the X_Train shape
None
```


```python
# __SOLUTION__

# Read in the data
df = pd.read_csv("data.csv")

# Drop the "id" column
df.drop("id", axis=1, inplace=True)

# Separate into X and y
X = df.drop("target", axis=1)
y = df["target"]

# Test/train split (set the random state to 2021) and check the X_Train shape
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2021)
X_train.shape
```




    (187, 300)




```python
# Run this code block without any changes

# Assert

assert type(df) == pd.DataFrame
assert type(X) == pd.DataFrame
assert type(y) == pd.Series

assert X_train.shape == (187, 300)
assert y_train.shape == (187,)
```


```python
# __SOLUTION__

# Assert

assert type(df) == pd.DataFrame
assert type(X) == pd.DataFrame
assert type(y) == pd.Series

assert X_train.shape == (187, 300)
assert y_train.shape == (187,)
```

## 2) Instantiate a `Sequential` Model

In the cell below, create an instance of a `Sequential` model ([documentation here](https://keras.io/guides/sequential_model/)) called `dense_model` with a `name` of `"dense"` and otherwise default arguments.

*In other words, create a model without any layers. We will add layers in a future step.*


```python
# Replace None with appropriate code

dense_model = None
```


```python
# __SOLUTION__

dense_model = Sequential(name="dense")
```

    Metal device set to: Apple M1 Pro


    2023-08-04 13:01:13.542446: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:305] Could not identify NUMA node of platform GPU ID 0, defaulting to 0. Your kernel may not have been built with NUMA support.
    2023-08-04 13:01:13.542894: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:271] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 0 MB memory) -> physical PluggableDevice (device: 0, name: METAL, pci bus id: <undefined>)



```python
# Run this code without change

# Assert

assert len(dense_model.layers) == 0
assert type(dense_model) == Sequential
assert dense_model.name == "dense"

```


```python
# __SOLUTION__

# Assert

assert len(dense_model.layers) == 0
assert type(dense_model) == Sequential
assert dense_model.name == "dense"
```

## 3) Determine Input and Output Shapes

How many input and output nodes should this model have?

Feel free to explore the attributes of `X` and `y` to determine this answer, or just to enter numbers based on the problem description above.


```python
# Replace None with appropriate code

num_input_nodes = None
num_output_nodes = None
```


```python
# __SOLUTION__

# The number of input nodes is the number of features
num_input_nodes = X.shape[1]
# For a binary classification task, we only need 1 output node
num_output_nodes = 1
```


```python
# Run this code without change

# Both values should be integers
assert type(num_input_nodes) == int
assert type(num_output_nodes) == int

score = 0

# 300 features, so 300 input nodes
if num_input_nodes == 300:
    score += 0.5
    
# binary output, so 1 output node
if num_output_nodes == 1:
    score += 0.5
elif num_output_nodes == 2:
    # Partial credit for this answer, since it's technically
    # possible to use 2 output nodes for this, although it's
    # confusingly redundant
    score += 0.25

score
```


```python
# __SOLUTION__

# Both values should be integers
assert type(num_input_nodes) == int
assert type(num_output_nodes) == int

score = 0

# 300 features, so 300 input nodes
if num_input_nodes == 300:
    score += 0.5
    
# binary output, so 1 output node
if num_output_nodes == 1:
    score += 0.5
elif num_output_nodes == 2:
    # Partial credit for this answer, since it's technically
    # possible to use 2 output nodes for this, although it's
    # confusingly redundant
    score += 0.25

score
```




    1.0



The code below will use the input and output shapes you specified to add `Dense` layers to the model:


```python
# Run this cell without changes

# Add input layer
dense_model.add(Dense(units=64, input_shape=(num_input_nodes,)))

# Add hidden layers
dense_model.add(Dense(units=64))
dense_model.add(Dense(units=64))

dense_model.layers
```


```python
# __SOLUTION__

# Add input layer
dense_model.add(Dense(units=64, input_shape=(num_input_nodes,)))

# Add hidden layers
dense_model.add(Dense(units=64))
dense_model.add(Dense(units=64))

dense_model.layers
```




    [<keras.layers.core.dense.Dense at 0x16209f850>,
     <keras.layers.core.dense.Dense at 0x162048af0>,
     <keras.layers.core.dense.Dense at 0x15212da30>]



## 4) Add an Output Layer

Specify an appropriate activation function ([documentation here](https://keras.io/api/layers/activations/)).

We'll simplify the problem by specifying that you should use the string identifier for the function, and it should be one of these options:

* `sigmoid`
* `softmax`

***Hint:*** is this a binary or a multi-class problem? This should guide your choice of activation function.


```python
# Replace None with appropriate code

activation_function = None
```


```python
# __SOLUTION__

# sigmoid
activation_function = "sigmoid"
```


```python
# Run this cell without changes

# activation_function should be a string
assert type(activation_function) == str

if num_output_nodes == 1:
    assert activation_function == "sigmoid"
else:
    # The number of output nodes _should_ be 1, but we'll
    # give credit for a matching function even if the
    # previous answer was incorrect
    assert activation_function == "softmax"
```


```python
# __SOLUTION__

# activation_function should be a string
assert type(activation_function) == str

if num_output_nodes == 1:
    assert activation_function == "sigmoid"
else:
    # The number of output nodes _should_ be 1, but we'll
    # give credit for a matching function even if the
    # previous answer was incorrect
    assert activation_function == "softmax"
```

Now we'll use that information to finalize the model.

If this code produces an error, consider restarting the kernel and re-running the code above. If it still produces an error, that is an indication that one or more of your answers above is incorrect.


```python
# Run this cell without changes

# Add output layer
dense_model.add(Dense(units=num_output_nodes, activation=activation_function))

# Determine appropriate loss function
if num_output_nodes == 1:
    loss = "binary_crossentropy"
else:
    loss = "categorical_crossentropy"

# Compile model
dense_model.compile(
    optimizer="adam",
    loss=loss,
    metrics=["accuracy"]
)

dense_model.summary()
```


```python
# __SOLUTION__

# Add output layer
dense_model.add(Dense(units=num_output_nodes, activation=activation_function))

# Determine appropriate loss function
if num_output_nodes == 1:
    loss = "binary_crossentropy"
else:
    loss = "categorical_crossentropy"

# Compile model
dense_model.compile(
    optimizer="adam",
    loss=loss,
    metrics=["accuracy"]
)

dense_model.summary()
```

    Model: "dense"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense (Dense)               (None, 64)                19264     
                                                                     
     dense_1 (Dense)             (None, 64)                4160      
                                                                     
     dense_2 (Dense)             (None, 64)                4160      
                                                                     
     dense_3 (Dense)             (None, 1)                 65        
                                                                     
    =================================================================
    Total params: 27,649
    Trainable params: 27,649
    Non-trainable params: 0
    _________________________________________________________________



```python
# Replace None as necessary

# Fit the model to the training data, using a subset of the
# training data as validation data
dense_model_results = dense_model.fit(
    x=None,
    y=None,
    batch_size=None,
    epochs=None,
    verbose=None,
    validation_split=0.4,
    shuffle=None
)
```


```python
# __SOLUTION__

# Fit the model to the training data, using a subset of the
# training data as validation data
dense_model_results = dense_model.fit(
    x=X_train,
    y=y_train,
    batch_size=None,
    epochs=20,
    verbose=1,
    validation_split=0.4,
    shuffle=False
)
```

    Epoch 1/20


    2023-08-04 13:02:16.373291: W tensorflow/core/platform/profile_utils/cpu_utils.cc:128] Failed to get CPU frequency: 0 Hz
    2023-08-04 13:02:16.503351: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.


    4/4 [==============================] - 1s 81ms/step - loss: 0.9172 - accuracy: 0.5893 - val_loss: 0.9539 - val_accuracy: 0.5600
    Epoch 2/20
    4/4 [==============================] - 0s 16ms/step - loss: 0.4043 - accuracy: 0.8036 - val_loss: 0.9276 - val_accuracy: 0.5733
    Epoch 3/20
    1/4 [======>.......................] - ETA: 0s - loss: 0.1492 - accuracy: 1.0000

    2023-08-04 13:02:16.824131: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.


    4/4 [==============================] - 0s 15ms/step - loss: 0.2300 - accuracy: 0.9375 - val_loss: 0.9474 - val_accuracy: 0.6267
    Epoch 4/20
    4/4 [==============================] - 0s 15ms/step - loss: 0.1493 - accuracy: 0.9732 - val_loss: 0.9921 - val_accuracy: 0.6667
    Epoch 5/20
    4/4 [==============================] - 0s 14ms/step - loss: 0.0983 - accuracy: 0.9821 - val_loss: 1.0534 - val_accuracy: 0.6400
    Epoch 6/20
    4/4 [==============================] - 0s 16ms/step - loss: 0.0612 - accuracy: 1.0000 - val_loss: 1.1272 - val_accuracy: 0.6533
    Epoch 7/20
    4/4 [==============================] - 0s 16ms/step - loss: 0.0364 - accuracy: 1.0000 - val_loss: 1.2082 - val_accuracy: 0.6267
    Epoch 8/20
    4/4 [==============================] - 0s 15ms/step - loss: 0.0220 - accuracy: 1.0000 - val_loss: 1.2890 - val_accuracy: 0.6400
    Epoch 9/20
    4/4 [==============================] - 0s 15ms/step - loss: 0.0139 - accuracy: 1.0000 - val_loss: 1.3638 - val_accuracy: 0.6267
    Epoch 10/20
    4/4 [==============================] - 0s 16ms/step - loss: 0.0093 - accuracy: 1.0000 - val_loss: 1.4296 - val_accuracy: 0.6267
    Epoch 11/20
    4/4 [==============================] - 0s 15ms/step - loss: 0.0065 - accuracy: 1.0000 - val_loss: 1.4858 - val_accuracy: 0.6267
    Epoch 12/20
    4/4 [==============================] - 0s 15ms/step - loss: 0.0047 - accuracy: 1.0000 - val_loss: 1.5328 - val_accuracy: 0.6267
    Epoch 13/20
    4/4 [==============================] - 0s 15ms/step - loss: 0.0036 - accuracy: 1.0000 - val_loss: 1.5720 - val_accuracy: 0.6267
    Epoch 14/20
    4/4 [==============================] - 0s 14ms/step - loss: 0.0029 - accuracy: 1.0000 - val_loss: 1.6047 - val_accuracy: 0.6400
    Epoch 15/20
    4/4 [==============================] - 0s 15ms/step - loss: 0.0024 - accuracy: 1.0000 - val_loss: 1.6321 - val_accuracy: 0.6400
    Epoch 16/20
    4/4 [==============================] - 0s 14ms/step - loss: 0.0020 - accuracy: 1.0000 - val_loss: 1.6555 - val_accuracy: 0.6400
    Epoch 17/20
    4/4 [==============================] - 0s 14ms/step - loss: 0.0018 - accuracy: 1.0000 - val_loss: 1.6757 - val_accuracy: 0.6400
    Epoch 18/20
    4/4 [==============================] - 0s 14ms/step - loss: 0.0016 - accuracy: 1.0000 - val_loss: 1.6935 - val_accuracy: 0.6400
    Epoch 19/20
    4/4 [==============================] - 0s 15ms/step - loss: 0.0014 - accuracy: 1.0000 - val_loss: 1.7095 - val_accuracy: 0.6400
    Epoch 20/20
    4/4 [==============================] - 0s 14ms/step - loss: 0.0013 - accuracy: 1.0000 - val_loss: 1.7240 - val_accuracy: 0.6400



```python
# Run this cell without changes

def plot_loss_and_accuracy(results, final=False):
    
    if final:
        val_label="test"
    else:
        val_label="validation"

    # Extracting metrics from model fitting
    train_loss = results.history['loss']
    val_loss = results.history['val_loss']
    train_accuracy = results.history['accuracy']
    val_accuracy = results.history['val_accuracy']

    # Setting up plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Plotting loss info
    ax1.set_title("Loss")
    sns.lineplot(x=results.epoch, y=train_loss, ax=ax1, label="train")
    sns.lineplot(x=results.epoch, y=val_loss, ax=ax1, label=val_label)
    ax1.legend()

    # Plotting accuracy info
    ax2.set_title("Accuracy")
    sns.lineplot(x=results.epoch, y=train_accuracy, ax=ax2, label="train")
    sns.lineplot(x=results.epoch, y=val_accuracy, ax=ax2, label=val_label)
    ax2.legend()
    
plot_loss_and_accuracy(dense_model_results)
```


```python
# __SOLUTION__

def plot_loss_and_accuracy(results, final=False):
    
    if final:
        val_label="test"
    else:
        val_label="validation"

    # Extracting metrics from model fitting
    train_loss = results.history['loss']
    val_loss = results.history['val_loss']
    train_accuracy = results.history['accuracy']
    val_accuracy = results.history['val_accuracy']

    # Setting up plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Plotting loss info
    ax1.set_title("Loss")
    sns.lineplot(x=results.epoch, y=train_loss, ax=ax1, label="train")
    sns.lineplot(x=results.epoch, y=val_loss, ax=ax1, label=val_label)
    ax1.legend()

    # Plotting accuracy info
    ax2.set_title("Accuracy")
    sns.lineplot(x=results.epoch, y=train_accuracy, ax=ax2, label="train")
    sns.lineplot(x=results.epoch, y=val_accuracy, ax=ax2, label=val_label)
    ax2.legend()
    
plot_loss_and_accuracy(dense_model_results)
```


    
![png](index_files/index_32_0.png)
    


## 5) Modify the Code Below to Use Regularization


The model appears to be overfitting. To deal with this overfitting, modify the code below to include regularization in the model. You can add L1, L2, both L1 and L2, or dropout regularization.

Hint: these might be helpful

 - [`Dense` layer documentation](https://keras.io/api/layers/core_layers/dense/)
 - [`regularizers` documentation](https://keras.io/regularizers/)
 
(`EarlyStopping` is a type of regularization that is not applicable to this problem framing, since it's a callback and not a layer.)


```python
# Complete the following code

def build_model_with_regularization(n_input, n_output, activation, loss):
    """
    Creates and compiles a tf.keras Sequential model with two hidden layers
    This time regularization has been added
    """
    # create classifier
    classifier = Sequential(name="regularized")

    # add input layer
    classifier.add(Dense(units=64, input_shape=(n_input,)))

    # add hidden layers


    # add output layer


    classifier.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    return classifier

model_with_regularization = build_model_with_regularization(
    num_input_nodes, num_output_nodes, activation_function, loss
)
model_with_regularization.summary()
```


```python
# __SOLUTION__

def build_model_with_regularization(n_input, n_output, activation, loss):
    """
    Creates and compiles a tf.keras Sequential model with two hidden layers
    This time regularization has been added
    """
    # create classifier
    classifier = Sequential(name="regularized")

    # add input layer
    classifier.add(Dense(units=64, input_shape=(n_input,)))

    # add hidden layers
    
    ### BEGIN SOLUTION
    
    # they might add a kernel regularizer
    classifier.add(Dense(units=64, kernel_regularizer=regularizers.l2(0.0000000000000001)))
    # they might add a dropout layer
    classifier.add(Dropout(0.8))
    classifier.add(Dense(units=64, kernel_regularizer=regularizers.l2(0.0000000000000001)))
    
    ### END SOLUTION

    # add output layer
    classifier.add(Dense(units=n_output, activation=activation))

    classifier.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    return classifier

model_with_regularization = build_model_with_regularization(
    num_input_nodes, num_output_nodes, activation_function, loss
)
model_with_regularization.summary()
```

    Model: "regularized"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense_4 (Dense)             (None, 64)                19264     
                                                                     
     dense_5 (Dense)             (None, 64)                4160      
                                                                     
     dropout (Dropout)           (None, 64)                0         
                                                                     
     dense_6 (Dense)             (None, 64)                4160      
                                                                     
     dense_7 (Dense)             (None, 1)                 65        
                                                                     
    =================================================================
    Total params: 27,649
    Trainable params: 27,649
    Non-trainable params: 0
    _________________________________________________________________



```python
# Run the code below without change

# Testing function to build model
assert type(model_with_regularization) == Sequential

def check_regularization(model):
    regularization_count = 0
    for layer in model.get_config()['layers']:
        
        # Checking if kernel regularizer was specified
        if 'kernel_regularizer' in layer['config']:
            if layer['config'].get('kernel_regularizer'):
                regularization_count += 1
                
        # Checking if layer is dropout layer
        if layer["class_name"] == "Dropout":
            regularization_count += 1
            
    return regularization_count > 0
    
score = .3

if check_regularization(model_with_regularization):
    score += .7
    
score
```


```python
# __SOLUTION__

# Testing function to build model
assert type(model_with_regularization) == Sequential

def check_regularization(model):
    regularization_count = 0
    for layer in model.get_config()['layers']:
        
        # Checking if kernel regularizer was specified
        if 'kernel_regularizer' in layer['config']:
            if layer['config'].get('kernel_regularizer'):
                regularization_count += 1
                
        # Checking if layer is dropout layer
        if layer["class_name"] == "Dropout":
            regularization_count += 1
            
    return regularization_count > 0
    
score = .3

if check_regularization(model_with_regularization):
    score += .7
    
score
```




    1.0



Now we'll evaluate the new model on the training set as well:


```python
# Run this cell without changes

# Fit the model to the training data, using a subset of the
# training data as validation data
reg_model_results = model_with_regularization.fit(
    x=X_train,
    y=y_train,
    batch_size=None,
    epochs=20,
    verbose=0,
    validation_split=0.4,
    shuffle=False
)

plot_loss_and_accuracy(reg_model_results)
```


```python
# __SOLUTION__

# Fit the model to the training data, using a subset of the
# training data as validation data
reg_model_results = model_with_regularization.fit(
    x=X_train,
    y=y_train,
    batch_size=None,
    epochs=20,
    verbose=0,
    validation_split=0.4,
    shuffle=False
)

plot_loss_and_accuracy(reg_model_results)
```

    2023-08-04 13:03:02.163324: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.
    2023-08-04 13:03:02.455015: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.



    
![png](index_files/index_40_1.png)
    


(Whether or not your regularization made a difference will partially depend on how strong of regularization you applied, as well as some random elements of your current TensorFlow configuration.)

Now we evaluate both models on the holdout set:


```python
# Run this cell without changes

final_dense_model_results = dense_model.fit(
    x=X_train,
    y=y_train,
    batch_size=None,
    epochs=20,
    verbose=0,
    validation_data=(X_test, y_test),
    shuffle=False
)

plot_loss_and_accuracy(final_dense_model_results, final=True)
```


```python
# __SOLUTION__

final_dense_model_results = dense_model.fit(
    x=X_train,
    y=y_train,
    batch_size=None,
    epochs=20,
    verbose=0,
    validation_data=(X_test, y_test),
    shuffle=False
)

plot_loss_and_accuracy(final_dense_model_results, final=True)
```


    
![png](index_files/index_43_0.png)
    


Plot the loss and accuracy your final regularized model.


```python
# Replace None, as necessary

final_reg_model_results = model_with_regularization.fit(
    x=None,
    y=None,
    batch_size=None,
    epochs=None,
    verbose=None,
    validation_data=(None, None),
    shuffle=None
)

plot_loss_and_accuracy(final_reg_model_results, final=True)
```


```python
# __SOLUTION__

# Loss and accuracy of final model

final_reg_model_results = model_with_regularization.fit(
    x=X_train,
    y=y_train,
    batch_size=None,
    epochs=20,
    verbose=0,
    validation_data=(X_test, y_test),
    shuffle=False
)

plot_loss_and_accuracy(final_reg_model_results, final=True)
```


    
![png](index_files/index_46_0.png)
    

