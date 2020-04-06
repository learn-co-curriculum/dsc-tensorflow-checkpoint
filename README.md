# Neural Network Regularization

This assessment covers building and training a `tf.keras` `Sequential` model, then applying regularization.  The dataset comes from a ["don't overfit" Kaggle competition](https://www.kaggle.com/c/dont-overfit-ii).  There are 300 features labeled 0-299, and a target called "target".  There are only 250 records total, meaning this is a very small dataset to be used with a neural network. 

_You can assume that the dataset has already been scaled._


```python
# Run this cell without changes

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score

import tensorflow as tf
from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
tf.logging.set_verbosity(tf.logging.ERROR)
```

In the cells below, the set of data has been split into a training and testing set and then fit to a neural network with two hidden layers. Run the cells below to see how well the model performs.


```python
# Run this cell without changes

df = pd.read_csv("data.csv")
df.drop("id", axis=1, inplace=True)

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2020)
X_train.shape
```


```python
# Run this cell without changes

def build_model():
    """
    Creates and compiles a tf.keras Sequential model with two hidden layers
    """
    # create classifier
    classifier = Sequential()

    # add input layer (shape is 300 because X has 300 features)
    classifier.add(Dense(units=64, input_shape=(300,)))

    # add hidden layers
    classifier.add(Dense(units=64))
    classifier.add(Dense(units=64))

    # add output layer
    classifier.add(Dense(units=1, activation='sigmoid'))

    classifier.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])
    return classifier
```


```python
# Run this cell without changes

def fit_and_cross_validate_model(model_func, X, y):
    """
    Given a function that builds a model and training X and y, validate the model based on
    cross-validated train and test data
    """
    keras_classifier = KerasClassifier(build_model, epochs=5, batch_size=50, verbose=1, shuffle=False)
    
    print("######################## Training cross-validated models ###########################")
    cross_val_scores = cross_val_score(keras_classifier, X, y, cv=5)
    
    print("########################### Training on full X_train ###############################")
    keras_classifier.fit(X, y)
    
    print("############################### Evaluation report ##################################")
    
    print("Approximate training accuracy:")
    print(accuracy_score(y, keras_classifier.predict(X)))
    
    print("Approximate testing accuracy:")
    print(np.mean(cross_val_scores), "+/-", np.std(cross_val_scores))
```


```python
# Run this cell without changes

fit_and_cross_validate_model(build_model, X_train, y_train);
```

## 1) Modify the code below to use regularization


The model appears to be overfitting. To deal with this overfitting, modify the code below to include regularization in the model. You can add L1, L2, both L1 and L2, or dropout regularization.

Hint: these might be helpful

 - [`Dense` layer documentation](https://keras.io/layers/core/)
 - [`regularizers` documentation](https://keras.io/regularizers/)


```python
def build_model_with_regularization():
    """
    Creates and compiles a tf.keras Sequential model with two hidden layers
    This time regularization has been added
    """
    # create classifier
    classifier = Sequential()

    # add input layer
    classifier.add(Dense(units=64, input_shape=(300,)))

    # add hidden layers
    
    # YOUR CODE HERE

    # add output layer
    classifier.add(Dense(units=1, activation='sigmoid'))

    classifier.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])
    return classifier

```


```python
# Run this cell without changes

fit_and_cross_validate_model(build_model_with_regularization, X_train, y_train);
```

### Based on the cross-validated scores, did the regularization you performed help prevent overfitting? Is the first or the second model better?


```python
# Your written answer here
```

### Now, evaluate both models on the holdout set


```python
# Run this cell without changes

classifier_1 = build_model()
classifier_1.fit(X_train, y_train, epochs=5, verbose=1, batch_size=50, shuffle=False)

classifier_2 = build_model_with_regularization()
classifier_2.fit(X_train, y_train, epochs=5, verbose=1, batch_size=50, shuffle=False)

print("Accuracy score without regularization:", accuracy_score(y_test, classifier_1.predict_classes(X_test)))
print("Accuracy score with regularization:", accuracy_score(y_test, classifier_2.predict_classes(X_test)))
```

### 2) Explain how regularization is related to the bias/variance tradeoff within Neural Networks and how it's related to the results you just achieved in the training and test accuracies of the previous models. What does regularization change in the training process (be specific to what is being regularized and how it is regularizing)?



```python
# Your answer here
```

### 3) How might L1  and dropout regularization change a neural network's architecture?


```python
# Your answer here
```
