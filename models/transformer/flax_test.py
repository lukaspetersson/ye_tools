##
import flax
import jax
import optax
from sklearn import datasets
from sklearn.model_selection import train_test_split
from jax import numpy as jnp

##
X, Y = datasets.load_boston(return_X_y=True)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=123)

X_train, X_test, Y_train, Y_test = jnp.array(X_train, dtype=jnp.float32),\
                                   jnp.array(X_test, dtype=jnp.float32),\
                                   jnp.array(Y_train, dtype=jnp.float32),\
                                   jnp.array(Y_test, dtype=jnp.float32)


samples, features = X_train.shape

X_train.shape, X_test.shape, Y_train.shape, Y_test.shape
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

from typing import Sequence, Tuple
from jax import random
import jax.numpy as jnp
from flax import linen

class MultiLayerPerceptronRegressor(linen.Module):
    features: Sequence[int] = (5,10,15,1)

    def setup(self):
        self.layers = [linen.Dense(feat) for feat in self.features]

    def __call__(self, inputs):
        x = inputs
        for i, lyr in enumerate(self.layers):
            x = lyr(x)
            if i != len(self.layers) - 1:
                x = linen.relu(x)
        return x

seed = random.PRNGKey(0)

model = MultiLayerPerceptronRegressor()
params = model.init(seed, X_train[:5])

for layer_params in params["params"].items():
    print("Layer Name : {}".format(layer_params[0]))
    weights, biases = layer_params[1]["kernel"], layer_params[1]["bias"]
    print("\tLayer Weights : {}, Biases : {}".format(weights.shape, biases.shape))

preds = model.apply(params, X_train[:5])

preds

def MeanSquaredErrorLoss(weights, input_data, actual):
    preds = model.apply(weights, input_data)
    return jnp.power(actual - preds.squeeze(), 2).mean()

seed = random.PRNGKey(0)
epochs=1000

model = MultiLayerPerceptronRegressor() ## Define Model
random_arr = jax.random.normal(key=seed, shape=(5, features))
params = model.init(seed, random_arr) ## Initialize Model Parameters

optimizer = optax.sgd(learning_rate=1/1e3) ## Initialize SGD Optimizer using OPTAX

optimizer_state = optimizer.init(params)
loss_grad = jax.value_and_grad(MeanSquaredErrorLoss)

for i in range(1,epochs+1):
    loss_val, gradients = loss_grad(params, X_train, Y_train) ## Calculate Loss and Gradients
    updates, optimizer_state = optimizer.update(gradients, optimizer_state)
    params = optax.apply_updates(params, updates) ## Update weights
    if i % 100 == 0:
        print('MSE After {} Epochs : {:.2f}'.format(i, loss_val))

test_preds = model.apply(params, X_test) ## Make Predictions on test dataset

test_preds = test_preds.ravel()

train_preds = model.apply(params, X_train) ## Make Predictions on train dataset

train_preds = train_preds.ravel()

from sklearn.metrics import r2_score

print("Train R^2 Score : {:.2f}".format(r2_score(train_preds.to_py(), Y_train.to_py())))
print("Test  R^2 Score : {:.2f}".format(r2_score(test_preds.to_py(), Y_test.to_py())))

