#!/usr/bin/env python3
"""
Module to perform Bayesian Optimization on a Deep Learning model
using GPyOpt.
"""
import GPyOpt
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


class BayesianOptimizationModel:
    """Class to perform Bayesian Optimization on a machine learning model."""

    def __init__(self):
        """Initializes the BayesianOptimizationModel class and dataset."""
        # Fix seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)

        # Generate a fixed dataset ONCE so the objective function is consistent
        input_shape = 10
        self.input_shape = input_shape
        self.X_train = np.random.rand(1000, input_shape)
        self.y_train = np.random.rand(1000, 1)
        self.X_val = np.random.rand(200, input_shape)
        self.y_val = np.random.rand(200, 1)

        # Search space for hyperparameters
        self.bounds = [
            {'name': 'lr', 'type': 'continuous', 'domain': (0.0001, 0.1)},
            {'name': 'units', 'type': 'discrete',
             'domain': (16, 32, 64, 128, 256)},
            {'name': 'dropout', 'type': 'continuous', 'domain': (0.1, 0.5)},
            {'name': 'l2', 'type': 'continuous', 'domain': (0.0001, 0.01)},
            {'name': 'batch_size', 'type': 'discrete',
             'domain': (16, 32, 64, 128)}
        ]

    def build_model(self, lr, units, dropout, l2_reg):
        """
        Builds and compiles a Keras Sequential model.

        Args:
            lr (float): Learning rate.
            units (int): Number of units in the hidden layer.
            dropout (float): Dropout rate.
            l2_reg (float): L2 regularization weight.

        Returns:
            model: Compiled Keras model.
        """
        model = Sequential([
            Dense(units, activation='relu', kernel_regularizer=l2(l2_reg),
                  input_shape=(self.input_shape,)),
            Dropout(dropout),
            Dense(1, activation='linear')
        ])

        optimizer = Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        return model

    def objective_function(self, X):
        """
        Objective function to minimize during Bayesian Optimization.

        Args:
            X (numpy.ndarray): Array of hyperparameter values to evaluate.

        Returns:
            best_val_loss (float): The minimum validation loss obtained.
        """
        x = X[0]
        lr = float(x[0])
        units = int(x[1])
        dropout = float(x[2])
        l2_reg = float(x[3])
        batch_size = int(x[4])

        model = self.build_model(lr, units, dropout, l2_reg)

        filename = (
            f"model_checkpoints/model_lr{lr:.4f}_units{units}_"
            f"dropout{dropout:.2f}_l2{l2_reg:.4f}_batch{batch_size}.h5"
        )

        # Callbacks: Early Stopping + Checkpoint saving the best epoch
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=filename, monitor='val_loss', save_best_only=True,
            verbose=0
        )

        history = model.fit(
            self.X_train, self.y_train,
            epochs=50,
            batch_size=batch_size,
            validation_data=(self.X_val, self.y_val),
            callbacks=[early_stopping, model_checkpoint],
            verbose=0
        )

        # Minimum validation loss achieved across epochs
        best_val_loss = float(np.min(history.history['val_loss']))
        return best_val_loss

    def run_optimization(self):
        """
        Runs the Bayesian Optimization process.

        Saves the best hyperparameters to bayes_opt.txt and outputs
        the convergence plot.
        """
        opt = GPyOpt.methods.BayesianOptimization(
            f=self.objective_function,
            domain=self.bounds,
            acquisition_type='EI',
            exact_feval=True
        )
        opt.run_optimization(max_iter=30)

        best_parameters = opt.x_opt
        best_performance = opt.fx_opt

        with open('bayes_opt.txt', 'w') as f:
            f.write("Best Hyperparameters:\n")
            f.write(f"Learning Rate: {best_parameters[0]}\n")
            f.write(f"Units: {int(best_parameters[1])}\n")
            f.write(f"Dropout: {best_parameters[2]}\n")
            f.write(f"L2: {best_parameters[3]}\n")
            f.write(f"Batch Size: {int(best_parameters[4])}\n")
            f.write(f"Best Metric: {best_performance}\n")

        # Plot convergence as required by the subject
        opt.plot_convergence(filename='convergence_plot.png')


if __name__ == "__main__":
    optimizer = BayesianOptimizationModel()
    optimizer.run_optimization()
