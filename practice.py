"""
Tasks
1. Set up environment:
1.1. Python with libraries such as TensorFlow, Keras, and scikit-learn.
1.2. Load the Fashion MNIST dataset.
1.3. Preprocess the data:
    1.3.1. Normalize the pixel values to be between 0 and 1.
    1.3.2.  Split the data into training, validation, and test sets.

2. Build a Baseline Model:
2.1. Create a simple neural network with the following architecture:
    2.1.1. A Flatten layer to convert the 28x28 images into a 1D array.
    2.1.2. A Dense (fully-connected) hidden layer with 128 neurons and a ReLU activation function.
    2.1.3. A Dropout layer with a rate of 0.2.
    2.1.4. An output layer with 10 neurons (one for each class) and a Softmax activation function.
    2.1.5. Compile the model using the Adam optimizer, a learning rate of 0.001, and the sparse_categorical_crossentropy loss function.
    2.1.6. Train settings of the model for 10 epochs with a batch size of 32.
    2.1.7. Provide a Evaluation_Report.MD for model evaluation on the test set and record the accuracy.

3. Provide Reports for:
3.1. Provide a Hyperparameter_tuning_report.MD to create a summary table that compares the performance of your different models.
3.2. Write a brief_report.MD that answers the following questions:
    3.2.1. Which hyperparameter had the most significant impact on the model's performance?
    3.2.2. How did the changes you made affect the training process (e.g., did the model train faster or slower? Did it overfit?)?
    3.2.3. What was the best set of hyperparameters you found, and what was the final test accuracy?
"""


