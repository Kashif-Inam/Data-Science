import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


# Reading the dataset
def read_dataset():
    df = pd.read_csv("sonar.all-data.txt")
    # print(len(df.columns))
    X = df[df.columns[0:60]].values 
    y1 = df[df.columns[60]]
    
    # Encode y variable
    encoder = LabelEncoder()
    encoder.fit(y1)
    y = encoder.transform(y1)
    Y = one_hot_encode(y)
    # print(X.shape)
    return X, Y, y1


def one_hot_encode(label):
    n_label = len(label)
    n_unique_label = len(np.unique(label))
    one_hot_encode = np.zeros((n_label, n_unique_label))
    one_hot_encode[np.arange(n_label), label] = 1 
    return one_hot_encode


X, Y, y1 = read_dataset()

learning_rate = 0.3
training_epochs = 1000
n_dim = X.shape[1]
# print(n_dim)
n_class = 2
model_path = "C:\\Users\Kashif\PycharmProjects\DeepLearning-Tensorflow (Sentdex)\Learnings\Rock&MinePrediction\RockMineModel"                                                    # Jahan pe model train kr k save krna he wo path

# Defining no. of hidden layers
n_hidden_1 = 60
n_hidden_2 = 60
n_hidden_3 = 60
n_hidden_4 = 60

x = tf.placeholder(tf.float32, [None, n_dim])
W = tf.Variable(tf.zeros([n_dim, n_class]))
b = tf.Variable(tf.zeros([n_class]))
y_ = tf.placeholder(tf.float32, [None, n_class])


def multilayer_perceptron(x, weights, biases):

    # (input_data * weights) + biases
    layer_1 = tf.add(tf.matmul(x, weights["h1"]), biases["b1"])
    layer_1 = tf.nn.sigmoid(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights["h2"]), biases["b2"])
    layer_2 = tf.nn.sigmoid(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, weights["h3"]), biases["b3"])
    layer_3 = tf.nn.sigmoid(layer_3)

    layer_4 = tf.add(tf.matmul(layer_3, weights["h4"]), biases["b4"])
    layer_4 = tf.nn.sigmoid(layer_4)

    out_layer = tf.matmul(layer_4, weights["out"]) + biases["out"]

    return out_layer


weights = {
    "h1": tf.Variable(tf.truncated_normal([n_dim, n_hidden_1])),
    "h2": tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
    "h3": tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3])),
    "h4": tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4])),
    "out": tf.Variable(tf.truncated_normal([n_hidden_4, n_class])),
}

biases = {
    "b1": tf.Variable(tf.truncated_normal([n_hidden_1])),
    "b2": tf.Variable(tf.truncated_normal([n_hidden_2])),
    "b3": tf.Variable(tf.truncated_normal([n_hidden_3])),
    "b4": tf.Variable(tf.truncated_normal([n_hidden_4])),
    "out": tf.Variable(tf.truncated_normal([n_class])),
}

init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)
saver.restore(sess, model_path)
y = multilayer_perceptron(x, weights, biases)

cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

prediction = tf.argmax(y, 1)
correct_prediction = tf.equal(prediction, tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print("0 Stands for Mine and 1 Stands for Rock")
for i in range(93, 101):
    prediction_run = sess.run(prediction, feed_dict={x: X[i].reshape(1, 60)})
    print("Original Class:", y1[i], " - Predicted Values:", prediction_run)
