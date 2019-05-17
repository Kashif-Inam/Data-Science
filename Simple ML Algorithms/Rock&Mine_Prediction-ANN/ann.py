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
    y = df[df.columns[60]]
    
    # Encode y variable
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    Y = one_hot_encode(y)
    # print(X.shape)
    return X, Y


def one_hot_encode(label):
    n_label = len(label)
    n_unique_label = len(np.unique(label))
    one_hot_encode = np.zeros((n_label, n_unique_label))
    one_hot_encode[np.arange(n_label), label] = 1
    return one_hot_encode


X, Y = read_dataset()
X, Y = shuffle(X, Y, random_state=1)

train_X, test_X, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=415)

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
y = multilayer_perceptron(x, weights, biases)

cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

sess = tf.Session()
sess.run(init) 

cost_history = np.empty(shape=[1], dtype=float)
mse_history = []
accuracy_history = []

for epoch in range(training_epochs):
    sess.run(optimizer, feed_dict={x: train_X, y_: train_y})
    cost = sess.run(cost_function, feed_dict={x: train_X, y_: train_y})
    cost_history = np.append(cost_history, cost)
    cost_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(cost_prediction, tf.float32))
    
    pred_y = sess.run(y, feed_dict={x: test_X})
    mse = tf.reduce_mean(tf.square(pred_y - test_y))
    mse_history.append(sess.run(mse))
    accuracy = (sess.run(accuracy, feed_dict={x: train_X, y_: train_y}))
    accuracy_history.append(accuracy)

    print("epoch:", epoch, " - cost:", cost, " - MSE:", sess.run(mse), "-TrainAccuracy", accuracy)

save_path = saver.save(sess, model_path)

plt.plot(mse_history, "r")
plt.show()
plt.plot(accuracy_history)
plt.show()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Test Accuracy:", (sess.run(accuracy, feed_dict={x: test_X, y_: test_y})))

pred_y = sess.run(y, feed_dict={x: test_X})
mse = tf.reduce_mean(tf.square(pred_y - test_y))
print("MSE: %.4f" % sess.run(mse))
