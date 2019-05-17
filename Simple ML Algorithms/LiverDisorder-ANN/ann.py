import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import pandas as pd

def read_dataset():
    df = pd.read_csv("bupa.data.txt")
    X = np.array(df.drop("Selector", 1))
    y = np.array(df["Selector"])

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

X, y = read_dataset()
X, y = shuffle(X, y, random_state=1)

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)

n_node_1 = 500
n_node_2 = 500
n_node_3 = 500

model_path = "C:\\Users\Kashif\PycharmProjects\DeepLearning-Tensorflow (Sentdex)\Learnings\LiverDisorder\LiverModel"
n_classes = 2
input_size = X.shape[1]

x = tf.placeholder(tf.float32, [None, input_size])
y = tf.placeholder(tf.float32, [None, n_classes])

def neural_network(x):

    hidden_1 = {"weights": tf.Variable(tf.random_normal([input_size, n_node_1])),
                "biases": tf.Variable(tf.random_normal([n_node_1]))}

    hidden_2 = {"weights": tf.Variable(tf.random_normal([n_node_1, n_node_2])),
                "biases": tf.Variable(tf.random_normal([n_node_2]))}

    hidden_3 = {"weights": tf.Variable(tf.random_normal([n_node_2, n_node_3])),
                "biases": tf.Variable(tf.random_normal([n_node_3]))}

    out_hidden = {"weights": tf.Variable(tf.random_normal([n_node_3, n_classes])),
                  "biases": tf.Variable(tf.random_normal([n_classes]))}

    # (input * weights) + biases

    layer_1 = tf.add(tf.matmul(x, hidden_1["weights"]), hidden_1["biases"])
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, hidden_2["weights"]), hidden_2["biases"])
    layer_2 = tf.nn.relu(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, hidden_3["weights"]), hidden_3["biases"])
    layer_3 = tf.nn.relu(layer_3)

    out_layer = tf.matmul(layer_3, out_hidden["weights"]) + out_hidden["biases"]

    return out_layer


def train_network(x):
    prediction = neural_network(x)
    cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost_function)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    loss_trace = []
    accuracy_trace = []

    epochs = 1000

    with tf.Session() as sess:
        sess.run(init)

        for i in range(epochs):
            sess.run(optimizer, feed_dict={x: train_X, y: train_y})
            loss = sess.run(cost_function, feed_dict={x: train_X, y: train_y})
            acc = sess.run(prediction, feed_dict={x: train_X, y: train_y})
            accuracy = np.mean(np.argmax(acc, 1) == np.argmax(train_y, 1))
            loss_trace.append(loss)
            accuracy_trace.append(accuracy)
            print("Epoch:", (i + 1), "Loss:", loss, "Accuracy:", accuracy)

        saver.save(sess, model_path)
        print("Final training result, ", "Loss:", loss, "Accuracy:", accuracy)
        loss_test = sess.run(cost_function, feed_dict={x: test_X, y: test_y})
        test_pred = np.argmax(sess.run(prediction, feed_dict={x: test_X, y: test_y}), axis=1)
        accuracy_test = np.mean(test_pred == np.argmax(test_y, axis=1))
        print('Results on test dataset:', 'loss:', loss_test, 'accuracy:', accuracy_test)


train_network(x)
