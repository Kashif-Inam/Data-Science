import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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


def pred_network(x):
    prediction = neural_network(x)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # sess.run(init)

        saver.restore(sess, model_path)

        pred = tf.argmax(prediction, 1)
        p = np.array([90,96,34,49,169,4.0])
        p = p.reshape(1, -1)
        p_run = sess.run(pred, feed_dict={x: p})

        print("0 Stands for Class-1 and 1 Stands for Class-2")
        print("Predicted Values:", p_run)


pred_network(x)
