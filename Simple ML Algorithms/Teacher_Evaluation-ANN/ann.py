import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, scale
from sklearn.utils import shuffle

def read_dataset():
    df = pd.read_csv("tae.data.txt")
    X = np.array(df.drop("Class", 1))
    y = np.array(df["Class"])

    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    Y = one_hot_encode(y)

    return X, Y

def one_hot_encode(label):
    n_label = len(label)
    n_unique_label = len(np.unique(label))
    one_hot_encode = np.zeros((n_label, n_unique_label))
    one_hot_encode[np.arange(n_label), label] = 1
    return one_hot_encode

X, y = read_dataset()
X = scale(X)
X, y = shuffle(X, y, random_state=1)
Class = {"Low": 1, "Medium": 2, "High": 3}

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)

n_nodes_1 = 500
n_nodes_2 = 500
n_nodes_3 = 500
n_nodes_4 = 500

model_path = "C:\\Users\Kashif\PycharmProjects\DeepLearning-Tensorflow (Sentdex)\Learnings\TeacherAssistantEvaluation\TAE_Model"
n_classes = 3
input_size = X.shape[1]

x = tf.placeholder(tf.float32, [None, input_size])
y = tf.placeholder(tf.float32, [None, n_classes])

def neural_network(x):
    hidden_1 = {"weights": tf.Variable(tf.random_normal([input_size, n_nodes_1])),
                "biases": tf.Variable(tf.random_normal([n_nodes_1]))}

    hidden_2 = {"weights": tf.Variable(tf.random_normal([n_nodes_1, n_nodes_2])),
                "biases": tf.Variable(tf.random_normal([n_nodes_2]))}

    hidden_3 = {"weights": tf.Variable(tf.random_normal([n_nodes_2, n_nodes_3])),
                "biases": tf.Variable(tf.random_normal([n_nodes_3]))}

    hidden_4 = {"weights": tf.Variable(tf.random_normal([n_nodes_3, n_nodes_4])),
                "biases": tf.Variable(tf.random_normal([n_nodes_4]))}

    out_layer = {"weights": tf.Variable(tf.random_normal([n_nodes_4, n_classes])),
                 "biases": tf.Variable(tf.random_normal([n_classes]))}

    # (input * weights) + biases

    layer_1 = tf.add(tf.matmul(x, hidden_1["weights"]), hidden_1["biases"])
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, hidden_2["weights"]), hidden_2["biases"])
    layer_2 = tf.nn.relu(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, hidden_3["weights"]), hidden_3["biases"])
    layer_3 = tf.nn.relu(layer_3)

    layer_4 = tf.add(tf.matmul(layer_3, hidden_4["weights"]), hidden_4["biases"])
    layer_4 = tf.nn.relu(layer_4)

    output = tf.matmul(layer_4, out_layer["weights"]) + out_layer["biases"]

    return output

def train_model(x):
    prediction = neural_network(x)
    cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost_function)

    init = tf.global_variables_initializer()
    loss_trace = []
    accuracy_trace = []
    saver = tf.train.Saver()

    epochs = 1000

    with tf.Session() as sess:
        sess.run(init)
        for i in range(epochs):
            sess.run(optimizer, feed_dict={x: train_X, y: train_y})
            loss = sess.run(cost_function, feed_dict={x: train_X, y: train_y})
            accuracy = np.mean(np.argmax(sess.run(prediction, feed_dict={x: train_X, y: train_y}), axis=1) == np.argmax(train_y, axis=1))
            loss_trace.append(loss)
            accuracy_trace.append(accuracy)
            print('Epoch:', (i + 1), 'loss:', loss, 'accuracy:', accuracy)

        saver.save(sess, model_path)
        print('Final training result:', 'loss:', loss, 'accuracy:', accuracy)
        loss_test = sess.run(cost_function, feed_dict={x: test_X, y: test_y})
        test_pred = np.argmax(sess.run(prediction, feed_dict={x: test_X, y: test_y}), axis=1)
        accuracy_test = np.mean(test_pred == np.argmax(test_y, axis=1))
        print('Results on test dataset:', 'loss:', loss_test, 'accuracy:', accuracy_test)


train_model(x)
