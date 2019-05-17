import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder


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
Class = {"Low": 0, "Medium": 1, "High": 2}

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
    hidden_1 = {"weights": tf.Variable(tf.truncated_normal([input_size, n_nodes_1])),
                "biases": tf.Variable(tf.truncated_normal([n_nodes_1]))}

    hidden_2 = {"weights": tf.Variable(tf.truncated_normal([n_nodes_1, n_nodes_2])),
                "biases": tf.Variable(tf.truncated_normal([n_nodes_2]))}

    hidden_3 = {"weights": tf.Variable(tf.truncated_normal([n_nodes_2, n_nodes_3])),
                "biases": tf.Variable(tf.truncated_normal([n_nodes_3]))}

    hidden_4 = {"weights": tf.Variable(tf.truncated_normal([n_nodes_3, n_nodes_4])),
                "biases": tf.Variable(tf.truncated_normal([n_nodes_4]))}

    output = {"weights": tf.Variable(tf.truncated_normal([n_nodes_4, n_classes])),
                "biases": tf.Variable(tf.truncated_normal([n_classes]))}

    # (input * weights) + biases

    layer_1 = tf.add(tf.matmul(x, hidden_1["weights"]), hidden_1["biases"])
    layer_1 = tf.nn.sigmoid(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, hidden_2["weights"]), hidden_2["biases"])
    layer_2 = tf.nn.sigmoid(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, hidden_3["weights"]), hidden_3["biases"])
    layer_3 = tf.nn.sigmoid(layer_3)

    layer_4 = tf.add(tf.matmul(layer_3, hidden_4["weights"]), hidden_4["biases"])
    layer_4 = tf.nn.sigmoid(layer_4)

    out_layer = tf.matmul(layer_4, output["weights"]) + output["biases"]

    return out_layer

def predict_model(x):
    prediction = neural_network(x)
    cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost_function)

    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, model_path)

    pred = tf.argmax(prediction, 1)
    p = np.array([2,15,1,2,19])
    p = p.reshape(1, -1)
    p_run = sess.run(pred, feed_dict={x: p})

    print("Teacher Assistant Evaluation:")
    print(p_run)

    for key, val in Class.items():
        if  val == p_run:
            print("Predicted Values:", key)

predict_model(x)
