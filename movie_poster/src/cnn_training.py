import os
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta
from keras.callbacks import TensorBoard
import time

def poster_file(movie):
    return input_path / f"{movie[0]}.jpg"


def has_poster(movie):
    return Path(poster_file(movie)).is_file()


def has_genre(movie):
    return len(movie[1]) > 0

def movie_poster(movie):
    with Image.open(poster_file(movie)) as poster:
        poster = poster.resize((poster_width, poster_height), resample=Image.LANCZOS)
        poster = poster.convert('RGB')
        return np.asarray(poster) / 255


def unique(series_of_lists):
    seen = set()
    return [e for lst in series_of_lists
            for e in lst
            if not (e in seen or seen.add(e))]

# check if the movie has the comedy in its own comedy list
def bitmap(lst, uniques):
    bmp = []
    for u in range(0, len(uniques)):
        if uniques[u] in lst:
            bmp.append(1.0)
        else:
            bmp.append(0.0)
    return bmp


def encode(series, uniques):
    return [bitmap(lst, uniques) for lst in series]


def load_data(path):
    csv = path / "MovieGenre.csv"
    movies = pd.read_csv(csv, encoding="ISO-8859-1", usecols=['imdbId', 'Genre'], keep_default_na=False).sample(frac=0.05)
    print(movies.shape)
    movies = movies[movies.apply(lambda d: has_genre(d) and has_poster(d), axis=1)]
    movies = movies.sample(frac=1).reset_index(drop=True)
    posters = list(map(movie_poster, movies.values))
    genres = movies.Genre.str.split("|")
    print("genres: {0}".format(genres[0:1]))
    unique_genres = unique(genres)
    print("unique genres: {0}".format(unique_genres))
    x = np.array(posters)
    y = np.array(encode(genres.values, unique_genres))
    return x, y, unique_genres

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x, stride_x, stride_y):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,stride_x,stride_y,1], padding="SAME")

def compute_accuracy(v_xs, v_ys, session, prediction, xs, ys, keep_prob):
    print(prediction)
    y_pre = session.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = session.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def tf_training(x_train, y_train, x_validation, y_validation):

    # placeholders for giving data from outside
    xs = tf.placeholder(tf.float32)
    ys = tf.placeholder(tf.float32)
    keep_prob = tf.placeholder(tf.float32)
    x_image = tf.reshape(xs, (-1, poster_height, poster_width, poster_channels))

    ## conv1 layer ##
    W_conv1 = weight_variable([3, 3, 3, 16])  # patch 3x3, in size 3, out size 16
    b_conv1 = bias_variable([16])
    conv_layer1 = conv2d(x_image, W_conv1) + b_conv1
    # 非线性化处理
    h_conv1 = tf.nn.relu(conv_layer1)  # output size 64x48x16
    h_pool1 = max_pool_2x2(h_conv1, 2, 2)  # output size 32x24x16

    ## conv2 layer ##
    W_conv2 = weight_variable([5, 5, 16, 32])  # patch 5x5, in size 16, out size 32
    b_conv2 = bias_variable([32])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output size 32x24x32
    h_pool2 = max_pool_2x2(h_conv2, 2, 2)  # output size 16x12x32

    ## fc1 layer ##
    h_pool_dropout = tf.nn.dropout(h_pool2, 0.25)
    W_fc1 = weight_variable([16 * 12 * 32, 128])
    b_fc1 = bias_variable([128])
    # [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64] (flatten)
    h_pool3_flat = tf.reshape(h_pool_dropout, [-1, 16 * 12 * 32])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)  # matmul means matrix multiply
    h_fc1_drop = tf.nn.dropout(h_fc1, 0.5)  # avoid overfitting

    ## fc2 layer ##
    W_fc2 = weight_variable([128, len(movie_genres)])
    b_fc2 = bias_variable([len(movie_genres)])
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # the error between prediction and real data
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                                  reduction_indices=[1]))  # loss
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    init = tf.global_variables_initializer()
    local_init = tf.local_variables_initializer()

    with tf.Session() as session:
        session.run(init)
        session.run(local_init)
        num_batches = int(x_train.shape[0] / batch_size)
        matrix_x = np.array_split(x_train, num_batches)
        matrix_y = np.array_split(y_train, num_batches)
        for i in range(epochs):
            for batch_xs, batch_ys in zip(matrix_x, matrix_y):
                # tf_training(batch_xs, batch_ys, session)
                session.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 1})

            print("epochs: {0}, accuracy: {1}".format(i, compute_accuracy(x_validation, y_validation, session, prediction, xs, ys, keep_prob)))


def create_keras_cnn(height, width, channels, genres):
    cnn = Sequential([
        Conv2D(filters=16, kernel_size=(5, 5), padding='valid', activation="relu", input_shape=(height, width, channels)),
        MaxPooling2D(pool_size=(2, 2), padding='valid'),
        Conv2D(filters=32, kernel_size=(5, 5), padding='valid', activation='relu'),
        MaxPooling2D(pool_size=(2, 2), padding='valid'),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(genres), activation='softmax')
    ])
    cnn.compile(loss=categorical_crossentropy,
                optimizer=Adadelta(),
                metrics=['accuracy'])
    return cnn

def keras_training():
    model = create_keras_cnn(poster_height, poster_width, poster_channels, movie_genres)
    model.summary()
    model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_validation, y_validation))
    (validation_loss, validation_accuracy) = model.evaluate(x_validation, y_validation, verbose=0)
    print('\nValidation loss:', validation_loss)
    print('Validation accuracy:', validation_accuracy)

if __name__ == "__main__":
    python_platform_path = os.path.abspath(__file__ + "/../")
    global input_path
    input_path = Path(python_platform_path+"/MoviePosters")
    data_path = Path(python_platform_path+"/data")

    global poster_height
    global poster_width
    global poster_channels
    global movie_genres

    poster_width = 48  # 182 / 3.7916
    poster_height = 64  # 268 / 4.1875
    poster_channels = 3  # RGB

    epochs = 20
    batch_size = 100

    x_data, y_data, movie_genres = load_data(data_path)
    print(x_data.shape)
    separator = len(x_data) * 3 // 4
    x_train = x_data[0:separator]
    y_train = y_data[0:separator]
    x_validation = x_data[separator:len(x_data)]
    y_validation = y_data[separator:len(y_data)]

    tf_training(x_train, y_train, x_validation, y_validation)
    keras_training()

