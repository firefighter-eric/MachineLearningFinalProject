from time import time
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def scale(train_x, test_x):
    sc_x = StandardScaler()
    sc_x.fit(train_x)
    train_x_sc = sc_x.transform(train_x)
    test_x_sc = sc_x.transform(test_x)
    return train_x_sc, test_x_sc


def knn(train_x, train_y, test_x, test_y, k):
    print('knn running...')
    n_test = 100
    test_x, test_y = test_x[:n_test, :], test_y[:n_test]
    train_x, test_x = scale(train_x, test_x)

    clf = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    clf.fit(train_x, train_y)

    t_test = time()
    s = clf.score(test_x, test_y)
    t_test = time() - t_test
    print('test time:', t_test)
    print("K:", k, "Score:", s)


def svm(train_x, train_y, test_x, test_y):
    print('svm running..')
    train_x, test_x = scale(train_x, test_x)

    clf = SVC(kernel='rbf', max_iter=100)

    t_train = time()
    clf = clf.fit(train_x, train_y)
    t_train = time() - t_train

    t_test = time()
    score = clf.score(test_x, test_y)
    t_test = time() - t_test

    print('train time:', t_train, 'predict time:', t_test)
    print('score:', score)


def linear_classifier(train_x, train_y, test_x, test_y):
    print('linear classifier running..')
    model = keras.Sequential([layers.Dense(10, activation='softmax', input_shape=(28 * 28,))])
    model.compile(optimizer='adam',
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    t_train = time()
    out = model.fit(train_x,
                    train_y,
                    epochs=40,
                    batch_size=1024,
                    validation_data=(test_x, test_y))
    t_train = time() - t_train

    t_test = time()
    p = model.predict(test_x, batch_size=1024)
    t_test = time() - t_test
    print('training time:', t_train, 'test time:', t_test)
    return out


def cnn_basic_model():
    model = keras.Sequential([
        layers.Conv2D(12, activation='relu', kernel_size=3, input_shape=(28, 28, 1)),
        layers.Conv2D(20, activation='relu', kernel_size=3),
        layers.Conv2D(20, activation='relu', kernel_size=3),
        layers.Flatten(),
        layers.Dense(100, activation='relu'),
        layers.Dense(10, activation='softmax'),
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    return model


def cnn_drop_model():
    model = keras.Sequential([
        layers.Conv2D(12, activation='relu', kernel_size=3, input_shape=(28, 28, 1)),
        layers.Dropout(0.5),
        layers.Conv2D(20, activation='relu', kernel_size=3),
        layers.Dropout(0.5),
        layers.Conv2D(20, activation='relu', kernel_size=3),
        layers.Dropout(0.5),
        layers.Flatten(),
        layers.Dense(100, activation='relu'),
        layers.Dense(10, activation='softmax'),
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    return model


def cnn_stride_model():
    model = keras.Sequential([
        layers.Conv2D(12, activation='relu', kernel_size=3, strides=2, input_shape=(28, 28, 1)),
        layers.Conv2D(20, activation='relu', kernel_size=3, strides=2),
        layers.Conv2D(20, activation='relu', kernel_size=3, strides=2),
        layers.Flatten(),
        layers.Dense(100, activation='relu'),
        layers.Dense(10, activation='softmax'),
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    return model


def cnn_basic(train_x, train_y, test_x, test_y):
    model = cnn_basic_model()

    t_train = time()
    out = model.fit(train_x,
                    train_y,
                    epochs=2,
                    batch_size=32,
                    validation_data=(test_x, test_y))
    t_train = time() - t_train

    t_test = time()
    model.predict(test_x, batch_size=32)
    t_test = time() - t_test
    print('train time:', t_train, 'predict time:', t_test)


def cnn_with_aug(train_x, train_y, test_x, test_y):
    data_generator_with_aug = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=False,
                                                                           width_shift_range=0.2,
                                                                           height_shift_range=0.2)
    data_generator_with_aug = data_generator_with_aug.flow(train_x, train_y, batch_size=32)

    model = cnn_basic_model()

    t_train = time()
    out = model.fit(data_generator_with_aug,
                    epochs=15,
                    validation_data=(test_x, test_y), )
    t_train = time() - t_train

    t_test = time()
    p = model.predict(test_x, batch_size=1024)
    t_test = time() - t_test
    print('training time:', t_train, 'test time:', t_test)
    return out


def cnn_with_drop(train_x, train_y, test_x, test_y):
    print('cnn with drop running..')
    model = cnn_drop_model()

    t_train = time()
    out = model.fit(train_x,
                    train_y,
                    epochs=10,
                    batch_size=32,
                    validation_data=(test_x, test_y))
    t_train = time() - t_train

    t_test = time()
    model.predict(test_x, batch_size=32)
    t_test = time() - t_test
    print('train time:', t_train, 'predict time:', t_test)


def cnn_with_stride(train_x, train_y, test_x, test_y):
    print('cnn with stride running..')
    model = cnn_stride_model()

    t_train = time()
    out = model.fit(train_x,
                    train_y,
                    epochs=2,
                    batch_size=32,
                    validation_data=(test_x, test_y))
    t_train = time() - t_train

    t_test = time()
    model.predict(test_x, batch_size=32)
    t_test = time() - t_test
    print('train time:', t_train, 'predict time:', t_test)
