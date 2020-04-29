from tensorflow import keras
import model

if __name__ == '__main__':
    # load data
    # x: (n, 28, 28)  y: (n,)
    (train_images, train_labels), (test_images,
                                   test_labels) = keras.datasets.mnist.load_data()
    n_train, n_test = train_images.shape[0], test_images.shape[0]

    # flatten data
    train_x_flatten = train_images.reshape((n_train, -1))
    test_x_flatten = test_images.reshape((n_test, -1))

    # run model
    # traditional model
    model.knn(train_x_flatten, train_labels, test_x_flatten, test_labels, 3)
    model.svm(train_x_flatten, train_labels, test_x_flatten, test_labels)
    linear_classifier = model.linear_classifier(train_x_flatten, train_labels, test_x_flatten, test_labels)

    # cnn
    cnn_basic = model.cnn_basic(train_images.reshape(-1, 28, 28, 1), train_labels,
                                test_images.reshape(-1, 28, 28, 1), test_labels)

    cnn_with_aug = model.cnn_with_aug(train_images.reshape(-1, 28, 28, 1), train_labels,
                                      test_images.reshape(-1, 28, 28, 1), test_labels)
    cnn_with_drop = model.cnn_with_drop(train_images.reshape(-1, 28, 28, 1), train_labels,
                                        test_images.reshape(-1, 28, 28, 1), test_labels)
    cnn_with_stride = model.cnn_with_stride(train_images.reshape(-1, 28, 28, 1), train_labels,
                                            test_images.reshape(-1, 28, 28, 1), test_labels)
