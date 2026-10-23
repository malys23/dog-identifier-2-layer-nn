import numpy as np
from build_model import build_model

def main():
    train_set_x_orig = np.load('./data/train_dogvnondog/train_set_x.npy')
    train_set_y = np.load('./data/train_dogvnondog/train_set_y.npy')
    test_set_x_orig = np.load('./data/test_dogvnondog/test_set_x.npy')
    test_set_y = np.load('./data/test_dogvnondog/test_set_y.npy')
    classes = np.load('./data/train_dogvnondog/list_classes.npy')

    #1: Resizing and find values for m_train, m_test, and num_px
    m_train = train_set_x_orig.shape[0]
    m_test = test_set_x_orig.shape[0]
    num_px = train_set_x_orig.shape[1]
    train_y = train_set_y.reshape(1, m_train)
    test_y = test_set_y.reshape(1, m_test)

    #2: Reshape data sets so images are flattened
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

    #3: Standardize dataset (divide by 255, max val of pixel channel)
    train_set_x = train_set_x_flatten / 255
    test_set_x = test_set_x_flatten / 255

    #4: Run Two Layer Model
    n_x = 12288
    n_h = 7
    n_y = 1
    layers_dims=(n_x, n_h, n_y)
    learning_rate=0.0075
    num_iterations=300
    two_layer_model = build_model(train_set_x, train_y, test_set_x, test_y, layers_dims=layers_dims, learning_rate=learning_rate, num_iterations=num_iterations, print_cost=True)

if __name__ == "__main__":
    main()
