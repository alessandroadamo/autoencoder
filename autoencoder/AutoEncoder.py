import logging, inspect
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError

import numpy as np


class AutoEncoder(BaseEstimator, TransformerMixin):
    """
    AutoEncoder feature learning and extraction
    """

    def __init__(self, n_encoding=None,
                 random_state=None,
                 activation='linear',
                 loss_function='l2',
                 l1_penalty=None,
                 l2_penalty=None,
                 beta=None,
                 rho=None,
                 dropout_keep_prob=1.0,
                 train_epochs=1000,
                 batch_size=128,
                 learning_rate=1e-3):
        """
        Initialize the AutoEncoder

        :param n_encoding Number of encoding dimensions. If n_encoding is not set all components are setted as
        ```python
        n_encoding = min(n_samples, n_features)
        ```
        :param random_state RandomState, optional, default None
        :param activation Activation function, optional, default linear. The activation function must be **linear**, **relu**,
        **sigmoid** or **tanh**.
        :param loss_function Loss function, default = 'l2',
        :param l1_penalty $\ell_1$ penalty coefficient, optional, default None
        :param l2_penalty $\ell_2$ penalty coefficient, optional, default None
        :param beta sparsity coefficient, optional, default None
        :param rho sparsity coefficient, optional, default None
        :param dropout_keep_prob dropdown probability, optional, default None
        :param train_epochs Train epoch, optional, default 1000.
        :param batch_size Batch size, optional, default 64
        :param learning_rate Learning rate, optional, default 1e-3
        """
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)

        self.costs_ = []
        self.b1_ = None
        self.b2_ = None
        self.H_ = None

        self._sess = None
        self._encode = None
        self._decode = None
        self._x = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._sess is not None:
            self._sess.close()

    def fit(self, X):
        """
        Fit the model with X.
        :param X: array-like, shape (n_samples, n_features). Training data, where n_samples in the number of samples and n_features is the number of features.
        :return: Returns the instance itself.

        """
        assert self.activation in ['linear', 'sigmoid', 'tanh', 'relu'], \
            "activation parameter must be linear or sigmoid"
        assert self.random_state is None or (type(self.random_state) == int), \
            "random_state parameter must be None or integer"
        assert type(self.train_epochs) == int and self.train_epochs > 0, \
            "train_epochs parameter must be integer greater than 0"
        assert type(self.batch_size) == int or self.batch_size > 0, \
            "batch_size parameter must be integer greater than 0"
        assert type(self.learning_rate) == float or self.learning_rate > 0.0, \
            "learning_rate parameter must be float greater than 0"
        assert self.loss_function in ['l1', 'l2'], \
            "loss_function parameter must be l1 or l2"
        assert self.l1_penalty is None or self.l1_penalty > 0.0, \
            "l1_penalty parameter must be None or greater than 0.0"
        assert self.l2_penalty is None or self.l2_penalty > 0.0, \
            "l2_penalty parameter must be None or greater than 0.0"
        assert (self.beta is None and self.rho in not None) or (self.beta is not None and self.rho is None), \
            "beta and rho must be both None or real"
        assert self.beta is None or self.rho > 0.0, \
            "beta parameter must be None or greater than 0.0"
        assert self.rho is None or (0.0 < self.rho < 1.0), \
            "rho parameter must be None or in the interval (0.0, 1.0)"
        assert 0.0 < self.dropout_keep_prob <= 1.0, \
            "dropout_keep_prob parameter must be greater than 0.0 and less then 1.0"

        n_samples, n_features = X.shape
        assert self.n_encoding is None or self.n_encoding > 0, \
            "n_encoding parameter must be None or integer greater than 0"

        if self.n_encoding is None:
            self.n_encoding = min([n_samples, n_features])

        # set the random seed
        if self.random_state is not None:
            tf.set_random_seed(self.random_state)

        # tf Graph input (only pictures)
        x = tf.placeholder(tf.float32, [None, n_features])
        self._x = x

        # encode
        low = -4.0 * np.sqrt(6.0 / (n_features + self.n_encoding))  # use 4 for sigmoid, 1 for tanh activation
        high = 4.0 * np.sqrt(6.0 / (n_features + self.n_encoding))
        H = tf.Variable(tf.random_uniform([n_features, self.n_encoding], minval=low, maxval=high, dtype=tf.float32))

        b1 = tf.Variable(tf.zeros([self.n_encoding]))

        # decode
        b2 = tf.Variable(tf.zeros([n_features]))

        # define the model
        encode = {
            'linear': tf.add(tf.matmul(x, H), b1),
            'relu': tf.nn.relu(tf.add(tf.matmul(x, H), b1)),
            'tanh': tf.nn.tanh(tf.add(tf.matmul(x, H), b1)),
            'sigmoid': tf.nn.sigmoid(tf.add(tf.matmul(x, H), b1))
        }[self.activation]

        # dropout
        encode = tf.nn.dropout(encode, self.dropout_keep_prob)
        decode = tf.add(tf.matmul(encode, tf.transpose(H, name='H_t')), b2)

        self._encode = encode
        self._decode = decode

        y_pred = decode
        y_true = x

        # define loss and optimizer
        loss = {
            'l1': tf.reduce_sum(tf.abs(tf.subtract(y_true, y_pred))),
            'l2': tf.reduce_sum(tf.square(tf.subtract(y_true, y_pred))),
        }[self.loss_function]

        # regularization
        if self.l1_penalty is not None:
            loss += self.l1_penalty * tf.reduce_sum(tf.abs(H))

        if self.l2_penalty is not None:
            loss += self.l2_penalty * tf.reduce_sum(tf.square(H))

        # sparsity
        if self.beta is not None:
            loss += self.beta * tf.reduce_sum(self.rho * tf.log(self.rho  / (encode + 1e-10)) +
                                  (1.0 - self.rho) * tf.log((1.0 - self.rho) / (encode + 1e-10)))

        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        # initializing the variables
        init = tf.global_variables_initializer()

        # launch the graph
        self.costs_ = []
        self._sess = tf.Session()

        self._sess.run(init)
        total_batch = int(n_samples / self.batch_size)
        # training cycle
        for epoch in range(self.train_epochs):
            # Loop over all batches
            for i in range(total_batch):
                batch_x = X[(i * self.batch_size):((i + 1) * self.batch_size), :]
                _, c = self._sess.run([optimizer, loss], feed_dict={x: batch_x})

            self.costs_.append(c)
            # display logs per epoch step
            if epoch % 100 == 0:
                logging.info('Epoch: {:6d} cost= {:.9f}'.format(epoch + 1, c))
                print('Epoch: {:6d} cost= {:.9f}'.format(epoch + 1, c))

        logging.info('Epoch: {:6d} cost= {:.9f}'.format(epoch + 1, c))
        print('Epoch: {:6d} cost= {:.9f}'.format(epoch + 1, c))

        self.b1_ = self._sess.run(b1)
        self.b2_ = self._sess.run(b2)
        self.H_ = self._sess.run(H)

        return self

    def transform(self, X, y=None):
        """
        Apply dimensionality reduction to X.
        :param X: array-like, shape (n_samples, n_features). Training data, where n_samples is the number of samples
        and n_features is the number of features.
        :return: array-like, shape (n_samples, n_components)
        """
        if self._sess is None:
            raise NotFittedError("AutoEncoder not fitted yet.")

        X_enc = self._sess.run(self._encode, feed_dict={self._x: X})

        return X_enc

    def fit_transform(self, X, y=None, **kwargs):
        """
        Fit the model with X and apply the dimensionality reduction on X.
        :param X: array-like, shape (n_samples, n_features). Training data, where n_samples is the number of samples
        and n_features is the number of features.
        :return: array-like, shape (n_samples, n_components)
        """
        if self._sess is None:
            self.fit(X)

        return self.transform(X)

    def score(self, X, y=None):
        #return (sum(self.predict(X)))
        return 0.0

if __name__ == '__main__':
    from sklearn.datasets import fetch_mldata
    from sklearn.preprocessing import MinMaxScaler

    mnist = fetch_mldata("MNIST original")
    # rescale the data, use the traditional train/test split
    X, y = mnist.data / 255., mnist.target
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    scaler = MinMaxScaler([0.0, 1.0])
    X_sc = scaler.fit_transform(X_train)

    ae = AutoEncoder(n_encoding=30,
                     train_epochs=20,
                     random_state=123,
                     activation='sigmoid',
                     loss_function='l2',
                     # l1_penalty=0.001,
                     # l2_penalty=0.001,
                     learning_rate=0.01,
                     dropout_keep_prob=1.0)
    ae.fit(X_sc)
