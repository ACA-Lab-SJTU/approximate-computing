import tensorflow as tf


def relative_error(origin, prediction):
    err = origin - prediction
    return tf.abs(tf.div(err, origin + 1e-11))


def absolute_error(origin, prediction):
    return tf.abs(origin - prediction)

def image_diff(origin, prediction):
    return tf.sqrt(tf.reduce_mean(tf.square(origin - prediction), axis=1, keep_dims=True))

def miss_rate(origin, prediction, error_bound):
    err = absolute_error(origin, prediction)
    return tf.less(err, tf.constant(error_bound, shape=err.shape))


def error_func(benchmark):
    func = {
        'blackscholes': relative_error,
        'fft': absolute_error,
        'jpeg': image_diff,
        'kmeans': image_diff,
        'sobel': image_diff,
        'inversek2j': relative_error,
        'bessel_Jnu': absolute_error,
        'jmeint': relative_error

    }
    return func[benchmark]