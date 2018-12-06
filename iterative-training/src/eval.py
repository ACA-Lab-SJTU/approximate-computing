from keras.models import Sequential,model_from_json
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, RMSprop
import error
import math
import numpy as np
import json
import sys

 
def AcceleratorModel(net_list):

    if len(net_list) < 2:
        print 'ERROR! input net structrue is wrong!'
        exit(0)

    model = Sequential()

    model.add(Dense(net_list[1], input_dim=net_list[0], init='uniform'))
    model.add(Activation('sigmoid'))

    for i in net_list[2:]:
        model.add(Dense(i, init='uniform'))
        model.add(Activation('sigmoid'))

    # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    prop = RMSprop(lr=0.01)
    model.compile(loss='mse', optimizer=prop)
    return model


def format_data(data):
    try:
        return data.reshape((data.shape[0], 1)) if len(data.shape) == 1 else data
    except AttributeError as e:
        print 'ERROR! data is not a numpy object, format_data failed!'
        exit(0)
        
def load_data(app_name):
    X0 = np.loadtxt('../data/' + app_name + '/train.x')
    Y0 = np.loadtxt('../data/' + app_name + '/train.y')
    X1 = np.loadtxt('../data/' + app_name + '/test.x')
    Y1 = np.loadtxt('../data/' + app_name + '/test.y')
    return format_data(X0), format_data(Y0), format_data(X1), format_data(Y1)

def main(benchmark, weights_file):
    X0, Y0, X1, Y1 = load_data(benchmark)
    A = AcceleratorModel([1, 4, 4, 2])
    A.load_weights(weights_file)

    Y2 = A.predict(X1)
    print A.evaluate(X1, Y1);

    re = [error.relative_error(Y1[i], Y2[i]) for i in xrange(len(X1))]
    count = [0] * 101
    for v in re:
        count[int(math.floor(v * 100))] += 1
    s = sum(count)
    count = [float(v) / s for v in count]
    for i in range(101):
        print i, count[i]

    for i in xrange(len(X1)):
        print Y1[i], Y2[i], error.relative_error(Y1[i], Y2[i])





if __name__ == "__main__":
    if len(sys.argv) != 3:
        print 'usage: python eval.py [benchmark name] [weights file]'
    else:
        main(sys.argv[1], sys.argv[2])
