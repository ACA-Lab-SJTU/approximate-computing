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


def get_output_name(app_name, epoch, batch_size, net_list):
    output_name = '{}_ep{}_bs{}_net{}'.format(app_name, epoch, batch_size, '->'.join([str(x) for x in net_list]))
    return output_name


def keras_to_fann(A, fann_sample, fann_output):
    sample_lines = []

    #read fann sample file.
    with open(fann_sample, 'r') as sample:
        sample_lines = sample.readlines()

    #transfer keras model to fann model.
    with open(fann_output, 'w') as f:

        #write 1st line. (unchanged)
        f.write(sample_lines[0])

        #write 2nd line. (num_layers)
        layers = [l for l in A.layers if len(l.get_weights()) > 0]
        f.write('num_layers=')
        f.write(str(len(layers)+1) + '\n')

        #write 3rd-32nd line. (unchanged)
        for line in sample_lines[2:32]:
            f.write(line)

        #write 33rd line. (layer_sizes)
        f.write('layer_sizes=')
        layer_sizes = []
        for layer in layers:
            layer_sizes.append(layer.get_weights()[0].shape[0] + 1)
        layer_sizes.append(layers[len(layers)-1].get_weights()[0].shape[1] + 1)
        print layer_sizes
        f.write(' '.join([str(v) for v in layer_sizes]) + ' \n')

        #write 34th line. (unchanged)
        f.write(sample_lines[33])

        #write 35th line. (neurons)
        f.write('neurons (num_inputs, activation_function, activation_steepness)=')
        for i in range(layer_sizes[0]):
            f.write('(0, 0, 0.00000000000000000000e+00) ')
        for ls in range(len(layer_sizes[1:])):
            for i in range(layer_sizes[ls+1]):
                f.write('({}, 3, 5.00000000000000000000e-01) '.format(layer_sizes[ls]) if i < (layer_sizes[ls+1] - 1) else '(0, 3, 0.00000000000000000000e+00) ')
        f.write('\n')

        #write 36th line. (connections)
        f.write('connections (connected_to_neuron, weight)=')
        index = 0
        for layer in layers:
            weights = layer.get_weights()
            w = weights[0].shape[1]
            h = weights[0].shape[0]
            for i in range(w):
                for j in range(h):
                    f.write('({}, {:.20e}) '.format(index+j, weights[0][j][i]))
                f.write('({}, {:.20e}) '.format(index+h, weights[1][i]))
            index = index + h + 1 
        f.write('\n')


def train_origin(A, X0, Y0, X1, Y1, epoch, batch_size, output_name):

    A.fit(X0, Y0, nb_epoch=epoch, batch_size=batch_size)
    acc = A.predict(X1)

    result_num = [0 for i in range(100)]
    result_sum = 0
    for i in range(len(X1)):
        rate = 0.01
        tmp_re = error.relative_error(Y1[i], acc[i])
        result_sum += tmp_re
        for j in range(100):
            if tmp_re <= rate:
                result_num[j] += 1
            rate += 0.01

    result_mre = result_sum / float(len(X1))
    result_re_list = [(100.0 * result_num[x] / float(len(X1))) for x in range(len(result_num))]

    print result_mre
    print result_re_list

    f_results = open('../results/origin/{}.csv'.format(output_name), 'w')
    f_results.write('mre,{}\n'.format(result_mre))
    f_results.write('re_bound,re_percent\n')
    for i in range(100):
        f_results.write('{}%,{:.3f}%\n'.format(i + 1, result_re_list[i]))
        f_results.flush()
    f_results.close()
    A.save_weights('../weights/origin/{}.weights'.format(output_name), overwrite=True)
    keras_to_fann(A, 'fann_sample.nn', '../fann_model/{}.nn'.format(output_name))




    '''
    accept = gen_accept(error_bound)
    evaluate = gen_evaluate(X1, Y1, accept)

    X_origin = X0.copy()
    Y_origin = Y0.copy()
    X_now = X0.copy()
    Y_now = Y0.copy()


    
    f_results = open('../results/origin/eb{}_epochA{}_epochC{}.csv'.format(error_bound, epochA, epochC), 'w')
    item = {'iteration':0}
    item.update(evaluate(A, C))
    keys = item.keys()
    f_results.write(','.join(keys) + '\n')
    f_results.write(','.join([str(item[v]) for v in keys]) + '\n')
    
    results = [item]

    for index in range(1, iteration):
        # train accelerator with the current data(X0, Y0)
        A.fit(X_now, Y_now, nb_epoch=epochA, batch_size=batch_size)

        # get the result of accelerator
        acc = A.predict(X_origin)

        # generate the truly classification
        cls_t = [accept(Y_origin[i], acc[i]) for i in xrange(len(X_origin))]

        # train the Classifer with X_origin, cls_t
        Y_cls = np.array([1 if v else 0 for v in cls_t]).reshape((len(cls_t), 1))
        C.fit(X_origin, Y_cls, nb_epoch=epochC, batch_size=batch_size, show_accuracy=True)

        # get the classifer result of origin data
        cls_c = [round(v[0]) == 1 for v in C.predict(X_origin)]

        # X_now, Y_now is still X_origin, Y_origin
        # X_now = np.array([X_origin[i] for i in xrange(len(X_origin)) if cls_c[i] and cls_t[i]])
        # Y_now = np.array([Y_origin[i] for i in xrange(len(X_origin)) if cls_c[i] and cls_t[i]])
       
        item = {'iteration':index}
        item.update(evaluate(A, C))
        print item
        f_results.write(','.join([str(item[v]) for v in keys]) + '\n')
        f_results.flush()
        results.append(item)

    f_results.close()

    return results
    '''


def main(app_name, epoch, batch_size, net_list):
    X0, Y0, X1, Y1 = load_data(app_name)

    A = AcceleratorModel(net_list)

    output_name = get_output_name(app_name, epoch, batch_size, net_list)

    print output_name
    
    train_origin(A, X0, Y0, X1, Y1, epoch, batch_size, output_name)

   

'''
def test_bacthsize(epoch, batch_size):
    def evaluate(X, Y, A):
        acc = A.predict(X)
        re = [error.relative_error(Y[i], acc[i]) for i in xrange(len(X))]
        mre = np.mean(re)
        error_count = [0] * 101
        for e in re:
            i = int(e * 100.0)
            error_count[i] += 1
        return {'mre':mre, 'error_count':error_count}

    X0, Y0, X1, Y1 = load_data()
    A = AcceleratorModel()
    
    for index in xrange(epoch/50):
        A.fit(X0, Y0, nb_epoch=50, batch_size=batch_size, verbose=0)
        r0 = evaluate(X0, Y0, A)
        r1 = evaluate(X1, Y1, A)
        item = {'epoch':(index+1)*50, 'r0':r0, 'r1':r1}
        print json.dumps(item)
        

def test():
    X0, Y0, X1, Y1 = load_data()
    A = AcceleratorModel()
    C = ClassiferModel()

    error_bound = 0.05
    iteration = 100
    epochA = 100
    epochC = 2
    batch_size = 1024 

    X_origin = X0.copy()
    Y_origin = Y0.copy()

    for index in range(1, iteration):
        # train accelerator with the current data(X0, Y0)
        A.fit(X_now, Y_now, nb_epoch=epochA, batch_size=batch_size, verbose=0)

        # get the result of accelerator
        acc = A.predict(X_origin)

        # generate the truly classification
        cls_t = [accept(Y_origin[i], acc[i]) for i in xrange(len(X_origin))]
        re = [error.relative_error(Y_origin[i], acc[i]) for i in xrange(len(X_origin))]

        # train the Classifer with X_origin, cls_t
        Y_cls = np.array([1 if v else 0 for v in cls_t]).reshape((len(cls_t), 1))
        print 'iteration {}, train epoch {}:'.format(index, index*epochA)
        print 'truly invocation :', sum(Y_cls) / float(len(Y_cls))
        print 'mean relative error ', sum(re) / float(len(re))
        print 'relative error in [0   - 2.5%] : {:.3f}%'.format(
                100.0 * len([1 for v in re if v <= 0.025]) / len(re)
        )
        print 'relative error in [2.5% -  5%] : {:.3f}%'.format(
                100.0 * len([1 for v in re if 0.025 < v and v <= 0.05]) / len(re)
        )
        print 'relative error in [5%   - 10%] : {:.3f}%'.format(
                100.0 * len([1 for v in re if 0.05 < v and v <= 0.1]) / len(re)
        )
        print 'relative error in [10%  - 20%] : {:.3f}%'.format(
                100.0 * len([1 for v in re if 0.1 < v and v <= 0.2]) / len(re)
        )
        print 'relative error in [20%  -    ] : {:.3f}%'.format(
                100.0 * len([1 for v in re if 0.2 < v]) / len(re)
        )
        print

'''




if __name__ == "__main__":
    if len(sys.argv) > 5:
        net_list = []
        for i in sys.argv[4:]:
            net_list.append(int(i))
        main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), net_list)
    else:
        print 'Usage: python train_origin.py [benchmark_name] [epoch] [batch_size] [net_struct]'
        exit(0)


