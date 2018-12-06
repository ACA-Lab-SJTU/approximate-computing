from keras.models import Sequential,model_from_json
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils, generic_utils
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


def ClassiferModel(net_list):
    if len(net_list) < 2:
        print 'ERROR! input net structrue is wrong!'
        exit(0)

    model = Sequential()

    model.add(Dense(net_list[1], input_dim=net_list[0], init='uniform'))
    
    if len(net_list) > 2:
        model.add(Activation('sigmoid'))
        for i in net_list[2:-1]:
            model.add(Dense(i, init='uniform'))
            model.add(Activation('sigmoid'))
        model.add(Dense(net_list[-1], init='uniform'))
    
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, class_mode='categorical')
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


def gen_accept(re_bound):
    def accept(v0, v1):
        return error.relative_error(v0, v1) <= re_bound
    return accept

# add wxy
def choose_flag(shrink_type):
    return [lambda a, c: a or c,
            lambda a, c: c,
            lambda a, c: a,
            lambda a, c: a and c,
            lambda a, c: 1][shrink_type]


def choose_flag_old(type):
    if type == 0:
        def choose(a, c):
            return a or c
        return choose
    elif type == 1:
        def choose(a, c):
            return c
        return choose
    elif type == 2:
        def choose(a, c):
            return a
        return choose
    elif type == 3:
        def choose(a, c):
            return a and c
        return choose
    else:
        print 'ERROR! wrong choose_type!'
        exit(0)


def gen_evaluate(X, Y, accept):
    N = len(X)
    def evaluate(A, C):
        # accelerator results 
        acc = A.predict(X)
        # classification by C and truly 
        cls_c = [1 if v[1] > v[0] else 0 for v in C.predict(X)]
        cls_t = [accept(Y[i], acc[i]) for i in xrange(N)] # to bool
        # relatvie error for all test data
        re = [error.relative_error(Y[i], acc[i]) for i in xrange(N)]
        re_c = [error.relative_error(Y[i], acc[i]) for i in xrange(N) if cls_c[i]]

        # accuracy of C, recall of C
        accuracy_of_C = sum([1.0 if cls_t[i] == cls_c[i] else 0 for i in xrange(N)]) / float(1e-10 + N)
        recall_of_C = sum([1.0 if cls_t[i] and cls_c[i] else 0 for i in xrange(N)]) / float(1e-10 + sum([1 if v else 0 for v in cls_t]))

        # invocation of C, invocation truly
        invocation_of_C = float(sum([1 if v else 0 for v in cls_c])) / float(1e-10 + N)
        invocation_truly = float(sum([1 if v else 0 for v in cls_t])) / float(1e-10 + N)

        # re of A, re of A with C
        mean_relative_error_of_A = sum(re) / float(1e-10 + len(re))
        mean_relative_error_of_A_with_C = sum(re_c) / (1e-10 + len(re_c))

        return {
            'accuracy_of_C': accuracy_of_C,
            'recall_of_C': recall_of_C,
            'invocation_of_C': invocation_of_C,
            'invocation_truly': invocation_truly,
            'mean_relative_error_of_A': mean_relative_error_of_A,
            'mean_relative_error_of_A_with_C': mean_relative_error_of_A_with_C
        }

    return evaluate



#get_output_name(app_name, error_bound, choose_type, epochA, epochC, batch_sizeA, batch_sizeC, net_A, net_C)
def get_output_name(app_name, error_bound, choose_type, epochA, epochC, batch_sizeA, batch_sizeC, net_A, net_C):
    def get_name(iteration):
        output_name = '{}_it{}_eb{}_ct{}_epA{}_epC{}_bsA{}_bsC{}_netA{}_netC{}'.format(app_name, iteration, error_bound, choose_type, epochA, epochC, batch_sizeA, batch_sizeC, '_'.join([str(x) for x in net_A]), '_'.join([str(x) for x in net_C]))
        return output_name
    return get_name


#train_iteration(A, C, X0, Y0, X1, Y1, iteration, error_bound, choose_type, epochA, epochC, batch_sizeA, batch_sizeC, net_A, net_C, get_name)
def train_iteration(A, C, X0, Y0, X1, Y1, iteration, error_bound, choose_type, epochA, epochC, batch_sizeA, batch_sizeC, net_A, net_C, get_name):
    print 'start training'
    accept = gen_accept(error_bound)
    evaluate = gen_evaluate(X1, Y1, accept)
    choose = choose_flag(choose_type)

    X_origin = X0.copy()
    Y_origin = Y0.copy()
    X_now = X0.copy()
    Y_now = Y0.copy()

    f_results = open('../results/iteration/{}.csv'.format(get_name(iteration)), 'w')

    keys = []
    results = []

    for index in range(iteration):
        if len(X_now) == 0:
            print 'No training data, end!'
            break

        # train accelerator with the current data(X0, Y0)
        A.fit(X_now, Y_now, nb_epoch=epochA, batch_size=batch_sizeA, verbose=2)

        # get the result of accelerator
        acc = A.predict(X_origin)

        # generate the truly classification
        cls_t = [accept(Y_origin[i], acc[i]) for i in xrange(len(X_origin))]

        print len([1 for i in cls_t if i])

        # train the Classifer with X_origin, cls_t
        Y_cls = np.array([1 if v else 0 for v in cls_t]).reshape((len(cls_t), 1))
        Y_cls = np_utils.to_categorical(Y_cls, 2)
        
        C.fit(X_origin, Y_cls, nb_epoch=epochC, batch_size=batch_sizeC, show_accuracy=True, verbose=2)

        # get the classifer result of origin data
        cls_c = [1 if v[1] > v[0] else 0 for v in C.predict(X_origin)]
        print len([1 for i in cls_c if i])

        # generate the next X_now, Y_now

        X_now = np.array([X_origin[i] for i in xrange(len(X_origin)) if choose(cls_t[i], cls_c[i])])
        Y_now = np.array([Y_origin[i] for i in xrange(len(X_origin)) if choose(cls_t[i], cls_c[i])])

        # save weights
        A.save_weights('../weights/iteration/A_{}.weights'.format(get_name(index)), overwrite=True)
        C.save_weights('../weights/iteration/C_{}.weights'.format(get_name(index)), overwrite=True)

        # save this iteration results
        item = {'iteration':index}
        item.update(evaluate(A, C))
        print item
        if len(results) == 0:
            keys = item.keys()
            f_results.write(','.join(keys) + '\n')
        results.append(item)
        f_results.write(','.join([str(item[v]) for v in keys]) + '\n')
        f_results.flush()

    f_results.close()

    return results

'''
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
'''



#[benchmark_name] [iteration] [error_bound] [choose_type] [epochA] [epochC] [batch_sizeA] [batch_sizeC] [net_A] [net_C]
def main(app_name, iteration, error_bound, choose_type, epochA, epochC, batch_sizeA, batch_sizeC, net_A, net_C):
    X0, Y0, X1, Y1 = load_data(app_name)

    A = AcceleratorModel(net_A)

    C = ClassiferModel(net_C)

    get_name = get_output_name(app_name, error_bound, choose_type, epochA, epochC, batch_sizeA, batch_sizeC, net_A, net_C)

    print get_name(iteration)
    
    train_iteration(A, C, X0, Y0, X1, Y1, iteration, error_bound, choose_type, epochA, epochC, batch_sizeA, batch_sizeC, net_A, net_C, get_name)

if __name__ == "__main__":
    if len(sys.argv) == 11:
        net_A = [int(x) for x in sys.argv[9].split('_')[1:]]
        net_C = [int(x) for x in sys.argv[10].split('_')[1:]]
        print net_A
        print net_C
        main(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8]), net_A, net_C)
    else:
        print 'Usage: python train_origin.py [benchmark_name] [iteration] [error_bound] [choose_type] [epochA] [epochC] [batch_sizeA] [batch_sizeC] [net_A] [net_C]'
        print '#choose_type: 0|1|2|3|4, [0: A or C] | [1: onlyC] | [2: onlyA] | [3: A and C] | [4: true]'
        print '#net_A|net_C: like a_6_8_8_1, c_6_8_2'
        exit(0)


