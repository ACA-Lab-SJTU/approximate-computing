import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def format_data(data):
    try:
        return data.reshape((data.shape[0], 1)) if len(data.shape) == 1 else data
    except AttributeError as e:
        print('ERROR! data is not a numpy object, format_data failed!')
        exit(0)

def draw(data_list, label_list, label, dir):
    assert len(data_list)==len(label_list)
    plt.figure(0, figsize=(16,9))
    for i in range(len(data_list)):
        plt.plot(data_list[i], label=label_list[i])
    plt.legend()
    plt.title(label)
    plt.savefig(dir + '/' + label + '.png', dpi=200)
    plt.clf()
    plt.close(0)
        

def load_data(app_name):
    X0 = np.loadtxt('./data/' + app_name + '/train.x')
    Y0 = np.loadtxt('./data/' + app_name + '/train.y')
    X1 = np.loadtxt('./data/' + app_name + '/test.x')
    Y1 = np.loadtxt('./data/' + app_name + '/test.y')
    return format_data(X0), format_data(Y0), format_data(X1), format_data(Y1)