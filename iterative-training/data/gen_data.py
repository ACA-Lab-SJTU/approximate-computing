import sys

def gen_file(directory, in_name, out_name):
    fin = open(directory + '/' + in_name, 'r')
    X, Y = [], []
    flag = True
    for line in fin.readlines()[1:]:
        if flag:
            X.append(line)
        else:
            Y.append(line)
        flag = not flag

    with open(directory + '/' + out_name + '.x', 'w') as f:
        for x in X:
            f.write(x)

    with open(directory + '/' + out_name + '.y', 'w') as f:
        for y in Y:
            f.write(y)


def main(directory):
    gen_file(directory, 'aggregated_test.fann', 'test')
    gen_file(directory, 'aggregated_train.fann', 'train')



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print 'usage: python gen_data.py [directory]'
    else:
        main(str(sys.argv[1]))

    

