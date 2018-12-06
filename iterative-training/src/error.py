import math
import numpy

def mean(L):
    return sum(L) / float(len(L))

def relative_error(origin, prediction):
    def re(orig, pred):
        a = abs(orig - pred)
        b = abs(orig)
        if b == 0:
            e = 1.0
        elif math.isnan(a) or math.isnan(b):
            e = 1.0
        elif a/b > 1:
            e = 1.0
        else:
            e = a/b
        return e
    return mean([re(origin[i], prediction[i]) for i in range(len(origin))])

def absolute_error(origin, prediction):
    def re(orig, pred):
        a = abs(orig - pred)
        return a
    return mean([re(origin[i], prediction[i]) for i in range(len(origin))])
        




