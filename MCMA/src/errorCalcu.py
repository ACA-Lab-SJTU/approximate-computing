#from globalSetting import *
import numpy as np

def er(pred, tar):
    absErr = np.absolute(pred-tar)
    absTar = np.absolute(tar)
    if (absTar<0.00000001): return 1.0
    return min(1.0,float(absErr)/absTar)

def mre(predict, target):
    return np.array([er(x[0],x[1]) for x in zip(predict,target)]).mean()

def mae(predict, target):
    return np.array([np.absolute(x[0]-x[1])for x in zip(predict,target)]).mean()

def rmse(predict, target):
    return np.sqrt(((np.array(target)-np.array(predict))**2).mean())

if (__name__=="__main__"):
    print ("Error test")
    print (mre([1,1],[0.1,1]))
