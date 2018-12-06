from globalSetting import *
from utils import *

def loadData(benchName):
    trainSrc = np.loadtxt(dataDir+benchName+'/train.x')
    trainTgt = np.loadtxt(dataDir+benchName+'/train.y')
    testSrc = np.loadtxt(dataDir+benchName+'/test.x')
    testTgt = np.loadtxt(dataDir+benchName+'/test.x')
    return tensorData(trainSrc), tensorData(trainTgt),\
           tensorData(testSrc), tensorData(testTgt)

startx = 0
starty = 0
# src==1: src, src==0 tgtMinibatch
def miniBatch(tens, batchSize, src):
    global startx, starty
    start = (startx if (src==1) else starty)
    start+=batchSize
    if (start>=len(tens)-(batchSize/2)):
            start+= (batchSize/2)-len(tens)
    if (src==1): startx = start
    else: starty = start
    return tens[start:batchSize+start]

if (__name__=="__main__"):
    print ("test data.py")
    a,b,c,d = loadData(benchName)

