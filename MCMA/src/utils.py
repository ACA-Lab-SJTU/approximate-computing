from globalSetting import *
import errorCalcu

def npData(data):
    return (data.reshape((data.shape[0],1)) if (len(data.shape)==1) else data)
def tensorData(dataLst):
    npLst = npData(dataLst)
    return torch.tensor(npLst,dtype=torch.float)

def timeSince(startTime):
    return (time.time()-startTime)
def toCate(lst, classes):
    return np.eye(numA,dtype='uint8')
def cateFromOutput(output):
    topN,topI = output.topk(1)
    categoryI = topI[0].item()
    return categoryI
def softmax2num(softTens):
    return np.array([cateFromOutput(softTens[i]) for i in range(softTens.size[0])])

def errorFunc(errorType):
    if (errorType == "meanRelativeError"): return error.mre
    if (errorType == "meanAbsoluteError"): return error.mae
    if (errorType == "rootMeanSquareError"): return error.rmse

def getNetStructure(benchName,numA):
    netA = []
    netC = []
    if (benchName=='fft'):
        netA=[1,2,2,2]
        netC=[1,2,numA+1]
    if (benchName=='bessel_Jnu'):
        netA=[2,4,4,1]
        netC=[2,4,numA+1]
    if (benchName=='blackscholes'):
        netA=[6,8,1]
        netC=[6,8,numA+1]
    if (benchName=='jmeint'):
        netA=[18,32,16,2]
        netC=[18,16,numA+1]
    if (benchName=='jpeg'):
        netA=[64,16,64]
        netC=[64,18,numA+1]
    if (benchName=='inversek2j'):
        netA=[2,8,2]
        netC=[2,8,numA+1]
    if (benchName=='sobel'):
        netA=[9,8,1]
        netC=[9,8,numA+1]
    if (benchName=='kmeans'):
        netA=[6,8,4,1]
        netC=[6,8,4,numA+1]
    return netA,netC

if (__name__=="__main__"):
    print ("utils test")
    t,_,_,_ = loadData("bessel_Jnu")

