from globalSetting import *
from initiation import *
from utils import *
from data import *
from model import *
from errorCalcu import *

errMeasure = rmse

def parserSetting():
    parser = argparse.ArgumentParser(description="--bench benchName")
    parser.add_argument(
            "--bench",
            type = str,
            nargs = 1,
            default = 'bessel_Jnu',
            help = 'Check ../data/* for all the benchmarks.'
            )
    args = parser.parse_args()
    benchName = args.bench[0]
    return args

def optiModel(model, src, tgt):
    predict = model(src)
    model.optimizer.zero_grad()
    loss = model.criterion(predict, tgt)
    loss.backward()
    model.optimizer.step()

def modelDeduct(model,src):
    with model.no_grad():
        return model(src)

def getLabel(pred,tgt,eb):
    label = []
    for i in range(len(tgt)):
        label.append[0]
        for j in range(numA):
            if (errMeasure(pred[j][i],tgt[i])<eb):
                label[i]=j+1
                break
    label = tensorData(toCate(label,numA))
    return label

def predByAs(A,src):
    pred = []
    for i in range(len(A)):
        pred.append(modelDeduct(A[i].src))
    return pred

def trainAs(A,src,tgt):
    for i in range(len(A)):
        if (src[i].shape[0]>0):
            optiModel(A[i],src[i],tgt[i])

def loadTrainParas():
    global epochA, epochC, batchsize, iterNum, dataType, eb
    epochA = c['train']['epochA']
    epochC = c['train']['epochC']
    batchsize = c['train']['batchsize']
    iterNum = c['train']['iteration']
    dataType = c['train']['dataUpdateType'] #See this in the paper
    eb = c['train']['errorBound']

def prepareNextIter(labelA,labelC,src,tgt):
    """ labelA labelC are np arraies, src tgt are tensors """
    nextSrc = [[] for i in range(numA)]
    nextTgt = [[] for i in range(numA)]
    for i in range(len(src)):
        nextSrc[labelC[i]].append(src[i])
        nextTgt[labelC[i]].append(tgt[i])
    nextSrc = [tensorData(nextSrc[i]) for i in range(numA)]
    nextTgt = [tensorData(nextTgt[i]) for i in range(numA)]
    return nextSrc,nextTgt

def evaluate(A,C,testSrc,testTgt):
    """ predictC is the prediction, whereas predictA is the groundtruth, undersand this """
    N = len(testSrc)*1.0
    predictC = softmax2num(modelDeduct(C,testSrc))
    predictAs = predByAs(A,testSrc)
    predictA = getLabel(predictAs,testTgt,eb)
    accuracy = sum([(predictC[i]>0 and errMeasure(predictAs[predictC[i]],testTgt[i])<eb) for i in range(N)])/double(N)
    

def train():
    print ("training begin")
    # Parameters
    loadTrainParas()
    srcEphem = [trainSrc.clone() for i in range(numA)]
    tgtEphem = trainTgt.clone()
    for iterN in range(iterNum):
        trainAs(A,srcEphem,tgtEphem)
        predictA = predByAs(A,trainSrc)
        labelA = getLabel(predictA,trainTgt,eb)
        optiModel(C,trainSrc,labelA)
        predictC = softmax2num(modelDeduct(C,trainSrc))
        srcEphem,tgtEphem = prepareNextIter(np.array(labelA),predictC,trainSrc,trainTgt)
        evaluate(A,C,testSrc,testTgt)
    
if (__name__=="__main__"):
    print ("Process begins")
    parserSetting()
    configSetting()
    logSetting()
    dataReading()
    modelLoading()
    train()
    
    
