from globalSetting import *
def configSetting():
    global c
    print("config loadding")
    c = json.load(open(configPath,'r'))
    configSavedPath = os.path.join(workDir, 'config.json')
    configSavedFile = open(configSavedPath, 'w')
    json.dump(c, configSavedFile, sort_keys=False, indent=4)

def logSetting():
    logging.basicConfig(
        filename=logPath,
        filemode="a+",
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%y/%m/%d_%H:%M:%S",
        level=logging.DEBUG
    )
    logging.debug("test log function")

def dataReading():
    global trainSrc,trainTgt,testSrc,testTgt,benchName
    print ("data reading")
    trainSrc,trainTgt,testSrc,testTgt = loadData(benchName)
    print ("Src, Tgt shapes{} {}".format(trainSrc.size,trainTgt.size))

def modelLoading():
    print ("model loading")
    global A,C,netA,netC,numA
    numA = c['model']['numA']
    netA,netC = getNetStructure(benchName,numA)
    A = [ANet(netA) for i in range(numA)]
    C = CNet(netC)
