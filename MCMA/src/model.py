from globalSetting import *
from utils import *
from data import *
class ANet(nn.Module):
    def __init__(self,netLst,activate=nn.Sigmoid()):
        super(ANet, self).__init__()
        self.activate = []
        # Use ModuleList to turn the normal List into an iteriable list for
        # the optimizer to iterate
        self.layer = nn.ModuleList()
        for i in range(len(netLst)-1):
            self.layer.append(nn.Linear(netLst[i],netLst[i+1]))
            self.activate.append(activate)
        self.optimizer = torch.optim.Adam(self.parameters())
        self.criterion = torch.nn.MSELoss(reduction = 'sum')

    # Input.size = [batchSize, layer[0].inputSize]
    def forward(self, dataflow):
        for i in range(len(self.layer)):
            dataflow = self.layer[i](dataflow)
            dataflow = self.activate[i](dataflow)
        return dataflow 

class CNet(nn.Module):
    def __init__(self,netLst, activate=nn.Sigmoid()):
        super(CNet, self).__init__()
        self.activate = [] 
        self.layer = nn.ModuleList() 
        for i in range(len(netLst)-1):
            self.layer.append(nn.Linear(netLst[i],netLst[i+1]))
            self.activate.append(activate if (i!=len(netLst)-2) else nn.LogSoftmax(dim=1))
        self.optimizer = torch.optim.Adam(self.parameters())
        self.criterion = torch.nn.NLLLoss()

    def forward(self, dataflow):
        for i in range(len(self.layer)):
            dataflow = self.layer[i](dataflow)
            dataflow = self.activate[i](dataflow)
        return dataflow

if (__name__=="__main__"):
    print ("Model test")
    benchName = 'bessel_Jnu'
    x,y,_,_ = loadData(benchName)
    #x = x[:100]
    #y = y[:100]
    minix = miniBatch(x,8,1)

    netA,netC = getNetStructure(benchName,3)
    A = ANet(netA)

    criterion = torch.nn.MSELoss(reduction = 'sum')
    for t in range(5000):
        A.optimizer.zero_grad()

        y_pred = A(x)
        loss = criterion(y_pred, y)
        if (t%100==0):
            print (t, loss.item())
        loss.backward()
        A.optimizer.step()
