import argparse
import torch
import copy
import numpy as np
import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import StandardScaler
np.random.seed(2)
torch.manual_seed(2)
class send_para(object):
    def __init__(self,data_dir='./data/sensor/',ld=16,lr=0.01,momentum=0.5,batch_size=16,epochs=100,nh=100,missingrate=0.1,test_seed=0,get_train=True):
        self.ld = ld
        self.lr = lr
        self.momentum = momentum
        self.batch_size = batch_size
        self.epochs = epochs
        self.nh = nh
        
        if get_train:
            
            self.trset = np.load(data_dir + 'trset.npy')
            np.random.seed(6)
            self.trset[np.random.choice([False,True],size=self.trset.shape,p=[1-missingrate,missingrate])]=np.nan
            
            self.vaset = np.load(data_dir + 'vaset.npy')
            np.random.seed(7)
            self.vaset[np.random.choice([False,True],size=self.vaset.shape,p=[1-missingrate,missingrate])]=np.nan
            
            self.trlabels = np.load(data_dir + 'trlabels.npy')
            self.valabels = np.load(data_dir + 'valabels.npy')
            
        else:
            
            self.teset = np.load(data_dir + 'teset.npy')
            np.random.seed(test_seed)
            self.teset[np.random.choice([False,True],size=self.teset.shape,p=[1-missingrate,missingrate])]=np.nan
            self.telabels = np.load(data_dir + 'telabels.npy')
        
        
        
        
class LoadData(Dataset):
    def __init__(self, data, labels):
        self.data = np.copy(data)
        self.labels = np.copy(labels)
        self.ms_pt = np.isnan(self.data)
        self.data[self.ms_pt] = 0
        self.ms_pt = np.concatenate((~self.ms_pt, self.ms_pt), axis=1).astype(np.float32)
    def __getitem__(self, index):
        instance = self.data[index]
        ms_pt = torch.from_numpy(self.ms_pt[index])
        y = torch.tensor(self.labels[index].astype(int))
        return ms_pt, instance, y
    def __len__(self):
        return self.data.shape[0]
        

def train(args, model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    for batch_idx, (ms_pt, data, target) in enumerate(train_loader):
        ms_pt, data, target = ms_pt.to(device), data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(ms_pt,data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def test(args, model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for ms_pt, data, target in test_loader:
            ms_pt, data, target = ms_pt.to(device), data.to(device), target.to(device)
            output = model(ms_pt, data)
            test_loss += criterion(output, target).item()*len(data) # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    return(test_accuracy)


class ScaleNet(nn.Module):
    def __init__(self, fd, n_class, ld, num_hidden):
        super().__init__()
        self.weight1 = nn.Parameter(torch.empty(fd*2,ld))  # define the trainable parameter fd:feature dimension
        self.weight1.data.uniform_(-0.1, 0.1)              # initialise
        self.weight2 = nn.Parameter(torch.empty(ld,fd))    # define the trainable parameter
        self.weight2.data.uniform_(-0.1, 0.1)              # initialise
        self.fc1 = nn.Linear(fd, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.fc3 = nn.Linear(num_hidden, n_class)
    def forward(self, mp, x):
        mp = torch.mm(mp,self.weight1)
        mp = torch.mm(mp,self.weight2)
        x =  torch.mul(x, mp)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':   
    data_dir = './nn_data/avila/'
    p = np.load("./nn_data/avila/para.npy")
    results_mean = []
    results_std = []
    for i,missing_rate in enumerate([0.1,0.3,0.5,0.7,0.9]):
                    
        args = send_para(data_dir=data_dir,ld=p[i],missingrate=missing_rate,test_seed=i)
        trset = args.trset
        trlabels = args.trlabels
        vaset = args.vaset   
        valabels = args.valabels
        
        scaler = StandardScaler()
        trset = scaler.fit_transform(trset)
        vaset = scaler.transform(vaset)
        
        device = torch.device("cuda")
        
        train_loader = torch.utils.data.DataLoader(
                       dataset=LoadData(trset,trlabels),
                       batch_size=args.batch_size, 
                       shuffle=True)
        va_loader = torch.utils.data.DataLoader(
                      dataset=LoadData(vaset,valabels),
                      batch_size=1024, 
                      shuffle=False)
        
        
        best_acc_ld = 0
            
        model = ScaleNet(fd=trset.shape[1], n_class=int(trlabels.max()+1), ld=args.ld, num_hidden=args.nh).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    

        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, criterion, optimizer, epoch)
            va_accuracy = test(args, model, device, va_loader, criterion)
            if va_accuracy >= best_acc_ld:
                best_acc_ld = va_accuracy
                best_model = copy.deepcopy(model)

        test_acc = []
        for rep in range(5):
            args = send_para(data_dir=data_dir,ld=p[i],missingrate=missing_rate,test_seed=rep,get_train=False)
            teset = args.teset   
            telabels = args.telabels
            teset = scaler.transform(teset)
            te_loader = torch.utils.data.DataLoader(
                        dataset=LoadData(teset,telabels),
                        batch_size=1024, 
                        shuffle=False)
            test_acc.append(test(args, best_model, device, te_loader, criterion))
        print("Missing Rate: {0:1.1f}   Mean Accuracy: {1:1.3f}, Standard Deviation: {2:1.3f}".format(missing_rate,np.mean(test_acc),np.std(test_acc)))
                     

