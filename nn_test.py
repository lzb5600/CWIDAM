import argparse
import torch
import copy
import numpy as np
import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import StandardScaler

       
        
        
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
    def __init__(self, fd, n_class, ld, num_hidden=100):
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
    
    data_dir = './data/sensor/'
    trset = np.load(data_dir + 'trset.npy')
    teset = np.load(data_dir + 'teset.npy')
    vaset = np.load(data_dir + 'vaset.npy')
    trlabels = np.load(data_dir + 'trlabels.npy')
    telabels = np.load(data_dir + 'telabels.npy')
    valabels = np.load(data_dir + 'telabels.npy')
 
    k = 10 # rank(H)<=k
    scaler = StandardScaler()
    trset = scaler.fit_transform(trset)
    teset = scaler.transform(teset)
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
    te_loader = torch.utils.data.DataLoader(
                   dataset=LoadData(teset,telabels),
                   batch_size=1024, 
                   shuffle=False)


    model = ScaleNet(fd=trset.shape[1], n_class=int(trlabels.max()+1), ld=k, num_hidden=100).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    best_acc = 0
    for epoch in range(1, 200):
        train(args, model, device, train_loader, criterion, optimizer, epoch)
        va_accuracy = test(args, model, device, va_loader, criterion)
        if va_accuracy >= best_acc:
            best_acc = va_accuracy
            best_model = copy.deepcopy(model)
    
    test_acc=test(args, best_model, device, te_loader, criterion)
    print(test_acc)


