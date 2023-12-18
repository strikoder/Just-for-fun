import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyperparam
#N,1,28,28: 28 sequences each one has 28 features
input_size=28
sequence_length=28
num_layers=2
hidden_size=256
num_classes=10
learning_rate=0.001
batch_size=64
num_epochs=1

class RNN(nn.Module):
    def __init__(self,input_size,hideen_size,num_layers,num_classes, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hidden_size=hideen_size
        self.num_layers=num_layers
        self.rnn=nn.RNN(input_size, hideen_size, num_layers, batch_first=True)
        #self.lstm=nn.LSTM(etc)
        #self.gru=nn.GRU(etc)
        self.fc=nn.Linear(hidden_size*sequence_length,num_classes)
        #self.fc=nn.Linear(hidden_size,num_classes) taking only the last hidden state
        # do not forget to remove the reshape if you did that and the last out would be out=fc(out[:,-1,:])

    def forward(self,x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        #(for lstm) c0= torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        #out,_=self.lstm(x,(h0,c0))
        out,_=self.rnn(x,h0)
        #28*256
        out=out.reshape(out.shape[0],-1) 
        out=self.fc(out)

        return out


def check_acc(loader,model):
    if loader.dataset.train:
        print("checking acc on training data")
    else:
        print("chekcing acc on testing data")

    num_correct=0
    num_samples=0
    model.eval()

    with torch.no_grad():
        for x,y in loader:
            x=x.to(device).squeeze(1)
            y=y.to(device)
            
            scores=model(x)
            _,pred=scores.max(1)
            num_correct+=(pred==y).sum()
            num_samples+=pred.size(0)
            print(f'acc {float((num_correct)/(num_samples))*100:.2f}')
            
    
    model.train()



train_dataset = datasets.MNIST(root='dataset/',train=True,transform=transforms.ToTensor(),download=True)
train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

test_dataset= datasets.MNIST(root='dataset/',train=False,transform=transforms.ToTensor(),download=True)
test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)

model= RNN(input_size,hidden_size, num_layers,num_classes).to(device)


criterion = nn.CrossEntropyLoss()
optimzer=optim.Adam(model.parameters(),lr=learning_rate)

for epoch in range(num_epochs):
    for batch_idx,(data,targets) in enumerate(train_loader):
        data=data.to(device).squeeze(1)
        targets=targets.to(device)

        #forward
        scores=model(data)

        #loss & optim
        loss=criterion(scores,targets)
        optimzer.zero_grad()

        #backward
        loss.backward()
        optimzer.step()

check_acc(train_loader,model)
check_acc(train_loader,model)