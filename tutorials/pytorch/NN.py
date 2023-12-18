# imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Create fully connected network
class NN(nn.Module):
    def __init__(self, input_size,num_classes, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fc1=nn.Linear(input_size,50)
        self.fc2=nn.Linear(50,num_classes)
    
    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=self.fc2(x)
        return x
    
#model=NN(784,10)
#x=torch.randn(64,784)
#print(model(x).shape)
# Set device
device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparam
input_size=784
num_classes=10
learning_rate=0.001
batch_size=64 
num_epochs=1
# load data
train_dataset=datasets.MNIST(root="dataset/",train=True,transform=transforms.ToTensor(),download=True)
train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)


test_dataset=datasets.MNIST(root="dataset/",train=False,transform=transforms.ToTensor(),download=True)
test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)
# init network
model = NN(input_size=input_size,num_classes=num_classes).to(device)

#loss and optim
criterion=nn.CrossEntropyLoss()
optimzier=optim.Adam(model.parameters(),lr=learning_rate)
# train network
for epoch in range(num_epochs):
    for batch_idx,( data, targets) in enumerate (train_loader):
        
        data=data.to(device=device)
        targets=targets.to(device=device)
        
        # flatten
        data=data.reshape(data.shape[0],-1)
        #forward
        scores=model(data)
        # loss and optim
        loss=criterion(scores,targets)
        optimzier.zero_grad()
        #backward
        loss.backward()
        optimzier.step()

def check_accuracy(loader,model):
    if loader.dataset.train:
        print("train data acc")
    else:
        print("test data acc")
    num_correct=0
    num_samples=0
    model.eval()

    with torch.no_grad():
        for x,y in loader:
            x=x.to(device=device)
            y=y.to(device=device)
            x=x.reshape(x.shape[0],-1)

            scores=model(x)
            #scores 64x10, here as if I want to make soft max, so that I would take the correct label
            _, pred=scores.max(1)
            num_correct+=(pred==y).sum()
            num_samples+=pred.size(0)

        print (f"Got {num_correct}/{num_samples} with acc {float(num_correct)/float(num_samples)*100:.2f}")

    model.train()

# check acc
check_accuracy(train_loader, model)
check_accuracy(test_loader,model)




