# imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# CNN
class CNN(nn.Module):
    def __init__(self, in_channels=1,num_classes=10, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1=nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool=nn.MaxPool2d(kernel_size=(2,2),stride=2)

        self.conv2=nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.fc1=nn.Linear(16*7*7, num_classes)

    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=self.pool(x)
        x=F.relu(self.conv2(x))
        x=self.pool(x)
        x=x.reshape(x.shape[0], -1)
        x=self.fc1(x)

        return x
    
def check_accuracy(loader,model):
    if loader.dataset.train:
        print("Train data acc")
    else:
        print("test data acc")

    num_correct=0
    num_samples=0
    model.eval()

    with torch.no_grad():
        for x,y in loader:
            x=x.to(device)
            y=y.to(device)
            scores=model(x)

            _,pred=scores.max(1)
            num_correct+=(pred==y).sum()
            num_samples+=pred.size(0)

        print(f" Got{num_correct}/{num_samples} with an acc of {float(num_correct/num_samples)*100:.2f}")
    model.train()


device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# hyperparam
in_channels=1
num_classes=10
learning_rate=0.001
batch_size=64
num_epochs=1
model = CNN().to(device=device)
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=learning_rate)



train_dataset=datasets.MNIST(root="dataset/",train=True,transform=transforms.ToTensor(),download=True)
train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)


test_dataset=datasets.MNIST(root="dataset/",train=False,transform=transforms.ToTensor(),download=True)
test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)

for epoch in range (num_epochs):
    for batch_idx,(data,targets) in enumerate(train_loader):

        data=data.to(device)
        targets=targets.to(device)
        #forward
        scores=model(data)
        #loss & optim
        loss=criterion(scores,targets)
        optimizer.zero_grad()
        #backward
        loss.backward()
        optimizer.step()


check_accuracy(train_loader, model)
check_accuracy(test_loader,model)