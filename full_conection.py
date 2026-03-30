import torch
import torchvision
from torch import nn
from torch.utils import data
from torchvision import transforms
def get_numworkers():
   return 0

def check_env():
   device=torch.device("cuda"if torch.cuda.is_available() else "cpu")
   return device

def load_data_fashion_mnist(batch_size,resize=None):
    trans=[transforms.ToTensor()]
    if resize:
      trans.insert(0,transforms.Resize(resize))
    trans=transforms.Compose(trans)
    mnist_train=torchvision.datasets.FashionMNIST(root="../data",train=True,transform=trans,download=True)
    mnist_test=torchvision.datasets.FashionMNIST(root="../data",train=False,transform=trans,download=True)
    return (data.DataLoader(mnist_train,batch_size,shuffle=True,num_workers=get_numworkers()),
    data.DataLoader(mnist_test,batch_size,shuffle=True,num_workers=get_numworkers()))

batch_size=256
train_iter,test_iter=load_data_fashion_mnist(batch_size,64)


net=nn.Sequential(
        nn.Flatten(),
        nn.LazyLinear(10)
    ).to(check_env())
loss=nn.CrossEntropyLoss()
trainer=torch.optim.SGD(net.parameters(),lr=0.01)


correct,total=0,0
num_epoch=10
print(check_env())
for epoch in range(num_epoch):
   for X,y in train_iter:
      X,y=X.to(check_env()),y.to(check_env())
      y_hat=net(X)
      l=loss(y_hat,y)
      trainer.zero_grad()
      l.backward()
      trainer.step()
      y2=y_hat.argmax(dim=1)
      correct+=(y2==y).sum().item()
      total+=y.numel()
   accuracy=correct/total
   print(f'epoch {epoch+1}, accuracy {accuracy:.3f}')
      