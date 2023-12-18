import torch

device='cuda' if torch.cuda.is_available() else "cpu"

my_tensor = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.float32,device=device)

x=torch.zeros((3,3))
x=torch.eye(5,5)

# converting tensors to other types (int,float,double)
tensor=torch.arange(4)
tensor=tensor.bool() #converting to bool 
#.short()/.long()/.half()

# Math & comparsion
x=torch.tensor([[1,2,3],[1,2,3]]) #(2,3)
y=torch.tensor([[9,8,7],[1,2,3]])

# Addition and substraction
z1=x+y #or z=torch.add(x,y)

# ============================================================= #
#                               Tensor Math                 #
# ============================================================= #

# Matrix Multiplication (mm):
print(x.mm(y.T))
#Dot Product (dot):
#torch.dot(x,y) for 1D vectors
print(x@y.T)
#Element-wise Multiplication (*):
print(x*y)

#torch.bmm(tensor1,tensor2) ==> (batch,n,p)

# ============================================================= #
#                               Tensor Reshaping                 #
# ============================================================= #

x=torch.arange(9)

x_3x3=x.view(3,3) #for contiugous tensors
x_3x3=x.reshape(3,3)
y=x_3x3.t()
print(y.contiguous().view(9))
#.permute
#.unsqueez