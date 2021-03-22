import torch
import torch.nn as nn
import torch.nn.functional as F

class GateNet(nn.Module):
    def __init__(self, model_nums, v_dim, sm=1):
        super(GateNet, self).__init__()
        self.v_dim = v_dim
        self.model_nums = model_nums
        #self.fc1 = nn.Linear(v_dim, v_dim)
        self.fc2 = nn.Linear(v_dim*model_nums, model_nums)
        self.softmax = sm

    def forward(self, x):
        x = x.view(-1, self.v_dim*self.model_nums) #[1137, 2048] 
        #x = self.fc1(x) #[1137, 1024]
        x = self.fc2(x)
        if self.softmax:
            return F.softmax(x, dim=1)
        return x


if __name__ == '__main__':
    net = GateNet()
    input = torch.autograd.Variable(torch.randn(784))
    y = net(input)
    print(net)
    print(y.size())