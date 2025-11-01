from torch import nn
import torch

class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def __repr__(self):
        return f'View{self.shape}'

    def forward(self, input):
        '''
        Reshapes the input according to the shape saved in the view data structure.
        '''
        batch_size = input.size(0)
        shape = (batch_size, *self.shape)
        out = input.view(shape)
        return out
    
    
x = torch.tensor(torch.rand(500,2048,1,1))
# Viewクラスのインスタンスを作成
view = View((-1,1,1))

# テンソルを変換
y = view(x)
print(y.shape) # torch.Size([4, 6])
