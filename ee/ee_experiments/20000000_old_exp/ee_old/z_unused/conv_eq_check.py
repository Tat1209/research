import torch
import torch.nn as nn
import utils

in_c = 3
out_c = 3

h = 5
g = 3

device = 'cuda'


for i in range(1):
    # x = torch.arange(1*in_c*h*h).float().to(device)
    # x = x.view([1, in_c, h, h]).to(device)

    x = torch.rand([1, in_c, h, h]).to(device)
    y = x.clone().detach().to(device)

    a = nn.Conv2d(in_c,out_c,(3,3),(2,2),(1,1),bias=False).to(device)
    b = nn.Conv2d(in_c*g,out_c,(3,3),(2,2),(1,1),bias=False, groups=g).to(device)
    
    norm = nn.BatchNorm2d(num_features=out_c, affine=True).to(device)
    




    val = 0.1
    for p in a.parameters(): p.data.fill_(val)
    for p in b.parameters(): p.data.fill_(val)


    y = torch.cat([y for i in range(g)], dim=1)


    utils.torch_fix_seed()
    x = a(x)
    y = b(y)
    
    x = norm(x)
    y = norm(y)
    
    print(x)
    print(y)

    print(torch.sum(torch.abs(y-x)) )
    print(torch.sum(torch.abs(y-x)) / torch.sum((torch.abs(x) + torch.abs(y) / 2)))


