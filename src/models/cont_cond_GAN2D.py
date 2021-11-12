import torch
import torch.nn as nn
import numpy as np


#########################################################
# generator
bias_g = False
class cont_cond_generator(nn.Module):
    def __init__(self, ngpu=1, nz=2, nlabels=1, out_dim=2):
        super(cont_cond_generator, self).__init__()
        self.nz = nz
        self.nlabels = nlabels
        self.ngpu = ngpu
        self.out_dim = out_dim

        self.linear = nn.Sequential(
                nn.Linear(nz+nlabels, 128, bias=bias_g),
                nn.BatchNorm1d(128),
                nn.ReLU(True),

                nn.Linear(128, 256, bias=bias_g),
                nn.BatchNorm1d(256),
                nn.ReLU(True),

                nn.Linear(256, 512, bias=bias_g),
                nn.BatchNorm1d(512),
                nn.ReLU(True),

                nn.Linear(512, 1024, bias=bias_g),
                nn.BatchNorm1d(1024),
                nn.ReLU(True),

                nn.Linear(1024, 2048, bias=bias_g),
                nn.BatchNorm1d(2048),
                nn.ReLU(True),

                nn.Linear(2048, 4096, bias=bias_g),
                nn.BatchNorm1d(4096),
                nn.ReLU(True),

                nn.Linear(4096, self.out_dim, bias=bias_g),
            )

    def forward(self, input, labels):
        input = input.view(-1, self.nz)

        input = torch.cat((input, labels), 1)

        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.linear, input, range(self.ngpu))
        else:
            output = self.linear(input)
        return output

#########################################################
# discriminator
bias_d=False
class cont_cond_discriminator(nn.Module):
    def __init__(self, ngpu=1, nlabels=1, input_dim=2):
        super(cont_cond_discriminator, self).__init__()
        self.ngpu = ngpu
        self.nlabels = nlabels
        self.input_dim = input_dim
        
        self.main = nn.Sequential(
            nn.Linear(self.input_dim+self.nlabels, 4096, bias=bias_d),
            nn.ReLU(True),

            nn.Linear(4096, 2048, bias=bias_d),
            nn.ReLU(True),

            nn.Linear(2048, 1024, bias=bias_d),
            nn.ReLU(True),

            nn.Linear(1024, 512, bias=bias_d),
            nn.ReLU(True),

            nn.Linear(512, 256, bias=bias_d),
            nn.ReLU(True),

            nn.Linear(256, 1, bias=bias_d),
            nn.Sigmoid()
        )


    def forward(self, input, labels):
        input = input.view(-1, self.input_dim)

        input = torch.cat((input, labels), 1)

        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(-1, 1)

if __name__=="__main__":
    import numpy as np
    #test
    ngpu=1

    netG = cont_cond_generator(ngpu=ngpu, nz=2, out_dim=2)
    netD = cont_cond_discriminator(ngpu=ngpu, input_dim = 2)

    z = torch.randn(32, 2)
    y = np.random.randint(100, 300, 32)
    y = torch.from_numpy(y).type(torch.float).view(-1,1)
    x = netG(z,y)
    o = netD(x,y)
    print(y.size())
    print(x.size())
    print(o.size())
