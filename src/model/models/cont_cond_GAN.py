import torch
import torch.nn as nn
import numpy as np


#########################################################
# genearator
bias_g = False
class cont_cond_generator(nn.Module):
    def __init__(self, ngpu=1, nz=2, out_dim=2, label_min = 0., label_max = 1., const_mass = 500., axis = 'phi'):
        super(cont_cond_generator, self).__init__()
        self.nz = nz
        self.ngpu = ngpu
        self.out_dim = out_dim

        self.label_min = label_min
        self.label_max = label_max
        self.const_mass = const_mass
        self.axis = axis

        self.inner_dim = 100

        self.linear = nn.Sequential(
                nn.Linear(nz+2, self.inner_dim, bias=bias_g),
                nn.BatchNorm1d(self.inner_dim),
                nn.ReLU(True),

                nn.Linear(self.inner_dim, self.inner_dim, bias=bias_g),
                nn.BatchNorm1d(self.inner_dim),
                nn.ReLU(True),

                nn.Linear(self.inner_dim, self.inner_dim, bias=bias_g),
                nn.BatchNorm1d(self.inner_dim),
                nn.ReLU(True),

                nn.Linear(self.inner_dim, self.inner_dim, bias=bias_g),
                nn.BatchNorm1d(self.inner_dim),
                nn.ReLU(True),

                nn.Linear(self.inner_dim, self.inner_dim, bias=bias_g),
                nn.BatchNorm1d(self.inner_dim),
                nn.ReLU(True),

                nn.Linear(self.inner_dim, self.inner_dim, bias=bias_g),
                nn.BatchNorm1d(self.inner_dim),
                nn.ReLU(True),

                nn.Linear(self.inner_dim, self.out_dim, bias=bias_g),
            )

    def forward(self, input, labels):
        input = input.view(-1, self.nz)
        labels = labels.view(-1,1) * (self.label_max - self.label_min) + self.label_min

        labels = labels.view(-1, 1) * (self.label_max - self.label_min) + self.label_min
        if self.axis == 'phi':
            input = torch.cat((input, labels, torch.ones((len(labels),1)) * self.const_mass), 1)
        else:
            input = torch.cat((input, torch.ones((len(labels),1)) * self.const_mass, labels), 1)

        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.linear, input, range(self.ngpu))
        else:
            output = self.linear(input)
        return output

#########################################################
# discriminator
bias_d=False
class cont_cond_discriminator(nn.Module):
    def __init__(self, ngpu=1, input_dim = 2, label_min = 0., label_max = 1., const_mass = 500., axis = 'phi'):
        super(cont_cond_discriminator, self).__init__()
        self.ngpu = ngpu
        self.input_dim = input_dim

        self.label_min = label_min
        self.label_max = label_max
        self.const_mass = const_mass
        self.axis = axis

        self.inner_dim = 100
        self.main = nn.Sequential(
            nn.Linear(input_dim+2, self.inner_dim, bias=bias_d),
            nn.ReLU(True),

            nn.Linear(self.inner_dim, self.inner_dim, bias=bias_d),
            nn.ReLU(True),

            nn.Linear(self.inner_dim, self.inner_dim, bias=bias_d),
            nn.ReLU(True),

            nn.Linear(self.inner_dim, self.inner_dim, bias=bias_d),
            nn.ReLU(True),

            nn.Linear(self.inner_dim, self.inner_dim, bias=bias_d),
            nn.ReLU(True),

            nn.Linear(self.inner_dim, 1, bias=bias_d),
            nn.Sigmoid()
        )


    def forward(self, input, labels):
        input = input.view(-1, self.input_dim)
        labels = labels.view(-1, 1) * (self.label_max - self.label_min) + self.label_min

        if self.axis == 'phi':
            input = torch.cat((input, labels, torch.ones((len(labels),1)) * self.const_mass), 1)
        else:
            input = torch.cat((input, torch.ones((len(labels),1)) * self.const_mass, labels), 1)

        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(-1, 1)





if __name__=="__main__":
    import numpy as np
    #test
    ngpu=1

    netG = cont_cond_generator(ngpu=ngpu, nz=2, out_dim=2).cuda()
    netD = cont_cond_discriminator(ngpu=ngpu, input_dim = 2).cuda()

    z = torch.randn(32, 2).cuda()
    y = np.random.randint(100, 300, 32)
    y = torch.from_numpy(y).type(torch.float).view(-1,1).cuda()
    x = netG(z,y)
    o = netD(x,y)
    print(y.size())
    print(x.size())
    print(o.size())
