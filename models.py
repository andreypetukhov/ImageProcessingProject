import numpy as np
import torch
import torch.nn as nn






class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class Reshape(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), 192, 192)
    
    
# Encoder
class Encoder(nn.Module):
    
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=9, stride=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.1),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=9, stride=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.1),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=9, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.1),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=9, stride=1, padding=1),
                       
        )
        
    def forward(self, x):
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        return h
    
# Decoder
class Decoder(nn.Module):
    
    def __init__(self):
        super(Decoder, self).__init__()

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(2, 64, kernel_size=7, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.1),
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size= 11, stride=2, padding = 2, output_padding = 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.1),
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=9, stride=3, padding=2, output_padding= 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.1),
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=9, stride=3, padding = 2, output_padding= 1),
            Flatten()
            
        )
        
    def forward(self, x):
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        return torch.sigmoid(h)
    
# Discriminator
class Discriminator(nn.Module):
    
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.flatten = Flatten()
        self.layer1 =  nn.Sequential(
            nn.Linear(2, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.layer2 =  nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.layer3=  nn.Sequential(
            nn.Linear(1024, 1),
            )
        
    def forward(self, x):
        x = self.flatten(x)
        h = self.layer1(x)
        h = self.layer2(h)
        h = torch.sigmoid(self.layer3(h))
        return h
    

def gaussian_mixture(batchsize, ndim, num_labels):
    if ndim % 2 != 0:
        raise Exception("ndim must be a multiple of 2.")

    def sample(x, y, label, num_labels):
        shift = 1.4
        r = 2.0 * np.pi / float(num_labels) * float(label)
        new_x = x * np.cos(r) - y * np.sin(r)
        new_y = x * np.sin(r) + y * np.cos(r)
        new_x += shift * np.cos(r)
        new_y += shift * np.sin(r)
        return np.array([new_x, new_y]).reshape((2,))

    x_var = 0.5
    y_var = 0.05
    x = np.random.normal(0, x_var, (batchsize, ndim // 2))
    y = np.random.normal(0, y_var, (batchsize, ndim // 2))
    z = np.empty((batchsize, ndim), dtype=np.float32)
    for batch in range(batchsize):
        for zi in range(ndim // 2):
            z[batch, zi*2:zi*2+2] = sample(x[batch, zi], y[batch, zi], np.random.randint(0, num_labels), num_labels)
    return z

def swiss_roll(batchsize, ndim, num_labels):
    def sample(label, num_labels):
        uni = np.random.uniform(0.0, 1.0) / float(num_labels) + float(label) / float(num_labels)
        r = np.sqrt(uni) * 3.0
        rad = np.pi * 4.0 * np.sqrt(uni)
        x = r * np.cos(rad)
        y = r * np.sin(rad)
        return np.array([x, y]).reshape((2,))

    z = np.zeros((batchsize, ndim), dtype=np.float32)
    for batch in range(batchsize):
        for zi in range(ndim // 2):
            z[batch, zi*2:zi*2+2] = sample(np.random.randint(0, num_labels), num_labels)
    return z

def inv(t, n=1):
    if t > 1/2:
        return (2*t - 1)**(1/(2*n + 1))
    else:
        return -(1 - 2*t)**(1/(2*n + 1))
    
def linear(t):
    if t < 1/2:
        return -np.sqrt(1 - 2*t)
    else:
        return np.sqrt(2*t - 1)