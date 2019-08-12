import torch
import torch.nn as nn
import torchvision.models as models
#from collections import namedtuple
#import torch.nn.functional as F
from functools import reduce
#from .utils import load_state_dict_from_url


class Baseline(nn.Module):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=hidden_size, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(hidden_size),
            nn.Conv2d(hidden_size, hidden_size, 3, 2, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(hidden_size),
            nn.Conv2d(hidden_size, hidden_size, 3, 2, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(hidden_size),
            nn.Conv2d(hidden_size, hidden_size, 3, 2, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(hidden_size),
            nn.Conv2d(hidden_size, hidden_size, 3, 2, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(hidden_size),
            nn.Conv2d(hidden_size, out_size, 4, 1),
        )

    def forward(self, image):
        return self.net(image).squeeze(-1).squeeze(-1)


class Resnet(nn.Module):
    def __init__(self, out_size):
        super().__init__()
        model = models.resnet18(pretrained=True)
        model = list(model.children())[:-1]
        model.append(nn.Conv2d(512, out_size, 1))
        self.net = nn.Sequential(*model)

    def forward(self, image):
        return self.net(image).squeeze(-1).squeeze(-1)

class Inception(nn.Module):
    def __init__(self, out_size):
        super().__init__()
        model = models.inception_v3(pretrained=True)
#        model = list(model.children())[:-1]
#        model.append(nn.Conv2d(512, out_size, 1))
        self.net = nn.Sequential(*model)
    def forward(self, image):
        return self.net(image).squeeze(-1).squeeze(-1)

class densenet(nn.Module):
    def __init__(self, out_size):
        super().__init__()
        model = models.densnet121(pretrained=True)
#        model = list(model.children())[:-1]
#        model.append(nn.Conv2d(512, out_size, 1))
        self.net = nn.Sequential(*model)
    
    def forward(self, image):
        return self.net(image).squeeze(-1).squeeze(-1)

class SKConv(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,M=2,r=16,L=32):
        
        super(SKConv,self).__init__()
        d=max(in_channels//r,L)
        self.M=M
        self.out_channels=out_channels
        self.conv=nn.ModuleList()
        for i in range(M):
            self.conv.append(nn.Sequential(nn.Conv2d(in_channels,out_channels,3,stride,padding=1+i,dilation=1+i,groups=32,bias=False),
                                           nn.BatchNorm2d(out_channels),
                                           nn.ReLU(inplace=True)))
        self.global_pool=nn.AdaptiveAvgPool2d(1)
        self.fc1=nn.Sequential(nn.Conv2d(out_channels,d,1,bias=False),
                               nn.BatchNorm2d(d),
                               nn.ReLU(inplace=True))
        self.fc2=nn.Conv2d(d,out_channels*M,1,1,bias=False)
        self.softmax=nn.Softmax(dim=1)
        def forward(self, input):
            batch_size=input.size(0)
            output=[]
            #the part of split
            for i,conv in enumerate(self.conv):
                #print(i,conv(input).size())
                output.append(conv(input))
            #the part of fusion
            U=reduce(lambda x,y:x+y,output)
            s=self.global_pool(U)
            z=self.fc1(s)
            a_b=self.fc2(z)
            a_b=a_b.reshape(batch_size,self.M,self.out_channels,-1)
            a_b=self.softmax(a_b)
            #the part of selection
            a_b=list(a_b.chunk(self.M,dim=1))#split to a and b
            a_b=list(map(lambda x:x.reshape(batch_size,self.out_channels,1,1),a_b))
            V=list(map(lambda x,y:x*y,output,a_b))
            V=reduce(lambda x,y:x+y,V)
            return V
class SKBlock(nn.Module):
    expansion=2
    def __init__(self,inplanes,planes,stride=1,downsample=None):
        super(SKBlock,self).__init__()
        self.conv1=nn.Sequential(nn.Conv2d(inplanes,planes,1,1,0,bias=False),
                                 nn.BatchNorm2d(planes),
                                 nn.ReLU(inplace=True))
        self.conv2=SKConv(planes,planes,stride)
        self.conv3=nn.Sequential(nn.Conv2d(planes,planes*self.expansion,1,1,0,bias=False),
                                 nn.BatchNorm2d(planes*self.expansion))
        self.relu=nn.ReLU(inplace=True)
        self.downsample=downsample
    def forward(self, input):
        shortcut=input
        output=self.conv1(input)
        output=self.conv2(output)
        output=self.conv3(output)
        if self.downsample is not None:
            shortcut=self.downsample(input)
        output+=shortcut
        return self.relu(output)
class SKNet(nn.Module):
    def __init__(self,nums_class=1000,block=SKBlock,nums_block_list=[3, 4, 6, 3]):
        super(SKNet,self).__init__()
        self.inplanes=64
        self.conv=nn.Sequential(nn.Conv2d(3,64,7,2,3,bias=False),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True))
        self.maxpool=nn.MaxPool2d(3,2,1)
        self.layer1=self._make_layer(block,128,nums_block_list[0],stride=1)
        self.layer2=self._make_layer(block,256,nums_block_list[1],stride=2)
        self.layer3=self._make_layer(block,512,nums_block_list[2],stride=2)
        self.layer4=self._make_layer(block,1024,nums_block_list[3],stride=2)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Linear(1024*block.expansion,nums_class)
        self.softmax=nn.Softmax(-1)
    def forward(self, input):
        output=self.conv(input)
        output=self.maxpool(output)
        output=self.layer1(output)
        output=self.layer2(output)
        output=self.layer3(output)
        output=self.layer4(output)
        output=self.avgpool(output)
        output=output.squeeze(-1).squeeze(-1)
        output=self.fc(output)
        output=self.softmax(output)
        return output
    def _make_layer(self,block,planes,nums_block,stride=1):
        downsample=None
        if stride!=1 or self.inplanes!=planes*block.expansion:
            downsample=nn.Sequential(nn.Conv2d(self.inplanes,planes*block.expansion,1,stride,bias=False),
                                     nn.BatchNorm2d(planes*block.expansion))
        layers=[]
        layers.append(block(self.inplanes,planes,stride,downsample))
        self.inplanes=planes*block.expansion
        for _ in range(1,nums_block):
            layers.append(block(self.inplanes,planes))
        return nn.Sequential(*layers)


#class SKConv(nn.Module):
#    def __init__(self, features, WH, M, G, r, stride=1 ,L=32):
#        """ Constructor
#            Args:
#            features: input channel dimensionality.
#            WH: input spatial dimensionality, used for GAP kernel size.
#            M: the number of branchs.
#            G: num of convolution groups.
#            r: the radio for compute d, the length of z.
#            stride: stride, default 1.
#            L: the minimum dim of the vector z in paper, default 32.
#            """
#        super(SKConv, self).__init__()
#        d = max(int(features/r), L)
#        self.M = M
#        self.features = features
#        self.convs = nn.ModuleList([])
#        for i in range(M):
#            self.convs.append(nn.Sequential(
#                                            nn.Conv2d(features, features, kernel_size=3+i*2, stride=stride, padding=1+i, groups=G),
#                                            nn.BatchNorm2d(features),
#                                            nn.ReLU(inplace=False)
#                                            ))
#        # self.gap = nn.AvgPool2d(int(WH/stride))
#        self.fc = nn.Linear(features, d)
#        self.fcs = nn.ModuleList([])
#        for i in range(M):
#            self.fcs.append(
#                            nn.Linear(d, features)
#                            )
#        self.softmax = nn.Softmax(dim=1)
#
#    def forward(self, x):
#        for i, conv in enumerate(self.convs):
#            fea = conv(x).unsqueeze_(dim=1)
#            if i == 0:
#                feas = fea
#            else:
#                feas = torch.cat([feas, fea], dim=1)
#        fea_U = torch.sum(feas, dim=1)
#        # fea_s = self.gap(fea_U).squeeze_()
#        fea_s = fea_U.mean(-1).mean(-1)
#        fea_z = self.fc(fea_s)
#        for i, fc in enumerate(self.fcs):
#            vector = fc(fea_z).unsqueeze_(dim=1)
#            if i == 0:
#                attention_vectors = vector
#            else:
#                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
#        attention_vectors = self.softmax(attention_vectors)
#        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
#        fea_v = (feas * attention_vectors).sum(dim=1)
#        return fea_v
#
#
#class SKUnit(nn.Module):
#    def __init__(self, in_features, out_features, WH, M, G, r, mid_features=None, stride=1, L=32):
#        """ Constructor
#            Args:
#            in_features: input channel dimensionality.
#            out_features: output channel dimensionality.
#            WH: input spatial dimensionality, used for GAP kernel size.
#            M: the number of branchs.
#            G: num of convolution groups.
#            r: the radio for compute d, the length of z.
#            mid_features: the channle dim of the middle conv with stride not 1, default out_features/2.
#            stride: stride.
#            L: the minimum dim of the vector z in paper.
#            """
#        super(SKUnit, self).__init__()
#        if mid_features is None:
#            mid_features = int(out_features/2)
#        self.feas = nn.Sequential(
#                                  nn.Conv2d(in_features, mid_features, 1, stride=1),
#                                  nn.BatchNorm2d(mid_features),
#                                  SKConv(mid_features, WH, M, G, r, stride=stride, L=L),
#                                  nn.BatchNorm2d(mid_features),
#                                  nn.Conv2d(mid_features, out_features, 1, stride=1),
#                                  nn.BatchNorm2d(out_features)
#                                  )
#
#        if in_features == out_features:
#            # when dim not change, in could be added diectly to out
#            self.shortcut = nn.Sequential()
#        else: # when dim not change, in should also change dim to be added to out
#            self.shortcut = nn.Sequential(
#                                          nn.Conv2d(in_features, out_features, 1, stride=stride),
#                                          nn.BatchNorm2d(out_features)
#                                          )
#
#def forward(self, x):
#    fea = self.feas(x)
#    return fea + self.shortcut(x)
#
#
#class SKNet(nn.Module):
#    def __init__(self, class_num):
#        super(SKNet, self).__init__()
#        self.basic_conv = nn.Sequential(
#                                        nn.Conv2d(3, 224, 3, padding=1),
#                                        nn.BatchNorm2d(224)
#                                        ) # 32x32
#        self.stage_1 = nn.Sequential(
#                                     SKUnit(224, 256, 32, 2, 8, 2, stride=2),
#                                     nn.ReLU(),
#                                     SKUnit(256, 256, 32, 2, 8, 2),
#                                     nn.ReLU(),
#                                     SKUnit(256, 256, 32, 2, 8, 2),
#                                     nn.ReLU()
#                                     ) # 32x32
#        self.stage_2 = nn.Sequential(
#                                     SKUnit(256, 512, 32, 2, 8, 2, stride=2),
#                                     nn.ReLU(),
#                                     SKUnit(512, 512, 32, 2, 8, 2),
#                                     nn.ReLU(),
#                                     SKUnit(512, 512, 32, 2, 8, 2),
#                                     nn.ReLU()
#                                     ) # 16x16
#        self.stage_3 = nn.Sequential(
#                                     SKUnit(512, 1024, 32, 2, 8, 2, stride=2),
#                                     nn.ReLU(),
#                                     SKUnit(1024, 1024, 32, 2, 8, 2),
#                                     nn.ReLU(),
#                                     SKUnit(1024, 1024, 32, 2, 8, 2),
#                                     nn.ReLU()
#                                     ) # 8x8
#        self.pool = nn.AvgPool2d(8)
#        self.classifier = nn.Sequential(
#                                        nn.Linear(1024, class_num),
#                                        # nn.Softmax(dim=1)
#                                        )
#
#def forward(self, image):
#    print("make fea")
#    fea = self.basic_conv(image)
#    fea = self.stage_1(fea)
#    print("s1")
#    fea = self.stage_2(fea)
#    print("s2")
#    fea = self.stage_3(fea)
#    print("s3")
#    fea = self.pool(fea)
#    print("pool")
#    fea = fea.squeeze(-1).squeeze(-1)
##    print("no type")
#    #fea = self.classifier(fea)
#    #fea = self.net(fea).squeeze(-1).squeeze(-1)
#    return fea
