import torch.nn as nn
from torchvision import models


class CNNNet(nn.Module):
    def __init__(self, model_name, code_length, pretrained=True):
        super(CNNNet, self).__init__()
        if model_name == "alexnet":
            original_model = models.alexnet(pretrained)
            self.features = original_model.features
            cl1 = nn.Linear(256 * 6 * 6, 4096)
            cl2 = nn.Linear(4096, 4096)
            self.fc_encode1 = nn.Linear(224, 21504)
            self.fc_encode2 = nn.Linear(21504, 224)
            if pretrained:
                cl1.weight = original_model.classifier[1].weight
                cl1.bias = original_model.classifier[1].bias
                cl2.weight = original_model.classifier[4].weight
                cl2.bias = original_model.classifier[4].bias

            self.classifier = nn.Sequential(
                nn.Dropout(),
                cl1,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                cl2,
                nn.ReLU(inplace=True),
                nn.Linear(4096, code_length),
                nn.Tanh()
            )
            self.model_name = 'alexnet'

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_encode1.weight)
        nn.init.xavier_uniform_(self.fc_encode2.weight)

    def forward(self, x):
        attn = self.fc_encode2(self.tanh(self.fc_encode1(x)))
        attn = self.softmax(attn)
        x2 = attn + x
        x3 = self.vgg19_bn.features(x2)
        x4 = x3.view(x.size(0), -1)     
        f = self.features(x4)
        if self.model_name == 'alexnet':
            f = f.view(f.size(0), 256 * 6 * 6)

        y = self.classifier(f)
        return y

class CNNExtractNet(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super(CNNExtractNet, self).__init__()
        if model_name == "alexnet":
            original_model = models.alexnet(pretrained)
            self.features = original_model.features
            cl1 = nn.Linear(256 * 6 * 6, 4096)
            cl2 = nn.Linear(4096, 4096)
            self.fc_encode1 = nn.Linear(224, 21504)
            self.fc_encode2 = nn.Linear(21504, 224)
            if pretrained:
                cl1.weight = original_model.classifier[1].weight
                cl1.bias = original_model.classifier[1].bias
                cl2.weight = original_model.classifier[4].weight
                cl2.bias = original_model.classifier[4].bias

            self.classifier = nn.Sequential(
                nn.Dropout(),
                cl1,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                cl2,
                nn.ReLU(inplace=True),
            )
            self.model_name = 'alexnet'

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_encode1.weight)
        nn.init.xavier_uniform_(self.fc_encode2.weight)


    def forward(self, x):
        attn = self.fc_encode2(self.tanh(self.fc_encode1(x)))
        attn = self.softmax(attn)
        x2 = attn + x
        x3 = self.vgg19_bn.features(x2)
        x4 = x3.view(x.size(0), -1)     
        f = self.features(x4)
        if self.model_name == 'alexnet':
            f = f.view(f.size(0), 256 * 6 * 6)

        y = self.classifier(f)
        return y
