import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

def gen_alexnet(num_classes):
    
    alexnet_model = models.alexnet(pretrained=True)

    conv_layers = alexnet_model.features

    # enable further training on pre-tranied conv weights
    for param in conv_layers.parameters():
        param.requires_grad = True

    classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(256 * 3 * 3, 2048), 
        nn.LeakyReLU(),
        nn.Dropout(0.5),
        nn.Linear(2048, 2048),
        nn.LeakyReLU(),
        nn.Dropout(0.5),
        nn.Linear(2048, num_classes), 
        nn.LogSoftmax(dim=1)
    )

    alexnet = nn.Sequential(
        conv_layers,
        classifier
    )

    return(alexnet)

class simple_alexnet(nn.Module):
    def __init__(self, num_classes=2):
        super(simple_alexnet, self).__init__()
        self.alexnet = models.alexnet(pretrained=True)
        self.alexnet.classifier[6] = nn.Linear(self.alexnet.classifier[6].in_features, num_classes)
        
    def forward(self, x):
        return self.alexnet(x)
    
