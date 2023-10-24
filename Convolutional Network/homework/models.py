import torch
import torch.nn.functional as F

class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target):
        return F.cross_entropy(input, target)

class CNNClassifier(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, stride=1):
            super().__init__()
            self.network = torch.nn.Sequential(
                torch.nn.Conv2d(n_input, n_output, kernel_size=3, padding=1, stride=stride),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(),
                torch.nn.Conv2d(n_output, n_output, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU()
            )

        def forward(self, x):
            return self.network(x)

    def __init__(self, layers=[32, 64, 128], n_input_channels=3):
        super().__init__()
        L = [torch.nn.Conv2d(n_input_channels, 32, kernel_size=7, padding=3, stride=2),
             torch.nn.BatchNorm2d(32),
             torch.nn.ReLU(),
        ]
        c = 32
        for l in layers:
            L.append(self.Block(c, l, stride=2))
            c = l
        self.network = torch.nn.Sequential(*L)
        self.classifier = torch.nn.Linear(c, 6)

    def forward(self, x):
        average_pool = self.network(x)
        average_pool = average_pool.mean(dim=[2, 3])
        return self.classifier(average_pool)

    # class Block(torch.nn.Module):
    #         def __init__(self,n_input,n_output,stride=1):
    #             super().__init__()
    #             self.net = torch.nn.Sequential(
    #                 torch.nn.Conv2d(n_input, n_output, kernel_size = 3, padding = 1, stride=stride),
    #                 torch.nn.ReLU(),
    #                 torch.nn.Conv2d(n_input, n_output, kernel_size = 3, padding = 1, stride=1),
    #                 torch.nn.ReLU()
    #             )
    #         def forward(self,x):
    #             return self.net(x)
        
    #     # Start of ConvNet class definitions

    # def __init__(self, layers=[32,64,128], n_channels = 3):
    #     super().__init__()
    #     L = [
    #             torch.nn.Conv2d(n_channels, 32, kernel_size = 7, padding = 3, stride =2),
    #             torch.nn.ReLU(),
    #             torch.nn.MaxPool2d(kernel_size=3, stride = 2, padding =1)
    #         ]
    #     c = layers[0]
    #     for l in layers:
    #         L.append(self.Block(c,1,stride = 2))
    #         c = l
    #     self.network = torch.nn.Sequential(*L)
    #     self.classifier = torch.nn.Linear(c,1)

    # def forward(self, x):
    #     x = self.network(x)
    #     z = z.mean([2,3])
    #     return self.classifier(z)[:,0]



def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, CNNClassifier):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'cnn.th'))
    raise ValueError("model type '%s' not supported!"%str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = CNNClassifier()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'cnn.th'), map_location='cpu'))
    return r
