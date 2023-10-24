import torch
import torch.nn.functional as F

# Loss Function
class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target, weight = None):
        return F.cross_entropy(input, target)

# Tuned CNN Model
class CNNClassifier(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, stride=1):
            super().__init__()
            self.network = torch.nn.Sequential(
                torch.nn.Conv2d(n_input, n_output, kernel_size=3, padding=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(),
                torch.nn.Conv2d(n_output, n_output, kernel_size=3, padding=1, bias=False),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU()
            )
            self.downsample = None 
            if n_input != n_output or stride != 1:
                self.downsample = torch.nn.Sequential(torch.nn.Conv2d(n_input, n_output, kernel_size = 1, stride = stride),
                                                      torch.nn.BatchNorm2d(n_output))

        def forward(self, x):
            identity = x
            if self.downsample is not None:
                identity = self.downsample(identity)
            return self.network(x) + identity

    def __init__(self, layers=[32, 64, 128], n_input_channels=3):
        super().__init__()
        L = [torch.nn.Conv2d(n_input_channels, 32, kernel_size=3, padding=3, stride=2),
             torch.nn.BatchNorm2d(32),
             torch.nn.ReLU(),
             torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ]
        c = 32
        for l in layers:
            L.append(self.Block(c, l, stride=2))
            c = l
        self.network = torch.nn.Sequential(*L)
        self.classifier = torch.nn.Linear(c, 6)
        torch.nn.init.zeros_(self.classifier.weight)

    def forward(self, x):
        # """
        # Your code here
        # @x: torch.Tensor((B,3,64,64))
        # @return: torch.Tensor((B,6))
        # Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        # """
        torch.nn.Dropout2d(p=0.5)
        average_pool = self.network(x)
        average_pool = average_pool.mean(dim=[2, 3])
        return self.classifier(average_pool)



##### Start of FCN Model Implementation ##### 



# Double Convolution Block
class DoubleConv(torch.nn.Module):
    def __init__(self, n_input, n_output, kernel_size = 3, stride = 1, padding = 1):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(n_input, n_output, kernel_size = kernel_size, stride = stride, padding = padding),
            torch.nn.BatchNorm2d(n_output),
            torch.nn.ReLU(),
            torch.nn.Conv2d(n_output, n_output, kernel_size = kernel_size, stride = stride, padding = padding),
            torch.nn.BatchNorm2d(n_output),
            torch.nn.ReLU()
        )
        

        
        self.downsample = torch.nn.Sequential(torch.nn.Conv2d(n_input, n_output, kernel_size = 1, stride = 2),
                                                   torch.nn.BatchNorm2d(n_output))

    def forward(self, x):
        # identity = x
        # if self.downsample is not None:
        #     identity = self.downsample(identity)
        # return self.network(x) + identity
        x = self.network(x)
        
        return x
    
# Up Convolution
class UpConv(torch.nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.up_conv = torch.nn.ConvTranspose2d(input_channels, input_channels // 2, kernel_size = 2, stride = 2)
        self.conv = DoubleConv(input_channels, output_channels)

    def forward(self, x1, x2):
        x1 = self.up_conv(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
  
        x = torch.cat([x2,x1], dim=1)
        
        return self.conv(x)

# Down Convolution
class DownConv(torch.nn.Module):
    def __init__(self,input_channels, output_channels):
        super().__init__()
        self.MaxPool2x2 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, padding = 1),
            DoubleConv(input_channels, output_channels),
        )

    def forward(self, x):
        return self.MaxPool2x2(x)
    
# Out Convolution 
class OutConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# FCN Model
class FCN(torch.nn.Module):
    # Initialize double convolution block, 2 down convolution blocks, 2 up convolution blocks, and out convolution block
    def __init__(self, n_input_channels = 3, n_classes = 5):
        super().__init__()
        
        # Following the UNet Architecture:
        self.init = DoubleConv(n_input_channels, 64)
        self.down_1 = DownConv(64,128)
        self.down_2 = DownConv(128,256)
        self.up_3 = UpConv(256, 128)
        self.up_4 = UpConv(128, 64)
        self.outConv = OutConv(64, 5)

        
    def forward(self, x):
        # print('Original Input = ', x.shape)
        x1 = self.init(x)
        # print('Double Convolution = ', x1.shape)
        x2 = self.down_1(x1)
        # print('1st Down Conv  = ',x2.shape)
        x3 = self.down_2(x2)
        # print('2nd Down Conv = ',x3.shape)
        x = self.up_3(x3, x2)
        # print('1st Up Conv  = ',x.shape)
        x = self.up_4(x, x1)
        # print('2nd Up Conv  = ',x.shape)
        classifier = self.outConv(x)
        # print('Out Convolution = ',classifier.shape)
        return classifier
    
    # """
        # FCN Output and Tips:
        # @x: torch.Tensor((B,3,H,W))
        # @return: torch.Tensor((B,5,H,W))
        # Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        # Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
        #       if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding strided
        #       convolution.
        # Use heavy augementation, skip connections, residual connections, and dropout to increase validation accuracy.
        # """



model_factory = {
    'cnn': CNNClassifier,
    'fcn': FCN,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
