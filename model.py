import torch

class ResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = torch.nn.Conv2d(in_channel=3, out_channel=64, kernel_size =7, stride=2)

        self.ResBlock1 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 1)
            torch.nn.BatchNorm2d(),
            torch.nn.ReLU(inplace=True),
            #torch.nn.MaxPool2d(3,2)
            torch.nn.Conv2d(64, 128, 2)
            torch.nn.BatchNorm2d()
        )
        self.ResBlock2 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, 2)
            torch.nn.BatchNorm2d(),
            torch.nn.ReLU(inplace=True),
            #torch.nn.MaxPool2d(3,2)

            torch.nn.Conv2d(256, 512, 2)
            torch.nn.BatchNorm2d()
        )

        self.FC = torch.nn.Linear(512, 2)

    def forward(self, x):
        x = torch.nn.MaxPool2d(torch.nn.functional.relu(self.conv(x)), (3, 2))
        x = torch.nn.MaxPool2d(torch.nn.functional.relu(self.ResBlock1(x)), (3, 2))
        x = torch.nn.AvgPool2d(torch.nn.functional.relu(self.ResBlock2(x)), (3, 2))

        x = torch.flatten(x)
        x = self.FC(x)
        x = torch.nn.functional.softmax(x)

        return x



