import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import pickle

use_gpu = torch.cuda.is_available()
# ----------------------- PARAMETERS ------------------------ #
Krnl_sz, MP_factor, Pad, strd_sz, epoch_timer = [3, 2, 1, 1, 0]  # output_size = (W-F+2P)/S + 1; MP(2) /= 4
batch_size, num_classes, P_dropout, best_so_far = 100, 10, 0.25, 0.83
num_epoch, print_every = 100, 100  # Incremental display
base, num_MP_layers = 19, 3  # [Multiplication base, Max Pooling layers]
f_lay = [1, 1, 1, 2, 2, 2]  # Factor for every conv2d layer
fc = int(base * 2 ** (2 * len(f_lay) - 1) / (MP_factor ** (2 * num_MP_layers)))


def to_cuda(x):
    if use_gpu:
        x = x.cuda()
    return x


# ------------------- Neural Network Class ------------------ #
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # ------------------ Linear Layers ------------------ #
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, base * f_lay[0], kernel_size=Krnl_sz, stride=strd_sz, padding=Pad),
            nn.BatchNorm2d(base * f_lay[0]),
            nn.PReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(base * f_lay[0], base * f_lay[1], kernel_size=Krnl_sz, stride=strd_sz, padding=Pad),
            nn.BatchNorm2d(base * f_lay[1]),
            nn.PReLU())

        self.layer3 = nn.Sequential(
            nn.Conv2d(base * f_lay[1], base * f_lay[2], kernel_size=Krnl_sz, stride=strd_sz, padding=Pad),
            nn.BatchNorm2d(base * f_lay[2]),
            nn.PReLU())

        self.layer4 = nn.Sequential(
            nn.Conv2d(base * f_lay[2], base * f_lay[3], kernel_size=Krnl_sz, stride=strd_sz, padding=Pad),
            nn.BatchNorm2d(base * f_lay[3]),
            nn.PReLU(),
            nn.MaxPool2d(MP_factor))

        self.layer5 = nn.Sequential(
            nn.Conv2d(base * f_lay[3], base * f_lay[4], kernel_size=Krnl_sz, stride=strd_sz, padding=Pad),
            nn.BatchNorm2d(base * f_lay[4]),
            nn.PReLU(),
            nn.MaxPool2d(MP_factor))

        self.layer6 = nn.Sequential(
            nn.Conv2d(base * f_lay[4], base * f_lay[5], kernel_size=Krnl_sz, stride=strd_sz, padding=Pad),
            nn.BatchNorm2d(base * f_lay[5]),
            nn.PReLU(),
            nn.MaxPool2d(MP_factor))

        self.avg_pool = nn.AvgPool2d(kernel_size=1, stride=strd_sz)
        self.fc = nn.Linear(fc, num_classes)
        self.dropout = nn.Dropout(p=P_dropout)
        self.logsoftmax = nn.LogSoftmax(1)

    def forward(self, model_in):
        # Feature extraction
        out = self.layer1(model_in)
        out = self.layer2(out)
        out = self.dropout(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.dropout(out)
        out = self.layer5(out)
        out = self.layer6(out)
        # Classification
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)  # Flattening / Vectorizing final data
        out = self.dropout(out)
        out = self.fc(out)
        out = self.logsoftmax(out)
        return out


def evaluate_hw2():
    # Load model from file
    model = CNN()
    model = to_cuda(model)
    model.load_state_dict(torch.load('model.pkl'))
    model.eval()  # Sets model to test == Turns off Dropout, BatchNorm layers

    # Load and normalize the test data ([R, G, B] E mean, std)
    [RGB_mean, RGB_std] = [[0.4914, 0.4822, 0.4465], [0.247, 0.2434, 0.2615]]
    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(RGB_mean, RGB_std)])

    test_dataset = dsets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor(), download=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    ## work on GPU
    if torch.cuda.is_available():
        model = model.cuda()

    correct = 0
    total = 0
    for images, labels in test_loader:
        images = to_cuda(images)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum()

    model_accuracy = 100 * float(correct) / total

    return model_accuracy


Accuracy = evaluate_hw2()
print('Final Model Accuracy = %.3f [%%]' % Accuracy)
