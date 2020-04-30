import torch, pickle  # For imported functions !
from utils import *
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.datasets as dsets
import torchvision.transforms as transforms

# ----------------- GLOBAL INITIALIZATION ------------------- #
# Define normalization properties
[RGB_mean, RGB_std] = [[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]]

# Load and normalize the train data ([R, G, B] E mean, std)
Normalize = transforms.Normalize(RGB_mean, RGB_std)
transform_train = transforms.Compose([transforms.RandomCrop(32, padding=2),
                                      transforms.RandomHorizontalFlip(), transforms.ToTensor(), Normalize])

# Load and normalize the test data ([R, G, B] E mean, std)
transform_test = transforms.Compose([transforms.ToTensor(), Normalize])

# ----------------------- PARAMETERS ------------------------ #
Krnl_sz, MP_factor, Pad, strd_sz, epoch_timer = [3, 2, 1, 1, 0]  # output_size = (W-F+2P)/S + 1; MP(2) /= 4
batch_size, num_classes, P_dropout, best_so_far = 100, 10, 0.25, 0.83
num_epoch, print_every = 100, 100  # Incremental display
base, num_MP_layers = 19, 3  # [Multiplication base, Max Pooling layers]
f_lay = [1, 1, 1, 2, 2, 2]  # Factor for every conv2d layer
fc = int(base * 2 ** (2 * len(f_lay) - 1) / (MP_factor ** (2 * num_MP_layers)))

# CIFAR-10 dataset - ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck’]
train_dataset = dsets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = dsets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor(), download=True)

# Data loaders for train and test
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# ------------------ Global Initialization ------------------ #
train_error, train_loss, test_error, test_loss, train_accuracy, test_accuracy, lr_go \
    = [np.zeros(num_epoch) for _ in range(0, 7)]


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


model = to_cuda(CNN())
print('Number of parameters in model : ', sum(param.numel() for param in model.parameters()))
learning_rate = 0.001
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Load for a continuation
# model, optimizer = load_checkpoint(model, optimizer, 'Model_Online.pkl')

# SGD : optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.5)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)


def train():
    model.train()  # Sets model to train
    # scheduler.step()
    lr_go[epoch] = learning_rate  # np.asarray(scheduler.get_lr())
    train_correct, total_train_epoch, correct_train, train_loss_epoch, epoch_test = [0, 0, 0, 0, 0]
    for i, (images, labels) in enumerate(train_loader):
        images, labels = to_cuda(images), to_cuda(labels)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % print_every == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss %.4f'
                  % (epoch + 1, num_epoch, i + 1, len(train_loader), loss.item()))

        # Augment periodical TRAIN loss and error
        total_train_epoch += labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == labels).sum().item()
        train_loss_epoch += loss.item()

    # ------- Summarize TRAIN mean data of each epoch -------- #
    train_accuracy[epoch] = train_correct / total_train_epoch
    train_error[epoch] = 1 - train_accuracy[epoch]
    train_loss[epoch] = train_loss_epoch / (i + 1)


def test(test_flag):
    model.eval()  # Sets model to test == Turns off Dropout, BatchNorm layers
    test_correct, test_loss_epoch, total_epoch_test = [0, 0, 0]
    for i, (images, labels) in enumerate(test_loader):
        images, labels = to_cuda(images), to_cuda(labels)
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Augment periodical TEST loss and error
        test_loss_epoch += loss.item()
        total_epoch_test += labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        test_correct += (predicted == labels).sum().item()

    # -------- Summarize TEST mean data of each epoch -------- #
    test_accuracy[epoch] = test_correct / total_epoch_test
    test_error[epoch] = 1 - test_accuracy[epoch]
    test_loss[epoch] = test_loss_epoch / (i + 1)

    if test_flag == 'Validation':
        print('\n-------------- Epoch [{}] @ {} --------------'.format(epoch + 1, time_since(epoch_timer)))
        print('Train Loss = %.4f \tTrain Accuracy = %.4f\n'
              'Test  Loss = %.4f \tTest  Accuracy = %.4f' %
              (train_loss[epoch], train_accuracy[epoch],
               test_loss[epoch], test_accuracy[epoch]))
        print('----------------- lr = %.5f ------------------\n' % lr_go[epoch])
    else:
        print('----------- Model Accuracy: %.2f [%%] -----------\n' % (100 * test_accuracy[epoch]))


def plot_graphs():
    # ------------------- Plot LOSS graph -------------------- #
    y_i = epoch + 1
    x_i = range(0, y_i)
    fig1, ax1 = plt.subplots()
    ax1.plot(x_i, train_loss[0:y_i], '-', marker='+', markersize=7, linewidth=3, label='Train Loss')
    ax1.plot(x_i, test_loss[0:y_i], 'r-', marker='+', markersize=7, linewidth=3, label='Test Loss')
    ax1.set_xlabel('Epoch #', fontsize=14)
    ax1.set_ylabel('Loss', fontsize=14)
    ax1.set_title('Loss vs. Epochs', fontsize=17)
    ax1.grid()

    ax_1 = ax1.twinx()
    ax_1.plot(x_i, lr_go[0:y_i], 'g--', markersize=7, linewidth=2.5, label='Learning Rate')
    ax_1.set_ylabel('Learning Rate', color='g', fontsize=14)
    ax_1.tick_params('y', colors='g')
    ax1.legend()
    fig1.savefig("Loss_graph.png")
    plt.show()

    # ------------------- Plot ERROR graph ------------------- #
    fig2, ax2 = plt.subplots()
    ax2.plot(x_i, train_error[0:y_i], '-', marker='+', markersize=7, linewidth=3, label='Train Error')
    ax2.plot(x_i, test_error[0:y_i], 'r', marker='+', markersize=7, linewidth=3, label='Test Error')
    ax2.set_xlabel('Epoch #', fontsize=14)
    ax2.set_ylabel('Error', fontsize=14)
    ax2.set_title('Error vs. Epochs', fontsize=17)
    ax2.grid()

    ax_2 = ax2.twinx()
    ax_2.plot(x_i, lr_go[0:y_i], 'g--', markersize=7, linewidth=2.5, label='Learning Rate')
    ax_2.set_ylabel('Learning Rate', color='g', fontsize=14)
    ax_2.tick_params('y', colors='g')
    ax2.legend()
    fig2.savefig("Error_graph.png")
    plt.show()


def save_params():
    # Save ongoing variables
    with open('Model_Params.csv', 'wb') as file_variable:
        pickle.dump([test_loss, train_loss, test_error, train_error, lr_go], file_variable)


if __name__ == '__main__':
    for epoch in range(num_epoch):
        epoch_timer = time.time()
        train()
        test('Validation')
        plot_graphs()
        # ------------ Save Model and Parameters ------------- #
        save_params()
        if test_error[epoch] < best_so_far:
            best_so_far = test_error[epoch]
            save_model(model, 'Model_Online_i.pkl')
            save_checkpoint(model, optimizer, 'Model_Online.pkl') # Consider -> else(): lr_decrease
    test('Final Test')
