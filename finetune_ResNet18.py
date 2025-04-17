import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision.models import ResNet18_Weights
import copy
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "1"

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((224,224)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.ImageFolder(root='./data/ex7-carTypes/carTypes/train', transform=transform)
testset = torchvision.datasets.ImageFolder(root='./data/ex7-carTypes/carTypes/val', transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=1)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=1)
# dataloaders = {'train':trainloader, 'val':testloader}
# dataset_sizes = {'train': len(trainset), 'val': len(testset)}
# print(dataset_sizes)
print(trainset.classes)
classes = trainset.classes

if __name__ == '__main__':
    model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    inchannel = model.fc.in_features
    model.fc = nn.Linear(inchannel, 6)
    model.cuda()
    # -----定义优化器和Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    # -----训练--验证-----
    num_epochs = 20
    device = 'cuda:0'# 'cpu'
    best_acc = 0
    for epoch in range(num_epochs):
        message = 'Epoch {}/{} '.format(epoch+1, num_epochs)
        # Each epoch has a training and validation phase
        model.train()
        train_loss = 0.0
        train_corrects = 0
        test_loss = 0.0
        test_corrects = 0
        batch = 0
        for inputs, labels in trainloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(preds == labels.data)
            batch += 1
            if batch%20==0:
                print(f'Epoch: {epoch},Training Batch: {batch},Loss:{loss.item()}')
        scheduler.step()   
        
        train_loss = train_loss / len(trainloader)
        train_loss = train_corrects.double() / len(trainloader.dataset)
        message += ' {} Loss: {:.4f} Acc: {:.4f}'.format(
                'train', train_loss, train_loss)

        model.eval()
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                test_corrects += torch.sum(preds == labels.data)

        test_loss = test_loss / len(testloader)
        test_acc = test_corrects.double() / len(testloader.dataset)
        message += ' {} Loss: {:.4f} Acc: {:.4f}'.format(
                'test', test_loss, test_acc)
        
        if test_acc > best_acc:
                best_acc = test_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts,'mybestmodel.pt')
        print(message)
        
        

