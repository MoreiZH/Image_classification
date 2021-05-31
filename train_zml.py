#!usr/bin/python3
# -*- coding: utf-8 -*- 
"""
Project: resnet_zml
File: debug.py
IDE: PyCharm
Creator: morei
Email: zhangmengleiba361@163.com
Create time: 2021-02-22 12:28
Introduction:
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
from datasets.dataloder import LoadDataset
from torch.utils.data import DataLoader
from models.my_ResNet50 import resnet50


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    # 读取训练数据
    train_csv_path = './datasets/train.csv'
    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))
    trainset = LoadDataset(train_csv_path, img_transform=data_transform["train"])
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False,
                              num_workers=nw, drop_last=False)
    test_csv_path = './datasets/test.csv'
    # 读取验证数据
    valset = LoadDataset(test_csv_path,img_transform=data_transform["val"])
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False,
                            num_workers=nw, drop_last=False)
    print("using {} images for training, {} images for validation.".format(len(trainset),
                                                                           len(valset)))
    # 读取类别信息
    j_file = './output/torchID_wiki.json'
    cla_dict = json.load(open(j_file, 'r', encoding='utf-8'))

    net = resnet50()
    model_weight_path = "./models/pretrained_models/resnet50-19c8e357.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device))

    # change fc layer structure
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, len(cla_dict))    #zml nn.Linear(model, class_num)
    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)   

    epochs = 3
    best_acc = 0.0
    save_path = './output/resNet50.pth'
    train_steps = len(train_loader)   

    val_label_l = []
    predict_y_l = []
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch

        with torch.no_grad():
            val_bar = tqdm(val_loader, colour='green')
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        val_accurate = acc / len(valset)
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()

aa = 1
