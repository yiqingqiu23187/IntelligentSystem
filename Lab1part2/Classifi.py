import torchvision  # 数据库模块
import torch
from torch.autograd import Variable
from CNN import *
import torch.utils.data as data
import torchvision.datasets as dset
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(1)  # reproducible

# Hyper Parameters
EPOCH = 20
BATCH_SIZE = 5
LR = 0.001

train_loader = data.DataLoader(dataset=dset.ImageFolder('D:/PJ/IntelliLabs/lab1/train', transform=torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(1),  # 单通道
    torchvision.transforms.ToTensor(),  # 将图片数据转成tensor格式
])), batch_size=BATCH_SIZE, shuffle=True)

test = data.DataLoader(dataset=dset.ImageFolder('D:/PJ/IntelliLabs/lab1/test', transform=torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(1),  # 单通道
    torchvision.transforms.ToTensor(),  # 将图片数据转成tensor格式
])))

pred_file = open('pred.txt', 'w')
map = {0: 1, 1: 10, 2: 11, 3: 12, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8, 11: 9}

cnn = CNN()
# optimizer
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
# optimizer = torch.optim.Adam(cnn.parameters(), lr=LR, weight_decay=1e-5)#L2正则化
# optimizer = torch.optim.SGD(cnn.parameters(), lr = LR)#随机梯度下降
# optimizer = torch.optim.ASGD(params, lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)#平均随机梯度下降算法
# optimizer = torch.optim.Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0)#AdaGrad算法
# optimizer = torch.optim.Adadelta(params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)#自适应学习率调整 Adadelta算法
# optimizer = torch.optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)#RMSprop算法
# optimizer = torch.optim.Adamax(params, lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)#Adamax算法（Adamd的无穷范数变种
# optimizer = torch.optim.SparseAdam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08)#SparseAdam算法
# optimizer = torch.optim.LBFGS(params, lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-05, tolerance_change=1e-09, history_size=100, line_search_fn=None)#L-BFGS算法
# optimizer = torch.optim.Rprop(params, lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50))#弹性反向传播算法

# loss_fun
loss_func = nn.CrossEntropyLoss()

pltx = []
pltloss = []
pltaccuracy = []

# training loop
lastaccuracy = 0
for epoch in range(EPOCH):
    for i, (x, y) in enumerate(train_loader):
        batch_x = Variable(x)
        batch_y = Variable(y)
        # 输入训练数据
        output = cnn(batch_x)

        #使用L1正则化
        # reg_loss = 0
        # for papam in cnn.parameters():
        #     reg_loss += torch.sum(torch.abs(papam))
        # classify_loss = loss_func(output, batch_y)
        # loss = classify_loss + 0.01 * reg_loss

        # 计算误差
        loss = loss_func(output, batch_y)
        # 清空上一次梯度
        optimizer.zero_grad()
        # 误差反向传递
        loss.backward()
        # 优化器参数更新
        optimizer.step()


for step, (test_x, test_y) in enumerate(test):
    test_output = cnn(test_x)
    pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
    pred_file.write(str(map[int(pred_y)]) + '\n')
    # print("Epoch " + str(epoch) + "/" + str(EPOCH))
    # count = 0
    # totalloss = 0
    # for i, (x, y) in enumerate(test_x):
    #     batch_x = Variable(x)
    #     batch_y = Variable(y)
    #     output = cnn(batch_x)
    #     # 计算误差
    #     totalloss += loss_func(output, batch_y).item()
    #
    #     index = torch.max(output, 1)[1].data.numpy().squeeze()
    #     if index == y.item():
    #         count += 1
    # print("average loss: " + str(totalloss / len(test_x.dataset)))
    # print("accuracy: " + str(count / len(test_x.dataset)))
    # pltx.append(epoch + 1)
    # pltloss.append(totalloss / len(test_x.dataset))
    # pltaccuracy.append(count / len(test_x.dataset))

    #早停法
    # if count / len(test_x.dataset) <= lastaccuracy:
    #     break
    # else:
    #     lastaccuracy = count / len(test_x.dataset)

# draw
# plt.xlabel("epoch")
# plt.ylabel("average loss")
# plt.title('average loss')
# plt.legend()
# plt.plot(pltx, pltloss)
# plt.show()
#
# plt.xlabel("epoch")
# plt.ylabel("accuracy")
# plt.title('accuracy')
# plt.plot(pltx, pltaccuracy)
# plt.show()
