import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt


class MyDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X_ = torch.tensor(self.X[index], dtype=torch.float32)
        y_ = torch.tensor(self.y[index], dtype=torch.float32)
        return X_, y_


class MyNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(14, 128)
        self.ac0 = nn.ReLU()
        self.hidden1 = nn.Linear(128, 4)
        self.ac1 = nn.ReLU()
        self.apply(self.init_weights)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.ac0(x)
        x = self.hidden1(x)
        x = self.ac1(x)
        return x

    @staticmethod
    def init_weights(layer):
        if type(layer)==nn.Linear:
            nn.init.normal_(layer.weight, std=0.01)


def cross_feature(x: np.array):
    return np.concatenate((
        x,
        np.multiply(x[:], x[:]),
        np.multiply(x[:, 0:3], x[:, 1:4]),
        np.multiply(x[:, 0:2], x[:, 2:4]),
        np.multiply(x[:, 0:1], x[:, 3:4]),
    ), axis=1)


def balance_label(y: np.array):
    return np.concatenate((
        y[:, [0]]*1e3-100,
        y[:, [1]]-50,
        y[:, [2]]*1e-5*0.2,
        y[:, [3]]*1e-5*0.2,
    ), axis=1)


def test_model(model, inputs):
    x = np.array([[*inputs]])
    x = Normalizer().fit_transform(cross_feature(x))
    y = model(torch.tensor(x, dtype=torch.float32)).data[0].tolist()
    return (y[0]+100)*1e-3, y[1]+50, y[2]*5e5, y[3]*5e5


def get_data(file_name, batch_size):
    df = pd.read_csv(file_name, delim_whitespace=True)
    X, y = df.iloc[:, :4].values, df.iloc[:,4:].values
    X = cross_feature(X)
    X = Normalizer().fit_transform(X)
    y = balance_label(y)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=0)
    train_dataset = MyDataset(train_X, train_y)
    test_dataset = MyDataset(test_X, test_y)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=test_dataset, batch_size=batch_size*2, shuffle=False)
    return train_loader, valid_loader


def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    train_loss = 0
    for X, y in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*X.size(0)
    train_loss /= len(train_loader.dataset)
    print(f'Epoch:{epoch+1:6}    Training Loss: {train_loss:10.3f}', end='\t')
    return train_loss


def validate(valid_loader, model, criterion):
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for X, y in valid_loader:
            loss = criterion(model(X), y)
            valid_loss += loss.item()*X.size(0)
    valid_loss /= len(valid_loader.dataset)
    print(f'Validation Loss: {valid_loss:10.3f}', end='\n')
    return valid_loss


def plot_data(x, y1, y2, label1='训练集误差', label2='测试集误差', xlabel='迭代次数', ylabel='均方误差',loc=4):
    # 数据
    plt.figure()
    plt.rcParams['font.family'] = ['SimSun']
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.unicode_minus'] = False
    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.15)
    plt.plot(x, y1, label=label1)
    plt.plot(x, y2, label=label2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc=loc)
    plt.show()


def main():
    # # 训练保存模型
    # # x, y1, y2 = [], [], []
    # train_loader, valid_loader = get_data('result.txt', 16)
    # model = MyNet()
    # criterion = nn.MSELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=0.001)
    # min_loss = 1e100
    # for epoch in range(250):
    #     train_loss = train(train_loader, model, criterion, optimizer, epoch)
    #     valid_loss = validate(valid_loader, model, criterion)
    #     max_loss = max(train_loss, valid_loss)
    #     # x.append(epoch+1)
    #     # y1.append(train_loss)
    #     # y2.append(valid_loss)
    #     if max_loss<min_loss:
    #         min_loss = max_loss
    #         torch.save(model.state_dict(), './temp.pt')
    # print(f'Min loss: {min_loss}\n')
    # # plot_data(x, y1, y2)

    # # 加载使用模型
    # model = MyNet()
    # model.load_state_dict(torch.load('model2.pt'))
    # print(*test_model(model, (3, 3, 4, 2)), sep='\t')
    # print(*test_model(model, (7, 9, 10, 6)), sep='\t')
    # print(*test_model(model, (5, 5, 5, 5)), sep='\t')

    # 绘制预测误差图
    model = MyNet()
    model.load_state_dict(torch.load('model2.pt'))
    df = pd.read_csv('result.txt', delim_whitespace=True)
    X, y = df.iloc[:, :4].values, df.iloc[:,4:].values
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.05, random_state=0)
    test_dataset = MyDataset(test_X, test_y)
    ylabel = {
        0:'质量（t）',
        1:'最大应力（MPa）',
        2:'Y向刚度（N/m）',
        3:'Z向刚度（N/m）',
    }
    for out_num in range(4):
        x, y1, y2 = [], [], []
        i = 1
        for input_, output_ in test_dataset:
            x.append(i)
            y1.append(output_[out_num])
            y2.append(test_model(model, input_)[out_num])
            i += 1
        plot_data(x, y1, y2, '真实值', '预测值', '机匣序号', ylabel[out_num])


if __name__ == '__main__':
    main()
