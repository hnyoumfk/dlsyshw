import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    seq = nn.Sequential(
        nn.Linear(dim, hidden_dim) 
        , norm(hidden_dim)
        , nn.ReLU()
        , nn.Dropout(drop_prob)
        , nn.Linear(hidden_dim, dim)
        , norm(dim)
        )
    return nn.Residual(seq)
    ### END YOUR SOLUTION


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    mod_list = []
    mod_list.append(nn.Linear(dim, hidden_dim))
    mod_list.append(nn.ReLU())
    in_dim = hidden_dim
    out_dim = in_dim // 2
    for _ in range(num_blocks) :
        mod_list.append(ResidualBlock(in_dim, out_dim, norm, drop_prob))
        in_dim = out_dim
        out_dim = in_dim // 2
    mod_list.append(nn.Linear(out_dim, num_classes))
    return nn.Sequential(*mod_list)
    ### END YOUR SOLUTION




def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt is None:
        model.eval()
    else:
        model.train()

    loss_fun = ndl.nn.SoftmaxLoss()
    err_cnt = 0
    loss_sum = 0
    for batch in dataloader:
        batch_x, batch_y = batch[0], batch[1]
        batch_x = batch_x.reshape(-1, 784)
        pred_y = model(batch_x)
        pred_clz_y = np.argmax(pred_y.data, axis=1)
        err_cnt += np.count_nonzero(batch_y.data - pred_clz_y.data)
        loss = loss_fun(pred_y, batch_y)
        loss_sum += loss.data * batch_y.shape[0]
        if opt is not None:
            loss.backward()
            opt.step()
    
    total = len(dataloader)
    return err_cnt / total , loss_sum / total
    ### END YOUR SOLUTION



def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_dataset = ndl.data.MNISTDataset(data_dir+'/train-images-idx3-ubyte.gz', data_dir+'/train-labels-idx1-ubyte.gz')
    train_dataloader = ndl.data.Dataloader(train_dataset, batch_size, True)

    test_dataset = ndl.data.MNISTDataset(data_dir+'/t10k-images-idx3-ubyte.gz', data_dir+'/t10k-labels-idx1-ubyte.gz')
    test_dataloader = ndl.data.Dataloader(test_dataset, batch_size, True)

    model = MLPResNet(dim=784, hidden_dim=hidden_dim, num_classes=10)
    
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_err_rate = 0
    train_loss = 0
    for _ in range(epochs):
        train_err_rate, train_loss = epoch(train_dataloader, model, opt)
    
    test_err_rate, test_loss = epoch(test_dataloader, model)

    return train_err_rate, train_loss, test_err_rate, test_loss
    ### END YOUR SOLUTION



if __name__ == "__main__":
    train_mnist(data_dir="../data")
