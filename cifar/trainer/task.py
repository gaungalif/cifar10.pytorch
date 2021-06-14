import os

import torch
from tqdm import tqdm 

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train_batch(epoch, dataloader, net, criterion, optimizer, log_freq=2000):
    running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader, 0)):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        out = net(inputs)
        loss = criterion(out,labels)
        loss.backward()
        optimizer.step()

        # print statistics
        # print(loss)
        running_loss += loss.cpu().item()
        if i % log_freq == log_freq-1:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / log_freq))
            running_loss = 0.0
            
    return net, loss, running_loss
            
def valid_batch(epoch, dataloader, net, criterion, optimizer, log_freq=2000):
    net.eval()
    running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader, 0)):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        with torch.no_grad():
            out = net(inputs)
            loss = criterion(out,labels)

        # print statistics
        running_loss += loss.cpu().item()
        if i % log_freq == log_freq-1:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / log_freq))
            running_loss = 0.0
            
    return net, loss, running_loss


def train_network(epoch, tloader, vloader, net, criterion, optimizer, log_freq=2000):
    for ep in tqdm(range(epoch)):
        train_batch(ep, tloader, net, criterion, optimizer, log_freq=log_freq)
        valid_batch(ep, vloader, net, criterion, optimizer, log_freq=log_freq)

