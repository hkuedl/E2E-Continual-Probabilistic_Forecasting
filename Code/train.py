import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import copy
import numpy as np


class likelihood(nn.Module):
    def __init__(self):
        super(likelihood, self).__init__()
    def forward(self, label, mu, sigma):
        distribution = torch.distributions.normal.Normal(mu,sigma)
        loss = distribution.log_prob(label)
        return -torch.mean(loss)


class pinball_loss(nn.Module):
    def __init__(self, quantile):
        super(pinball_loss, self).__init__()
        self.quantile = quantile
    def forward(self, label, forecasts):
        quantiles = torch.tensor(self.quantile, device=forecasts.device)
        # 将 labels 扩展到与 forecasts 相同的形状
        labels = label.expand_as(forecasts)
        
        # 计算误差
        errors = labels - forecasts
        
        # 计算 Pinball Loss
        loss = torch.max(quantiles * errors, (quantiles - 1) * errors)
        
        # 计算平均 Pinball Loss
        return loss.mean()



def train_parameter(args, model, train_loader, val_loader,dir_best_model='../Model/best_ann.pt'):
    best_val_loss = float('inf')
    counter = 0
    lr=args.lr
    device=args.device
    num_epochs=args.num_epochs
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    criterion = likelihood()
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            inputs = inputs.to(device).float()  # 确保输入数据是 Float 类型
            labels = labels.to(device).float()  # 确保标签数据是 Float 类型
            inputs=inputs.reshape(-1,inputs.shape[-1])
            labels=labels.reshape(-1,1)
            mu, sigma = model(inputs)
            mu = mu.reshape(-1,1)   
            sigma = sigma.reshape(-1,1)
            loss = criterion(labels,mu,sigma)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Train Loss: {train_loss}')
    
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device).float()
                labels = labels.to(device).float()
                inputs=inputs.reshape(-1,inputs.shape[-1])
                labels=labels.reshape(-1,1)
                mu, sigma = model(inputs)
                mu = mu.reshape(-1,1)   
                sigma = sigma.reshape(-1,1)
                loss = criterion(labels,mu,sigma)
                val_loss += loss.item()

            val_loss /= len(val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), dir_best_model)
            counter = 0
        else:
            counter += 1
            if counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                print(best_val_loss)
                break
    model.load_state_dict(torch.load(dir_best_model))

def train_parameter_seq2seq(args, model, train_loader, val_loader,dir_best_model='../Model/best_ann.pt'):
    best_val_loss = float('inf')
    counter = 0
    lr=args.lr
    device=args.device
    num_epochs=args.num_epochs
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    criterion = likelihood()
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            inputs = inputs.to(device).float()  # 确保输入数据是 Float 类型
            labels = labels.to(device).float()  # 确保标签数据是 Float 类型
            mu, sigma = model(inputs)
            mu = mu.squeeze()
            sigma = sigma.squeeze()
            labels = labels.squeeze()
            loss = criterion(labels,mu,sigma)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Train Loss: {train_loss}')
    
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device).float()
                labels = labels.to(device).float()
                mu, sigma = model(inputs)
                mu = mu.squeeze()
                sigma = sigma.squeeze()
                labels = labels.squeeze()
                loss = criterion(labels,mu,sigma)
                val_loss += loss.item()

            val_loss /= len(val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), dir_best_model)
            counter = 0
        else:
            counter += 1
            if counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                print(best_val_loss)
                break
    model.load_state_dict(torch.load(dir_best_model))

def train_non_parameter(args, model, train_loader, val_loader,dir_best_model='../Model/best_ann.pt'):
    best_val_loss = float('inf')
    counter = 0
    device=args.device
    lr=args.lr
    num_epochs=args.num_epochs
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    quantiles=args.quantiles    
    criterion = pinball_loss(quantiles)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            inputs = inputs.to(device).float()  # 确保输入数据是 Float 类型
            labels = labels.to(device).float()  # 确保标签数据是 Float 类型
            inputs=inputs.reshape(-1,inputs.shape[-1])
            labels=labels.reshape(-1,1)
            forecasts = model(inputs)

            loss = criterion(labels,forecasts)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Train Loss: {train_loss}')
    
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device).float()
                labels = labels.to(device).float()
                inputs=inputs.reshape(-1,inputs.shape[-1])
                labels=labels.reshape(-1,1)
                forecasts = model(inputs)
                loss = criterion(labels,forecasts)
                val_loss += loss.item()

            val_loss /= len(val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), dir_best_model)
            counter = 0
        else:
            counter += 1
            if counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                print(best_val_loss)
                break
    model.load_state_dict(torch.load(dir_best_model))

def train_non_parameter_dynamic_price(args, model, train_loader, val_loader,dir_best_model='../Model/best_ann.pt'):
    best_val_loss = float('inf')
    counter = 0
    device=args.device
    lr=args.lr
    num_epochs=args.num_epochs
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    quantiles=args.quantiles    
    criterion = pinball_loss(quantiles)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            inputs = inputs.to(device).float()  # 确保输入数据是 Float 类型
            labels = labels.to(device).float()  # 确保标签数据是 Float 类型
            inputs=inputs.reshape(-1,inputs.shape[-1])
            labels=labels.reshape(-1,1)
            forecasts = model(inputs)

            loss = criterion(labels,forecasts)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Train Loss: {train_loss}')
    
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device).float()
                labels = labels.to(device).float()
                inputs=inputs.reshape(-1,inputs.shape[-1])
                labels=labels.reshape(-1,1)
                forecasts = model(inputs)
                loss = criterion(labels,forecasts)
                val_loss += loss.item()

            val_loss /= len(val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), dir_best_model)
            counter = 0
        else:
            counter += 1
            if counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                print(best_val_loss)
                break
    model.load_state_dict(torch.load(dir_best_model))


def test_parameter(args,model, test_loader):
    model.eval()
    predictions_mean = []
    predictions_log_var = []
    targets = []
    device=args.device
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device).float()  # 确保输入数据是 Float 类型
            labels = labels.to(device).float()  # 确保标签数据是 Float 类型
            inputs=inputs.reshape(-1,inputs.shape[-1])
            labels=labels.reshape(-1,1)
            mu, sigma = model(inputs)
            
            predictions_mean.append(mu)
            predictions_log_var.append(sigma)
            targets.append(labels)
    
    predictions_mean = torch.cat(predictions_mean, dim=0)
    predictions_log_var = torch.cat(predictions_log_var, dim=0)
    targets = torch.cat(targets, dim=0)
    
    return predictions_mean, predictions_log_var, targets

def test_parameter_seq2seq(args,model, test_loader):
    model.eval()
    predictions_mean = []
    predictions_log_var = []
    targets = []
    device=args.device
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device).float()  # 确保输入数据是 Float 类型
            labels = labels.to(device).float()  # 确保标签数据是 Float 类型
            mu, sigma = model(inputs)
            mu = mu.squeeze()
            sigma = sigma.squeeze()
            labels = labels.squeeze()

            if mu.dim() == 1:
                mu = mu.unsqueeze(0)
            if sigma.dim() == 1:
                sigma = sigma.unsqueeze(0)
            if labels.dim() == 1:
                labels = labels.unsqueeze(0)

            predictions_mean.append(mu)
            predictions_log_var.append(sigma)
            targets.append(labels)
    
    predictions_mean = torch.cat(predictions_mean, dim=0)
    predictions_log_var = torch.cat(predictions_log_var, dim=0)
    targets = torch.cat(targets, dim=0)
    
    return predictions_mean, predictions_log_var, targets

def test_non_parameter(args, model, test_loader):
    model.eval()
    predictions = []
    targets = []
    device=args.device
    quantiles=args.quantiles
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device).float()
            labels = labels.to(device).float()
            inputs=inputs.reshape(-1,inputs.shape[-1])
            labels=labels.reshape(-1,labels.shape[-1])
            forecasts = model(inputs)
            forecasts = forecasts.reshape(-1,24,len(quantiles))
            labels = labels.reshape(-1,24)
            predictions.append(forecasts)
            targets.append(labels)

    predictions = torch.cat(predictions, dim=0)
    targets = torch.cat(targets, dim=0)
    return predictions, targets
        