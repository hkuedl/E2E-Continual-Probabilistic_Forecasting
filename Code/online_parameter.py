
import torch
import numpy as np
from test_parameter import *
from utils import *
import time
from train import *
import torch.nn.functional as F

def Traditional_online_parameter(args, combined_model, combined_fine_tune_loader, combined_test_loader, scaler_y_pv, scaler_y_load,time_flag=False):    
    device = args.device
    cost_list = []
    solution_list_ahead = []
    solution_list_intra = []
    forecasts_pv={}
    forecasts_load={}
    for i in args.quantiles:
        forecasts_pv[i]=[]
        forecasts_load[i]=[]
    batch_idx=0
    time_list=[]

    optimizer_pv = torch.optim.SGD([
    {'params': combined_model.model_pv.parameters()}
    ], lr=args.ft_lr)

    optimizer_load = torch.optim.SGD([
    {'params': combined_model.model_load.parameters()}
    ], lr=args.ft_lr)


    data_iter = iter(combined_test_loader)
    next(data_iter)
    plot_flag=False
    for input_pv, input_load, labels_pv, labels_load,alpha,alpha_z,rho_z,rho_r in combined_fine_tune_loader:
        batch_idx +=1

        print('=======================')
        print('New test batch',batch_idx)
        combined_model.train()
        input_pv = input_pv.to(device).float()
        input_load = input_load.to(device).float()
        labels_pv = labels_pv.to(device).float()
        labels_load = labels_load.to(device).float()

        try:
            input_pv_test, input_load_test, labels_pv_test, labels_load_test,alpha_test,alpha_z_test,rho_z_test,rho_r_test = next(data_iter)
            # 处理数据
        except StopIteration:
            print("No more data to read.")
            break
        input_pv_test = input_pv_test.to(device).float()
        input_load_test = input_load_test.to(device).float()
        labels_pv_test = labels_pv_test.to(device).float()
        labels_load_test = labels_load_test.to(device).float()
        #alpha_test = alpha_test.to(device).float()
        #alpha_z_test = alpha_z_test.to(device).float()
        #rho_r_test = rho_r_test.to(device).float()
        #rho_z_test = rho_z_test.to(device).float()
        test_data=[input_pv_test, input_load_test, labels_pv_test, labels_load_test,alpha_test,alpha_z_test,rho_z_test,rho_r_test]
        
        for i in range(1):
            start_time = time.time()
            combined_model.model_pv, _ = traditional_fine_tuning_parameter(combined_model.model_pv, input_pv, labels_pv, optimizer_pv)
            combined_model.model_load, _ = traditional_fine_tuning_parameter(combined_model.model_load, input_load, labels_load, optimizer_load)
            end_time = time.time()
            time_list.append(end_time-start_time)

        print('Accuracy after fine tune')
        set_seed(42)
        with torch.no_grad():
            solution_ahead, solution_intra, obj,quantiles_list_pv,quantiles_list_load,_,_ = Evaluation_parameter_dynamic_price(args, combined_model, test_data, scaler_y_pv, scaler_y_load,plot_flag=plot_flag,accuracy_flag=True)
        if solution_list_ahead==[]:
            solution_list_ahead=solution_ahead
        else:
            for i in solution_ahead.keys():   
                solution_list_ahead[i]=torch.cat([solution_list_ahead[i],solution_ahead[i]])

        if solution_list_intra==[]:
            solution_list_intra=solution_intra
        else:
            for i in solution_intra.keys():   
                solution_list_intra[i]=torch.cat([solution_list_intra[i],solution_intra[i]])

        print(f"Test Objective: {obj.mean().item()}")
        cost_list+=list(obj.detach().cpu().numpy())
        
        for i in quantiles_list_pv.keys():
            forecasts_pv[i]=list(forecasts_pv[i])+list(quantiles_list_pv[i])
            forecasts_load[i]=list(forecasts_load[i])+list(quantiles_list_load[i])

    if time_flag:
        return solution_list_ahead,solution_list_intra, cost_list, forecasts_pv,forecasts_load,time_list
    else:
        return solution_list_ahead,solution_list_intra, cost_list, forecasts_pv,forecasts_load



def E2E_parameter(args, combined_model, combined_fine_tune_loader, combined_test_loader, scaler_y_pv, scaler_y_load,time_flag=False):
    device = args.device
    cost_list = []
    solution_list_ahead = []
    solution_list_intra = []
    forecasts_pv={}
    forecasts_load={}
    for i in args.quantiles:
        forecasts_pv[i]=[]
        forecasts_load[i]=[]
    time_list=[]
    
    optimizer = torch.optim.SGD([
        {'params': combined_model.model_pv.parameters()},
        {'params': combined_model.model_load.parameters()},
    ], lr=args.e2e_lr)
    # optimizer_mu = torch.optim.SGD([
    #      {'params': combined_model.model_load.distribution_mu.parameters()},
    #     {'params': combined_model.model_pv.distribution_mu.parameters()}
    # ], lr=args.e2e_lr)
    
    #optimizer_sigma_pv = torch.optim.SGD(combined_model.model_pv.distribution_presigma.parameters(), lr=args.ft_lr)
    #optimizer_sigma_load = torch.optim.SGD(combined_model.model_load.distribution_presigma.parameters(), lr=args.ft_lr)

    plot_flag=False
    for i in range(3):
        print('===============================')
        print("e2e epoch:",i)
        print('===============================')
        for input_pv, input_load, labels_pv, labels_load,alpha,alpha_z,rho_z,rho_r in combined_fine_tune_loader:
            combined_model.train()
            input_pv = input_pv.to(device).float()
            input_load = input_load.to(device).float()
            labels_pv = labels_pv.to(device).float()
            labels_load = labels_load.to(device).float()
            alpha = alpha.to(device).float()
            alpha_z = alpha_z.to(device).float()
            rho_r = rho_r.to(device).float()    
            rho_z = rho_z.to(device).float()
            start_time = time.time()
            combined_model, _ = E2E_fine_tuning_parameter(args, combined_model, [input_pv, input_load, labels_pv, labels_load,alpha,alpha_z,rho_z,rho_r], scaler_y_pv, scaler_y_load, optimizer)
            end_time = time.time()
            time_list.append(end_time-start_time)

    data_iter = iter(combined_test_loader)
    next(data_iter)
    while True:
        try:
            input_pv_test, input_load_test, labels_pv_test, labels_load_test, alpha_test,alpha_z_test,rho_z_test,rho_r_test= next(data_iter)
            # 处理数据
        except StopIteration:
            print("No more data to read.")
            break
            
        input_pv_test = input_pv_test.to(device).float()
        input_load_test = input_load_test.to(device).float()
        labels_pv_test = labels_pv_test.to(device).float()
        labels_load_test = labels_load_test.to(device).float()
        #alpha_test = alpha_test.to(device).float()
        #alpha_z_test = alpha_z_test.to(device).float()
        #rho_r_test = rho_r_test.to(device).float()
        #rho_z_test = rho_z_test.to(device).float()
        test_data=[input_pv_test, input_load_test, labels_pv_test, labels_load_test,alpha_test,alpha_z_test,rho_z_test,rho_r_test]
        
        print('Accuracy after fine tune')
        set_seed(42)
        with torch.no_grad():
            solution_ahead,solution_intra, obj,quantiles_list_pv,quantiles_list_load,_,_ = Evaluation_parameter_dynamic_price(args, combined_model, test_data, scaler_y_pv, scaler_y_load,plot_flag=plot_flag,accuracy_flag=True)
        if solution_list_ahead==[]:
            solution_list_ahead=solution_ahead
        else:
            for i in solution_ahead.keys():   
                solution_list_ahead[i]=torch.cat([solution_list_ahead[i],solution_ahead[i]])

        if solution_list_intra==[]:
            solution_list_intra=solution_intra
        else:
            for i in solution_intra.keys():   
                solution_list_intra[i]=torch.cat([solution_list_intra[i],solution_intra[i]])

        print(f"Test Objective: {obj.mean().item()}")
        cost_list+=list(obj.detach().cpu().numpy())
        
        for i in quantiles_list_pv.keys():
            forecasts_pv[i]=list(forecasts_pv[i])+list(quantiles_list_pv[i])
            forecasts_load[i]=list(forecasts_load[i])+list(quantiles_list_load[i])
    if time_flag:
        return solution_list_ahead,solution_list_intra, cost_list, forecasts_pv,forecasts_load,time_list
    else:
        return solution_list_intra,solution_list_intra, cost_list, forecasts_pv,forecasts_load


def E2E_online_parameter(args, combined_model, combined_fine_tune_loader, combined_test_loader,  scaler_y_pv, scaler_y_load,time_flag=False):
    device = args.device
    cost_list = []
    solution_list_ahead = []
    solution_list_intra = []
    forecasts_pv={}
    forecasts_load={}
    for i in args.quantiles:
        forecasts_pv[i]=[]
        forecasts_load[i]=[]
    batch_idx=0
    time_list=[]

    optimizer = torch.optim.SGD([
         {'params': combined_model.model_load.parameters()},
         {'params': combined_model.model_pv.parameters()}
     ], lr=args.e2e_ft_lr)

    # optimizer_mu = torch.optim.SGD([
    #      {'params': combined_model.model_load.distribution_mu.parameters()},
    #     {'params': combined_model.model_pv.distribution_mu.parameters()}
    # ], lr=args.e2e_ft_lr)
    
    #optimizer_sigma_pv = torch.optim.SGD(combined_model.model_pv.distribution_presigma.parameters(), lr=args.ft_lr)
    #optimizer_sigma_load = torch.optim.SGD(combined_model.model_load.distribution_presigma.parameters(), lr=args.ft_lr)
    data_iter = iter(combined_test_loader)
    next(data_iter)
    plot_flag=False
    for input_pv, input_load, labels_pv, labels_load,alpha,alpha_z, rho_z,rho_r in combined_fine_tune_loader:
        batch_idx +=1
        
        print('=======================')
        print('New test batch',batch_idx)
        combined_model.train()
        input_pv = input_pv.to(device).float()
        input_load = input_load.to(device).float()
        labels_pv = labels_pv.to(device).float()
        labels_load = labels_load.to(device).float()
        alpha = alpha.to(device).float()
        alpha_z = alpha_z.to(device).float()
        rho_z = rho_z.to(device).float()
        rho_r = rho_r.to(device).float()
        try:
            input_pv_test, input_load_test, labels_pv_test, labels_load_test,alpha_test,alpha_z_test, rho_z_test,rho_r_test = next(data_iter)
            
            # 处理数据
        except StopIteration:
            print("No more data to read.")
            break
        input_pv_test = input_pv_test.to(device).float()
        input_load_test = input_load_test.to(device).float()
        labels_pv_test = labels_pv_test.to(device).float()
        labels_load_test = labels_load_test.to(device).float()
        test_data=[input_pv_test, input_load_test, labels_pv_test, labels_load_test,alpha_test,alpha_z_test, rho_z_test,rho_r_test]
        # #print('Accuracy before fine tune')

        for i in range(1):
            start_time = time.time()
            combined_model, _ = E2E_fine_tuning_parameter(args, combined_model, [input_pv, input_load, labels_pv, labels_load,alpha,alpha_z,rho_z,rho_r], scaler_y_pv, scaler_y_load, optimizer)
            end_time = time.time()
            time_list.append(end_time-start_time)
            #combined_model.model_pv, _ = traditional_fine_tuning_parameter(combined_model.model_pv, input_pv, labels_pv, optimizer_sigma_pv)
            #combined_model.model_load, _ = traditional_fine_tuning_parameter(combined_model.model_load, input_load, labels_load, optimizer_sigma_load)

        print('Accuracy after fine tune')
        set_seed(42)
        with torch.no_grad():
            solution_ahead,solution_intra, obj,quantiles_list_pv,quantiles_list_load,_,_ = Evaluation_parameter_dynamic_price(args, combined_model, test_data, scaler_y_pv, scaler_y_load,plot_flag=plot_flag,accuracy_flag=True)
        if solution_list_ahead==[]:
            solution_list_ahead=solution_ahead
        else:
            for i in solution_ahead.keys():   
                solution_list_ahead[i]=torch.cat([solution_list_ahead[i],solution_ahead[i]])
        
        if solution_list_intra==[]:
            solution_list_intra=solution_intra
        else:
            for i in solution_intra.keys():   
                solution_list_intra[i]=torch.cat([solution_list_intra[i],solution_intra[i]])
        
        print(f"Test Objective: {obj.mean().item()}")
        cost_list+=list(obj.detach().cpu().numpy())
        
        for i in quantiles_list_pv.keys():
            forecasts_pv[i]=list(forecasts_pv[i])+list(quantiles_list_pv[i])
            forecasts_load[i]=list(forecasts_load[i])+list(quantiles_list_load[i])
    if time_flag:
        return solution_list_ahead,solution_list_intra, cost_list, forecasts_pv,forecasts_load,time_list
    else:
        return solution_list_ahead,solution_list_intra, cost_list, forecasts_pv,forecasts_load


def E2E_fine_tuning_parameter(args, combined_model, data, scaler_y_pv,scaler_y_load, optimizer):
    optimizer.zero_grad()
    input_pv=data[0]
    input_load=data[1]
    labels_pv=data[2]
    labels_load=data[3]
    alpha=data[4]
    alpha_z=data[5]
    rho_z=data[6]
    rho_r=data[7]

    input_pv=input_pv.reshape(-1,input_pv.shape[-1])
    input_load=input_load.reshape(-1,input_load.shape[-1])
        
    mu_pv, sigma_pv = combined_model.model_pv_forward(input_pv, scaler_y_pv,30)

    mu_load, sigma_load = combined_model.model_load_forward(input_load, scaler_y_load)

    scaler_pv_mean = torch.tensor(scaler_y_pv.mean_,dtype=torch.float64).to(args.device)
    scaler_pv_scale = torch.tensor(scaler_y_pv.scale_,dtype=torch.float64).to(args.device)
    scaler_load_mean = torch.tensor(scaler_y_load.mean_,dtype=torch.float64).to(args.device)
    scaler_load_scale = torch.tensor(scaler_y_load.scale_,dtype=torch.float64).to(args.device)

    labels_load = (labels_load.reshape(-1,1)*scaler_load_scale+scaler_load_mean).reshape(-1,24)
    labels_pv = (labels_pv.reshape(-1,1)*scaler_pv_scale+scaler_pv_mean).reshape(-1,13)
    labels_pv = F.pad(labels_pv, (6, 5), "constant", 0)
    
    solution_ahead,solutoin_intra, obj = combined_model.forward(args, mu_pv, sigma_pv, mu_load, sigma_load, labels_pv, labels_load, alpha, alpha_z, rho_z, rho_r)
    
    loss=obj
    loss=loss.mean()
    loss.backward()
    optimizer.step()
    return combined_model,loss


def traditional_fine_tuning_parameter(model, inputs,labels,optimizer_sigma,num_epochs=1,patience=1):
    list_loss = []
    best_loss = float('inf')
    epochs_without_improvement = 0
    #num_epochs = 5
    criterion = likelihood()
    #patience=3
    for epoch in range(num_epochs):
        optimizer_sigma.zero_grad()
        
        # 前向传播
        mu, sigma = model(inputs)
        mu = mu.squeeze()
        sigma = sigma.squeeze()
        labels = labels.squeeze()
        
        # 计算损失
        loss = criterion(labels, mu, sigma)
        list_loss.append(loss.item())
        
        # 反向传播和优化
        loss.backward()
        optimizer_sigma.step()

        # 检查是否有改善
        if loss.item() < best_loss:
            best_loss = loss.item()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # 如果连续的 epoch 没有改善，则停止训练
        if epochs_without_improvement >= patience:
            break

    return model, loss

def traditional_fine_tuning_parameter(model, inputs,labels,optimizer_sigma,num_epochs=1,patience=1):
    list_loss = []
    best_loss = float('inf')
    epochs_without_improvement = 0
    #num_epochs = 5
    criterion = likelihood()
    #patience=3
    for epoch in range(num_epochs):
        optimizer_sigma.zero_grad()
        
        # 前向传播
        mu, sigma = model(inputs)
        mu = mu.squeeze()
        sigma = sigma.squeeze()
        labels = labels.squeeze()
        
        # 计算损失
        loss = criterion(labels, mu, sigma)
        list_loss.append(loss.item())
        
        # 反向传播和优化
        loss.backward()
        optimizer_sigma.step()

        # 检查是否有改善
        if loss.item() < best_loss:
            best_loss = loss.item()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # 如果连续的 epoch 没有改善，则停止训练
        if epochs_without_improvement >= patience:
            break

    return model, loss