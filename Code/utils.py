import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import random
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import copy
from Data_loader import Dataset_wind,Dataset_load_seq2seq,Dataset_load,Dataset_pv_seq2seq,Dataset_pv
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

plt.rcParams["font.family"] = "Times New Roman"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def PICP(y, y_lower, y_upper):
    return np.mean((y >= y_lower) & (y <= y_upper))

def MAPE(y_actual,y_predicted):
    if isinstance(y_actual, torch.Tensor):
        y_actual = y_actual.detach().cpu().numpy()
    if isinstance(y_predicted, torch.Tensor):
        y_predicted = y_predicted.detach().cpu().numpy()
    MAPE=np.mean(abs((y_actual-y_predicted)/y_actual))
    return MAPE

def R2(y_actual,y_predicted):
    if isinstance(y_actual, torch.Tensor):
        y_actual = y_actual.detach().cpu().numpy()
    if isinstance(y_predicted, torch.Tensor):
        y_predicted = y_predicted.detach().cpu().numpy()
    R2 = 1 - np.sum(np.square(y_actual-y_predicted)) / np.sum(np.square(y_actual-np.mean(y_actual)))
    return R2

def RMSE(y_actual,y_predicted):
    if isinstance(y_actual, torch.Tensor):
        y_actual = y_actual.detach().cpu().numpy()
    if isinstance(y_predicted, torch.Tensor):
        y_predicted = y_predicted.detach().cpu().numpy()
    RMSE = np.sqrt(np.mean(np.square(y_actual-y_predicted)))
    return RMSE

def MAE(y_actual,y_predicted):
    if isinstance(y_actual, torch.Tensor):
        y_actual = y_actual.detach().cpu().numpy()
    if isinstance(y_predicted, torch.Tensor):
        y_predicted = y_predicted.detach().cpu().numpy()
    return np.mean(np.abs(y_actual-y_predicted))

def get_load_data(args,flag='train'):
    if flag=='train':
        shuffle_flag=True
        drop_last=True
    elif flag == 'val':
        shuffle_flag=False
        drop_last=False
    else:
        shuffle_flag=False
        drop_last=False
    if args.mode == 'seq2seq': 
        data_set=Dataset_load_seq2seq(flag=flag,size=[args.seq_len,args.label_len,args.pred_len])
    else:
        data_set=Dataset_load(flag=flag,size=[args.seq_len,args.label_len,args.pred_len])
    
    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=shuffle_flag,
        drop_last=drop_last)
    
    return data_set,data_loader

def get_pv_data(args,flag='train'):
    if flag=='train':
        shuffle_flag=True
        drop_last=True
    elif flag == 'val':
        shuffle_flag=False
        drop_last=False
    else:
        shuffle_flag=False
        drop_last=False
    if args.mode == 'seq2seq': 
        data_set=Dataset_pv_seq2seq(flag=flag,size=[args.seq_len,args.label_len,args.pred_len])
    else:
        data_set=Dataset_pv(flag=flag,size=[args.seq_len,args.label_len,args.pred_len])

    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=shuffle_flag,
        drop_last=drop_last)
     
    return data_set,data_loader

def get_combined_data(args,pv_data,load_data,alpha,alpha_z,rho_z,rho_r,flag='train'):
    pv_data_X = torch.tensor(pv_data.X, dtype=torch.float64)
    load_data_X = torch.tensor(load_data.X, dtype=torch.float64)
    pv_data_y = torch.tensor(pv_data.y, dtype=torch.float64)
    load_data_y = torch.tensor(load_data.y, dtype=torch.float64)
    alpha = torch.tensor(alpha, dtype=torch.float64)
    alpha_z = torch.tensor(alpha_z, dtype=torch.float64)
    
    rho_z = torch.tensor(rho_z, dtype=torch.float64)
    rho_r = torch.tensor(rho_r, dtype=torch.float64)
    batch_size = args.batch_size
    # 创建 TensorDataset
    combined_train_data = TensorDataset(pv_data_X, load_data_X, pv_data_y, load_data_y,alpha,alpha_z,rho_z,rho_r)
    shuffle_flag=True
    if flag=='test':
        shuffle_flag=False
        print('Test data is not shuffled')
        batch_size = 7
    combined_train_loader=DataLoader(combined_train_data, batch_size=batch_size, shuffle=shuffle_flag)
    return combined_train_data,combined_train_loader


def get_price_data(args,rho,rho_z,flag='train'):
    rho = torch.tensor(rho, dtype=torch.float64)
    rho_z = torch.tensor(rho_z, dtype=torch.float64)
    batch_size = args.batch_size
    # 创建 TensorDataset
    price_data = TensorDataset(rho, rho_z)
    shuffle_flag=True
    if flag=='test':
        shuffle_flag=False
        print('Test data is not shuffled')
        batch_size = 7
    price_loader=DataLoader(price_data, batch_size=batch_size, shuffle=shuffle_flag)
    return price_data,price_loader

def get_data(args,flag='train'):
    if flag=='train':
        shuffle_flag=True
        drop_last=True
    elif flag == 'val':
        shuffle_flag=False
        drop_last=False
    else:
        shuffle_flag=False
        drop_last=False
    data_set=Dataset_wind(args.root_path, data_path=args.dataset_paths, flag=flag,size=[args.seq_len,args.label_len,args.pred_len])
    
    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=shuffle_flag,
        drop_last=drop_last)
    
    return data_set,data_loader


def plot_quantiles(quantile_list,targets):
    x = range(len(targets))
    color_list=['#c0c8de', '#7ca3c3', '#427ea3', '#2c5382']
    plt.figure(figsize=(16, 6))
    plt.fill_between(x, quantile_list[0.1], quantile_list[0.9],  color=color_list[0], alpha=0.7, label='80% CI')
    plt.fill_between(x, quantile_list[0.2], quantile_list[0.8], color=color_list[1], alpha=0.7, label='60% CI')
    plt.fill_between(x, quantile_list[0.3], quantile_list[0.7],  color=color_list[2], alpha=0.7, label='40% CI')
    plt.fill_between(x, quantile_list[0.4], quantile_list[0.6], color=color_list[3], alpha=0.6, label='20% CI')
    plt.plot(x,quantile_list[0.5],color='blue',label='Median')
    plt.plot(x,targets,color='lightcoral',label='Actual')
    plt.legend(ncol=2)
    plt.show()


def generate_distribution_to_forecasts(predictions_mean, predictions_sigma, quantiles):
    if isinstance(predictions_mean, torch.Tensor):
        predictions_mean = predictions_mean.cpu().numpy().reshape(-1)
        predictions_sigma = predictions_sigma.cpu().numpy().reshape(-1)
    else:
        predictions_mean = predictions_mean.reshape(-1)
        predictions_sigma = predictions_sigma.reshape(-1)

    predictions_std = np.sqrt(predictions_sigma)
    quantile_list = {}
    for i in quantiles:
        quantile_list[i] = norm.ppf(i, predictions_mean, predictions_std)
        quantile_list[i][quantile_list[i] < 0] = 0

    return quantile_list

def pinball_loss_calculation(quantiles_list_inversed, label, quantile=[]):
    loss=[]
    for i in range(len(quantile)):
        errors = label - quantiles_list_inversed[quantile[i]]
        loss.append(np.maximum(quantile[i] * errors, (quantile[i] - 1) * errors))
    return np.mean(loss)

def winkler_score_calculation(quantiles_list, targets, alpha=0.2):
    lower_quantile = quantiles_list[alpha / 2]
    upper_quantile = quantiles_list[1 - alpha / 2]
    winkler_score = 0.0
    for i in range(len(targets)):
        if targets[i] < lower_quantile[i]:
            winkler_score += (upper_quantile[i] - lower_quantile[i]) + (2 / alpha) * (lower_quantile[i] - targets[i])
        elif targets[i] > upper_quantile[i]:
            winkler_score += (upper_quantile[i] - lower_quantile[i]) + (2 / alpha) * (targets[i] - upper_quantile[i])
        else:
            winkler_score += (upper_quantile[i] - lower_quantile[i])
    
    return winkler_score / len(targets)

def calculate_accuracy(args_train, quantiles_list, targets,metric_list=['pinball_loss','winkler_score_0.1','MAE','RMSE'],return_flag=False):
    result=[]
    for metric in metric_list:
        if metric=='pinball_loss':
            pinball_loss_value=pinball_loss_calculation(quantiles_list, targets, args_train.quantiles)
            result.append(pinball_loss_value)
            print('Pinball Loss:', pinball_loss_value)
        if 'winkler_score' in metric:
            alpha=float(metric.split('_')[-1])
            winkler_score_value=winkler_score_calculation(quantiles_list, targets, alpha=alpha)
            result.append(winkler_score_value)
            print(f'Winkler Score_{alpha}:', winkler_score_value)
        if metric=='MAE':
            MAE_value=MAE(quantiles_list[0.5], targets)
            result.append(MAE_value)
            print('MAE:', MAE_value)
        if metric=='RMSE':
            RMSE_value=RMSE(quantiles_list[0.5], targets)
            result.append(RMSE_value)
            print('RMSE:', RMSE_value)
    if return_flag:
        return result
    


def obtain_price(args,train_pv_data,train_load_data,val_pv_data,val_load_data,test_pv_data,test_load_data):
    if args.flag_dynamic_price:
        if args.flag_dynamic_mode==1:
            alpha_list_train = [args.alpha for j in range(np.shape(train_pv_data.X)[0])]  
            alpha_z_list_train = [[i*args.price_ratio_large for i in args.alpha_pos]+args.alpha_neg for j in range(np.shape(train_pv_data.X)[0])]
            rho_z_list_train = [[i*args.price_ratio_large for i in args.rho_z_pos]+args.rho_z_neg for j in range(np.shape(train_pv_data.X)[0])]
            rho_r_list_train = [[i*args.price_ratio_large for i in args.rho_r_pos]+args.rho_r_neg for j in range(np.shape(train_pv_data.X)[0])]

            alpha_list_val = [args.alpha for j in range(np.shape(val_pv_data.X)[0])]
            alpha_z_list_val = [[i*args.price_ratio_large for i in args.alpha_pos]+args.alpha_neg for j in range(np.shape(val_pv_data.X)[0])]
            rho_z_list_val = [[i*args.price_ratio_large for i in args.rho_z_pos]+args.rho_z_neg for j in range(np.shape(val_pv_data.X)[0])]
            rho_r_list_val = [[i*args.price_ratio_large for i in args.rho_r_pos]+args.rho_r_neg for j in range(np.shape(val_pv_data.X)[0])]

            alpha_list_test = [args.alpha for j in range(np.shape(test_pv_data.X)[0])]
            alpha_z_list_test = [[i*args.price_ratio_small for i in args.alpha_pos]+args.alpha_neg for j in range(np.shape(test_pv_data.X)[0])]
            rho_z_list_test = [[i*args.price_ratio_small for i in args.rho_z_pos]+args.rho_z_neg for j in range(np.shape(test_pv_data.X)[0])]
            rho_r_list_test = [[i*args.price_ratio_small for i in args.rho_r_pos]+args.rho_r_neg for j in range(np.shape(test_pv_data.X)[0])]

        else:
            alpha_list_train = [args.alpha for j in range(np.shape(train_pv_data.X)[0])]  
            alpha_z_list_train = [[i*args.price_ratio_small for i in args.alpha_pos]+args.alpha_neg for j in range(np.shape(train_pv_data.X)[0])]
            rho_z_list_train = [[i*args.price_ratio_small for i in args.rho_z_pos]+args.rho_z_neg for j in range(np.shape(train_pv_data.X)[0])]
            rho_r_list_train = [[i*args.price_ratio_small for i in args.rho_r_pos]+args.rho_r_neg for j in range(np.shape(train_pv_data.X)[0])]

            alpha_list_val = [args.alpha for j in range(np.shape(val_pv_data.X)[0])]
            alpha_z_list_val = [[i*args.price_ratio_small for i in args.alpha_pos]+args.alpha_neg for j in range(np.shape(val_pv_data.X)[0])]
            rho_z_list_val = [[i*args.price_ratio_small for i in args.rho_z_pos]+args.rho_z_neg for j in range(np.shape(val_pv_data.X)[0])]
            rho_r_list_val = [[i*args.price_ratio_small for i in args.rho_r_pos]+args.rho_r_neg for j in range(np.shape(val_pv_data.X)[0])]

            alpha_list_test = [args.alpha for j in range(np.shape(test_pv_data.X)[0])]
            alpha_z_list_test = [[i*args.price_ratio_large for i in args.alpha_pos]+args.alpha_neg for j in range(np.shape(test_pv_data.X)[0])]
            rho_z_list_test = [[i*args.price_ratio_large for i in args.rho_z_pos]+args.rho_z_neg for j in range(np.shape(test_pv_data.X)[0])]
            rho_r_list_test = [[i*args.price_ratio_large for i in args.rho_r_pos]+args.rho_r_neg for j in range(np.shape(test_pv_data.X)[0])]
    
    else:
        if args.flag_dynamic_mode==1:
            ratio=args.price_ratio_small
        else:
            ratio=args.price_ratio_large
        
        alpha_list_train = [args.alpha for j in range(np.shape(train_pv_data.X)[0])]
        alpha_z_list_train = [[i*ratio for i in args.alpha_pos]+args.alpha_neg for j in range(np.shape(train_pv_data.X)[0])]
        rho_z_list_train = [[i*ratio for i in args.rho_z_pos]+args.rho_z_neg for j in range(np.shape(train_pv_data.X)[0])]
        rho_r_list_train = [[i*ratio for i in args.rho_r_pos]+args.rho_r_neg for j in range(np.shape(train_pv_data.X)[0])]

        alpha_list_val = [args.alpha for j in range(np.shape(val_pv_data.X)[0])]
        alpha_z_list_val = [[i*ratio for i in args.alpha_pos]+args.alpha_neg for j in range(np.shape(val_pv_data.X)[0])]
        rho_z_list_val = [[i*ratio for i in args.rho_z_pos]+args.rho_z_neg for j in range(np.shape(val_pv_data.X)[0])]
        rho_r_list_val = [[i*ratio for i in args.rho_r_pos]+args.rho_r_neg for j in range(np.shape(val_pv_data.X)[0])]

        alpha_list_test = [args.alpha for j in range(np.shape(test_pv_data.X)[0])]
        alpha_z_list_test = [[i*ratio for i in args.alpha_pos]+args.alpha_neg for j in range(np.shape(test_pv_data.X)[0])]
        rho_z_list_test = [[i*ratio for i in args.rho_z_pos]+args.rho_z_neg for j in range(np.shape(test_pv_data.X)[0])]
        rho_r_list_test = [[i*ratio for i in args.rho_r_pos]+args.rho_r_neg for j in range(np.shape(test_pv_data.X)[0])]

        
    combined_train_data,combined_train_loader=get_combined_data(args,train_pv_data,train_load_data,
                        alpha_list_train,alpha_z_list_train,rho_z_list_train,rho_r_list_train,flag='train')

    combined_val_data,combined_val_loader=get_combined_data(args,val_pv_data,val_load_data,
                        alpha_list_val,alpha_z_list_val,rho_z_list_val,rho_r_list_val,flag='val')
                                                            
    combined_test_data,combined_test_loader=get_combined_data(args,test_pv_data,test_load_data,
                        alpha_list_test,alpha_z_list_test,rho_z_list_test,rho_r_list_test,flag='test')

    combined_fine_tune_data,combined_fine_tune_loader=get_combined_data(args,test_pv_data,test_load_data,
                        alpha_list_test,alpha_z_list_test,rho_z_list_test,rho_r_list_test,flag='test')

    return combined_train_data,combined_train_loader,combined_val_data,combined_val_loader,combined_test_data,combined_test_loader,combined_fine_tune_data,combined_fine_tune_loader