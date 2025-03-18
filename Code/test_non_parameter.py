import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from utils import *
import torch.nn.functional as F

def Evaluation_Ideal_non_parameter(args, combined_model, data, scaler_y_pv,scaler_y_load,plot_flag=True,accuracy_flag=True):
    input_pv=data[0]
    input_load=data[1]
    targets_pv=data[2]
    targets_load=data[3]
    alpha=data[4]
    alpha_z=data[5]
    rho_z=data[6]
    rho_r=data[7]
    input_pv=input_pv.reshape(-1,input_pv.shape[-1])
    input_load=input_load.reshape(-1,input_load.shape[-1])
    
    scaler_pv_mean = torch.tensor(scaler_y_pv.mean_,dtype=torch.float64).to(args.device)
    scaler_pv_scale = torch.tensor(scaler_y_pv.scale_,dtype=torch.float64).to(args.device)
    
    scaler_load_mean = torch.tensor(scaler_y_load.mean_,dtype=torch.float64).to(args.device)
    scaler_load_scale = torch.tensor(scaler_y_load.scale_,dtype=torch.float64).to(args.device)

    targets_load = (targets_load.reshape(-1,1)*scaler_load_scale+scaler_load_mean).reshape(-1,24)
    targets_pv = (targets_pv.reshape(-1,1)*scaler_pv_scale+scaler_pv_mean).reshape(-1,13)
    targets_pv = F.pad(targets_pv, (6, 5), "constant", 0)
    set_seed(46)
                         
    targets_pv[targets_pv<0]=0
    
    solution_dict_ahead, solution_dict_intra, obj = combined_model.forward(args, targets_pv, targets_load, targets_pv, targets_load,alpha,alpha_z,rho_z,rho_r, args.quantiles)

    return solution_dict_ahead, solution_dict_intra, obj

def Evaluation_deterministic_non_parameter(args, combined_model, data, scaler_y_pv,scaler_y_load,plot_flag=True,accuracy_flag=True):
    input_pv=data[0]
    input_load=data[1]
    targets_pv=data[2]
    targets_load=data[3]
    alpha=data[4]
    alpha_z=data[5]
    rho_z=data[6]
    rho_r=data[7]
    input_pv=input_pv.reshape(-1,input_pv.shape[-1])
    input_load=input_load.reshape(-1,input_load.shape[-1])
        
    predictions_load = combined_model.model_load_forward(input_load,scaler_y_load)
    predictions_pv = combined_model.model_pv_forward(input_pv,scaler_y_pv)
    quantile_pv_list={}
    quantile_load_list={}

    for i in range(len(args.quantiles)):
        quantile_pv_list[args.quantiles[i]]=predictions_pv[:,:,i]#.reshape(1,-1)[0]
        quantile_load_list[args.quantiles[i]]=predictions_load[:,:,i]#.reshape(1,-1)[0]

    scaler_pv_mean = torch.tensor(scaler_y_pv.mean_,dtype=torch.float64).to(args.device)
    scaler_pv_scale = torch.tensor(scaler_y_pv.scale_,dtype=torch.float64).to(args.device)
    scaler_load_mean = torch.tensor(scaler_y_load.mean_,dtype=torch.float64).to(args.device)
    scaler_load_scale = torch.tensor(scaler_y_load.scale_,dtype=torch.float64).to(args.device)

    targets_load = (targets_load.reshape(-1,1)*scaler_load_scale+scaler_load_mean).reshape(-1,24)
    targets_pv = (targets_pv.reshape(-1,1)*scaler_pv_scale+scaler_pv_mean).reshape(-1,13)
    targets_pv = F.pad(targets_pv, (6, 5), "constant", 0)
    
    quantile_load_list_array={} 
    quantile_pv_list_array={}
    for i in quantile_pv_list.keys():
        quantile_pv_list_array[i]=quantile_pv_list[i].detach().cpu().numpy().reshape(-1,24).reshape(1,-1)[0]
        quantile_pv_list_array[i][quantile_pv_list_array[i]<0]=0
        quantile_load_list_array[i]=quantile_load_list[i].detach().cpu().numpy().reshape(-1,24).reshape(1,-1)[0]
    
    targets_load_array = targets_load.detach().cpu().numpy().reshape(-1,24).reshape(1,-1)[0]
    targets_pv_array = targets_pv.detach().cpu().numpy().reshape(-1,24).reshape(1,-1)[0]

    set_seed(46)
                         
    targets_pv_array[targets_pv_array<0]=0
    solution_dict_ahead, solution_dict_intra, obj = combined_model.forward(args, quantile_pv_list, quantile_load_list, targets_pv, targets_load,alpha,alpha_z,rho_z,rho_r, args.quantiles)
    return solution_dict_ahead, solution_dict_intra, obj,quantile_pv_list_array,quantile_load_list_array,targets_pv_array,targets_load_array

def Evaluation_original_model_non_parameter(args_train, model_org, combined_test_loader,scaler_y_pv,scaler_y_load,mode='deterministic'):
    batch_idx=0
    cost_org=[]
    solution_list_org_ahead=[]
    solution_list_org_intra=[]
    targets_pv_list=[]  
    targets_load_list=[]
    forecasts_load_org={}
    forecasts_pv_org={}
    for i in args_train.quantiles:
        forecasts_pv_org[i]=[]
        forecasts_load_org[i]=[]
    

    data_iter = iter(combined_test_loader)
    next(data_iter)
    while True:
        try:
            input_pv_test, input_load_test, labels_pv_test, labels_load_test,alpha_test,alpha_z_test,rho_z_test,rho_r_test = next(data_iter)
            batch_idx+=1
            input_pv_test = input_pv_test.to(args_train.device).float()
            input_load_test = input_load_test.to(args_train.device).float()
            labels_pv_test = labels_pv_test.to(args_train.device).float()
            labels_load_test = labels_load_test.to(args_train.device).float()
            set_seed(42)
            
            test_data=[input_pv_test, input_load_test, labels_pv_test, labels_load_test, alpha_test,alpha_z_test,rho_z_test,rho_r_test]
            if mode=='deterministic':
                solution_org_ahead,solution_org_intra, obj_org, forecasting_pv,forecast_load,targets_pv,targets_load = Evaluation_deterministic_non_parameter(args_train, model_org, test_data, scaler_y_pv,scaler_y_load,plot_flag=False)
            elif mode=='dro':   
                solution_org_ahead,solution_org_intra, obj_org, forecasting_pv,forecast_load,targets_pv,targets_load = Evaluation_non_parameter_dynamic_price(args_train, model_org, test_data, scaler_y_pv,scaler_y_load,plot_flag=False)
            elif mode=='ideal':
                solution_org_ahead,solution_org_intra, obj_org = Evaluation_Ideal_non_parameter(args_train, model_org, test_data, scaler_y_pv,scaler_y_load,plot_flag=False)
            cost_org+=list(obj_org.detach().cpu().numpy())
            if solution_list_org_ahead==[]:
                solution_org_ahead=solution_org_ahead
            else:
                for i in solution_org_ahead.keys():   
                    solution_org_ahead[i]=torch.cat([solution_list_org_ahead[i],solution_org_ahead[i]])

            if solution_list_org_intra==[]:
                solution_list_org_intra=solution_org_intra
            else:
                for i in solution_org_intra.keys():   
                    solution_list_org_intra[i]=torch.cat([solution_list_org_intra[i],solution_org_intra[i]])
                    
            if mode=='dro':
                for i in args_train.quantiles:
                    forecasts_pv_org[i]=list(forecasts_pv_org[i])+list(forecasting_pv[i])
                    forecasts_load_org[i]=list(forecasts_load_org[i])+list(forecast_load[i])
                targets_pv_list+=list(targets_pv)
                targets_load_list+=list(targets_load)
            else:   
                pass

        except StopIteration:
            break
    if mode=='dro':
        return solution_list_org_intra, solution_list_org_ahead, cost_org, forecasts_pv_org, forecasts_load_org,targets_pv_list,targets_load_list
    else:
        return solution_list_org_intra, solution_list_org_ahead, cost_org

def Evaluation_non_parameter_dynamic_price(args, combined_model, data, scaler_y_pv,scaler_y_load,plot_flag=True,accuracy_flag=True):
    input_pv=data[0]
    input_load=data[1]
    targets_pv=data[2]
    targets_load=data[3]
    alpha=data[4]
    alpha_z=data[5]
    rho_z=data[6]
    rho_r=data[7]
    input_pv=input_pv.reshape(-1,input_pv.shape[-1])
    input_load=input_load.reshape(-1,input_load.shape[-1])
        
    predictions_load = combined_model.model_load_forward(input_load,scaler_y_load)
    predictions_pv = combined_model.model_pv_forward(input_pv,scaler_y_pv)
    quantile_pv_list={}
    quantile_load_list={}
    
    for i in range(len(args.quantiles)):
        quantile_pv_list[args.quantiles[i]]=predictions_pv[:,:,i]#.reshape(1,-1)[0]
        quantile_load_list[args.quantiles[i]]=predictions_load[:,:,i]#.reshape(1,-1)[0]

    scaler_pv_mean = torch.tensor(scaler_y_pv.mean_,dtype=torch.float64).to(args.device)
    scaler_pv_scale = torch.tensor(scaler_y_pv.scale_,dtype=torch.float64).to(args.device)
    scaler_load_mean = torch.tensor(scaler_y_load.mean_,dtype=torch.float64).to(args.device)
    scaler_load_scale = torch.tensor(scaler_y_load.scale_,dtype=torch.float64).to(args.device)

    targets_load = (targets_load.reshape(-1,1)*scaler_load_scale+scaler_load_mean).reshape(-1,24)
    targets_pv = (targets_pv.reshape(-1,1)*scaler_pv_scale+scaler_pv_mean).reshape(-1,13)
    targets_pv = F.pad(targets_pv, (6, 5), "constant", 0)
    
    quantile_load_list_array={} 
    quantile_pv_list_array={}
    for i in quantile_pv_list.keys():
        quantile_pv_list_array[i]=quantile_pv_list[i].detach().cpu().numpy().reshape(-1,24).reshape(1,-1)[0]
        quantile_load_list_array[i]=quantile_load_list[i].detach().cpu().numpy().reshape(-1,24).reshape(1,-1)[0]
    
    targets_load_array = targets_load.detach().cpu().numpy().reshape(-1,24).reshape(1,-1)[0]
    targets_pv_array = targets_pv.detach().cpu().numpy().reshape(-1,24).reshape(1,-1)[0]


    if accuracy_flag:
        print('--------------PV--------------')
        calculate_accuracy(args,quantile_pv_list_array, targets_pv_array)
        print('--------------Load--------------')
        calculate_accuracy(args,quantile_load_list_array, targets_load_array,metric_list=args.metric_list)
    
    if plot_flag:
        plot_quantiles(quantile_pv_list_array, targets_pv_array)
        plot_quantiles(quantile_load_list_array, targets_load_array)


    set_seed(46)
    solution_dict_ahead,solution_dict_intra, obj = combined_model.forward(args, quantile_pv_list, quantile_load_list, targets_pv, targets_load,alpha,alpha_z,rho_z,rho_r, args.quantiles)
    return solution_dict_ahead, solution_dict_intra, obj,quantile_pv_list_array,quantile_load_list_array,targets_pv_array,targets_load_array
