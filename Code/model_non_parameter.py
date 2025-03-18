
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from utils import MAPE, RMSE
from scipy import stats
import torch.nn.functional as F
class ANN(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        super(ANN, self).__init__()
        self.layers = nn.ModuleList()
        
        # 添加输入层
        self.layers.append(nn.Linear(input_size, layer_sizes[0]))
        
        # 添加隐藏层
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        
        # 添加输出层
        self.layers.append(nn.Linear(layer_sizes[-1], output_size))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)  # 最后一层不使用激活函数
        return x
    

# class ANN_quantile(nn.Module):
#     def __init__(self, input_size, hidden_layers, quantiles,pv_flag=False,threshold=10,scaler_y=None):#,pv_flag=False,threshold=10,scaler_y=None
#         super(ANN_quantile, self).__init__()
#         layers = []
#         in_size = input_size
#         output_size = len(quantiles)
#         self.mean_index=quantiles.index(0.5)    
#         for hidden_size in hidden_layers:
#             layers.append(nn.Linear(in_size, hidden_size))
#             layers.append(nn.ReLU())
#             in_size = hidden_size

#         self.output_layers = nn.Linear(in_size, 1)
#         self.output_quantiles = nn.Linear(in_size, output_size-1)
#         self.hidden_layers = nn.Sequential(*layers)
#         self.pv_flag=pv_flag
#         if pv_flag:
#             self.scaler_mean=scaler_y.mean_
#             self.scaler_std=scaler_y.scale_
#             self.threshold=torch.tensor((threshold-self.scaler_mean)/self.scaler_std,dtype=torch.float32)
#             self.zero_bias=torch.tensor(-self.scaler_mean/self.scaler_std,dtype=torch.float32)
#             print(self.threshold)
#             print(torch.tensor(-self.scaler_mean/self.scaler_std))
        

#     def forward(self, x):
#         out = self.hidden_layers(x)
#         out_mean = self.output_layers(out)
#         out_quantiles = self.output_quantiles(out)

#         out = torch.cat([out_quantiles[:,:self.mean_index],out_mean, out_quantiles[:,self.mean_index:]], dim=1)
#         out = torch.cumsum(out, dim=1)
#         if self.pv_flag:
#             self.threshold = self.threshold.to(x.device)
#             self.zero_bias = self.zero_bias.to(x.device)
#             out[out<self.threshold]=self.zero_bias
#         return out


class ANN_quantile(nn.Module):
    def __init__(self, input_size, hidden_layers, quantiles, pv_flag=False, threshold=10, scaler_y=None):
        super(ANN_quantile, self).__init__()
        layers = []
        in_size = input_size
        output_size = len(quantiles)
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size

        self.output_layers = nn.Linear(in_size, 1)
        self.output_quantiles = nn.Linear(in_size, output_size)
        self.hidden_layers = nn.Sequential(*layers)
        self.pv_flag = pv_flag
        if pv_flag:
            self.scaler_mean = scaler_y.mean_
            self.scaler_std = scaler_y.scale_
            self.threshold = torch.tensor((threshold - self.scaler_mean) / self.scaler_std, dtype=torch.float32)
            self.zero_bias = torch.tensor(-self.scaler_mean / self.scaler_std, dtype=torch.float32)

    def forward(self, x):
        out = self.hidden_layers(x)
        out_quantiles = self.output_quantiles(out)
        
        if self.pv_flag:
            self.threshold = self.threshold.to(x.device)
            self.zero_bias = self.zero_bias.to(x.device)
            mask = out < self.threshold
            out = torch.where(mask, self.zero_bias, out)
        return out_quantiles


class ANN_gaussian(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        super(ANN_gaussian, self).__init__()
        layers = []
        in_size = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size

        self.hidden_layers = nn.Sequential(*layers)
        self.distribution_presigma = nn.Linear(in_size, output_size)
        self.distribution_mu = nn.Linear(in_size, output_size)
        self.distribution_sigma = nn.Softplus()
        # self.pv_flag = pv_flag
        # self.mu_threshold = mu_threshold
        # self.sigma_min = 1e-4

    def forward(self, x):
        out = self.hidden_layers(x)
        mu = self.distribution_mu(out)
        #mu = F.relu(mu)
        pre_sigma = self.distribution_presigma(out)
        sigma = self.distribution_sigma(pre_sigma)
        
        mu = torch.squeeze(mu)  
        sigma = torch.squeeze(sigma)
        # if self.pv_flag and torch.any(mu < self.mu_threshold):
        #         print('-=--')
        #         sigma = sigma*0.001
        #         #torch.clamp(sigma, max=self.sigma_min)
        return mu,sigma
    
class Determinisitic_layer():
    def __init__(self, args):
        super(Determinisitic_layer, self).__init__()
        T = args.T
        N_g = args.N_g
        N = args.N
        l_for = cp.Parameter(T, nonneg=True)
        p_for = cp.Parameter(T, nonneg=True)
        alpha = cp.Parameter(N_g * T, nonneg=True)
        alpha_z = cp.Parameter(2 * N_g * T, nonneg=True)

        x = cp.Variable((N_g + 2) * T, nonneg=True)
        z_max = cp.Variable(2 * N_g * T, nonneg=True)

        x_max = torch.tensor(args.x_max, dtype=torch.float64)
        r_pos = torch.tensor(args.r_pos, dtype=torch.float64)
        r_neg = torch.tensor(args.r_neg, dtype=torch.float64)

        I_N_g = torch.eye(N_g * T, dtype=torch.float64)
        I = torch.eye(T, dtype=torch.float64)
        Ramp_matrix = torch.zeros((T - 1, T), dtype=torch.float64)
        for i in range(T - 1):
            Ramp_matrix[i, i] = 1
            if i + 1 < T:
                Ramp_matrix[i, i + 1] = -1
        Ramp_matrix_combined = torch.cat([
            torch.cat([Ramp_matrix, torch.zeros((T - 1, T), dtype=torch.float64), torch.zeros((T - 1, T), dtype=torch.float64)], dim=1),
            torch.cat([torch.zeros((T - 1, T), dtype=torch.float64), Ramp_matrix, torch.zeros((T - 1, T), dtype=torch.float64)], dim=1),
            torch.cat([torch.zeros((T - 1, T), dtype=torch.float64), torch.zeros((T - 1, T), dtype=torch.float64), Ramp_matrix], dim=1)
        ], dim=0)
        I_combined = torch.cat([I, I, I], dim=1)

        A_eq_l = torch.cat([I_combined, torch.zeros((T, T), dtype=torch.float64), I], dim=1)
        A_eq_p = torch.cat([torch.zeros((T, N_g * T), dtype=torch.float64), I, I], dim=1)

        row1 = torch.cat([I_N_g, torch.zeros((N_g * T, T), dtype=torch.float64), torch.zeros((N_g * T, T), dtype=torch.float64)], dim=1)
        row2 = torch.cat([Ramp_matrix_combined, torch.zeros((N_g * (T - 1), T), dtype=torch.float64), torch.zeros((N_g * (T - 1), T), dtype=torch.float64)], dim=1)
        row3 = torch.cat([-Ramp_matrix_combined, torch.zeros((N_g * (T - 1), T), dtype=torch.float64), torch.zeros((N_g * (T - 1), T), dtype=torch.float64)], dim=1)

        x_max_list = torch.tensor([x_max for i in range(N_g * T)], dtype=torch.float64)
        r_pos_list = torch.tensor([r_pos for i in range(N_g * (T - 1))], dtype=torch.float64)
        r_neg_list = torch.tensor([r_neg for i in range(N_g * (T - 1))], dtype=torch.float64)
        A_uq_x = torch.cat([row1, row2, row3], dim=0)
        b_uq_x = torch.cat([x_max_list, r_pos_list, r_neg_list])

        w_cur_max_cons_A = torch.cat([torch.zeros((T, N_g * T), dtype=torch.float64), I, torch.zeros((T, T), dtype=torch.float64)], dim=1)
        w_cur_max_cons_b = p_for

        row1 = torch.cat([I_N_g, torch.zeros((N_g * T, N_g * T), dtype=torch.float64)], dim=1)
        row2 = torch.cat([torch.zeros((N_g * T, N_g * T), dtype=torch.float64), I_N_g], dim=1)
        z_pos_max_list = []
        z_neg_maxlist = []
        for n in range(N_g):
            z_pos_max_list += [args.z_pos_max[n] for i in range(T)]
            z_neg_maxlist += [args.z_neg_max[n] for i in range(T)]
        z_pos_max_list = torch.tensor(z_pos_max_list, dtype=torch.float64)
        z_neg_maxlist = torch.tensor(z_neg_maxlist, dtype=torch.float64)
        A_uq_z = torch.cat([row1, row2], dim=0)
        b_uq_z = torch.cat([z_pos_max_list, z_neg_maxlist])

        row_1=torch.cat([I_N_g, torch.zeros((N_g * T, T), dtype=torch.float64), torch.zeros((N_g * T, T), dtype=torch.float64)], dim=1)
        row_2=torch.cat([I_N_g, torch.zeros((N_g * T, N_g * T), dtype=torch.float64)], dim=1)
        row_3=torch.cat([torch.zeros((N_g * T, N_g * T), dtype=torch.float64), -I_N_g], dim=1)
        H_uq_x = torch.cat([row_1, -row_1], dim=0)
        H_uq_z = torch.cat([row_2, row_3], dim=0)
        h_uq = torch.cat([x_max_list, torch.tensor([0 for i in range(N_g * T)], dtype=torch.float64)])
       
        constraints = []
        constraints += [A_eq_l @ x == l_for]
        constraints += [A_eq_p @ x == p_for]
        constraints += [w_cur_max_cons_A @ x <= w_cur_max_cons_b]
        constraints += [A_uq_z @ z_max <= b_uq_z]
        constraints += [A_uq_x @ x <= b_uq_x]
        constraints += [H_uq_x @ x +H_uq_z@z_max <= h_uq]

        objective = cp.Minimize(alpha @ x[0:N_g * T] + alpha_z @ z_max)

        problem = cp.Problem(objective, constraints)
        self.cvxpyLayer = CvxpyLayer(problem, parameters=[p_for, l_for, alpha, alpha_z],
                                     variables=[x, z_max])


class Intra_schdule_layer():
    def __init__(self, args):
        super(Intra_schdule_layer, self).__init__()
        T = args.T
        N = args.N
        N_g = args.N_g

        l_for = cp.Parameter(T,nonneg = True)
        p_for = cp.Parameter(T,nonneg = True)
        target_p = cp.Parameter(T,nonneg = True)
        target_l = cp.Parameter(T,nonneg = True)
        z_max = cp.Parameter(2 * N_g * T,nonneg = True)
        rho_z = cp.Parameter(2 * N_g * T,nonneg = True)
        rho_r = cp.Parameter(2*T,nonneg = True)
        
        error_l = l_for - target_l
        error_p = p_for - target_p
        
        z = cp.Variable((2 * N_g * T), nonneg=True)
        r = cp.Variable(2*T,nonneg = True)
        
        I_N_g = torch.eye(N_g * T, dtype=torch.float64)
        I = torch.eye(T, dtype=torch.float64)
        Ramp_matrix = torch.zeros((T - 1, T), dtype=torch.float64)
        for i in range(T - 1):
            Ramp_matrix[i, i] = 1
            if i + 1 < T:
                Ramp_matrix[i, i + 1] = -1

        I_combined = torch.hstack([I, I, I])
        F_eq_r = torch.hstack([I, -I])
        F_eq_z = torch.hstack([I_combined, -I_combined])
        G_eq_p = I
        G_eq_l = I
        h_eq = torch.zeros(T, dtype=torch.float64)
        row1 = torch.hstack([I_N_g, torch.zeros((N_g * T, N_g * T), dtype=torch.float64)])
        row2 = torch.hstack([torch.zeros((N_g * T, N_g * T), dtype=torch.float64), I_N_g])
        F_uq = torch.vstack([row1, row2])

        constraints = []
        constraints += [
            F_uq @ z <= z_max,
            F_eq_z @ z - G_eq_p @ error_p + G_eq_l @ error_l + F_eq_r@ r == h_eq
        ]

        objective = cp.Minimize(rho_z @ z + rho_r @ r)
        
            # 定义并求解问题
        problem = cp.Problem(objective, constraints)
        
        self.cvxpyLayer = CvxpyLayer(problem, parameters=[z_max, p_for, l_for, target_p, target_l,rho_z,rho_r], 
                                        variables=[z, r])


class DRO_layer_dynamic_price():
    def __init__(self, args):
        super(DRO_layer_dynamic_price, self).__init__()
        T = args.T
        N = args.N
        N_g = args.N_g
        
        l_for = cp.Parameter(T,nonneg = True)
        p_for = cp.Parameter(T,nonneg = True)
        xi_p_min = cp.Parameter(T,nonneg = True)
        xi_p_max = cp.Parameter(T,nonneg = True)
        xi_l_min = cp.Parameter(T,nonneg = True)
        xi_l_max = cp.Parameter(T,nonneg = True)
        samples_l = cp.Parameter((N,T),nonneg = True)
        samples_p = cp.Parameter((N,T),nonneg = True)
        epsion_l = cp.Parameter(nonneg = True)
        epsion_p = cp.Parameter(nonneg = True)

        alpha = cp.Parameter(N_g*T,nonneg = True)
        alpha_z = cp.Parameter(2 * N_g * T,nonneg = True)
        rho_z = cp.Parameter(2 * N_g * T,nonneg = True)
        rho_r = cp.Parameter(2*T,nonneg = True)
        

        phi = cp.Variable(N, nonneg=True)
        gamma_p = cp.Variable(nonneg=True)
        gamma_l = cp.Variable(nonneg=True)
        x = cp.Variable((N_g + 2) * T, nonneg=True)
        z_max = cp.Variable(2 * N_g * T, nonneg=True)
        z = cp.Variable((2 * N_g * T, N), nonneg=True)
        z_for_xi_max = cp.Variable(2 * N_g * T, nonneg=True)
        z_for_xi_min = cp.Variable(2 * N_g * T, nonneg=True)
        r = cp.Variable((2 * T, N), nonneg=True)
        r_max = cp.Variable(2*T, nonneg=True)
        r_min = cp.Variable(2*T, nonneg=True)

        
        I_N_g = torch.eye(N_g * T, dtype=torch.float64)
        I = torch.eye(T, dtype=torch.float64)
        Zero_matrix = torch.zeros((T, T), dtype=torch.float64)
        x_max = torch.tensor(args.x_max, dtype=torch.float64)
        r_pos = torch.tensor(args.r_pos, dtype=torch.float64)
        r_neg = torch.tensor(args.r_neg, dtype=torch.float64)
        Ramp_matrix = torch.zeros((T - 1, T), dtype=torch.float64)
        for i in range(T - 1):
            Ramp_matrix[i, i] = 1
            if i + 1 < T:
                Ramp_matrix[i, i + 1] = -1
        Ramp_matrix_combined = torch.cat([
            torch.cat([Ramp_matrix, torch.zeros((T - 1, T), dtype=torch.float64), torch.zeros((T - 1, T), dtype=torch.float64)], dim=1),
            torch.cat([torch.zeros((T - 1, T), dtype=torch.float64), Ramp_matrix, torch.zeros((T - 1, T), dtype=torch.float64)], dim=1),
            torch.cat([torch.zeros((T - 1, T), dtype=torch.float64), torch.zeros((T - 1, T), dtype=torch.float64), Ramp_matrix], dim=1)
        ], dim=0)
        I_combined = torch.cat([I, I, I], dim=1)

        A_eq_l = torch.cat([I_combined, torch.zeros((T, T), dtype=torch.float64), I], dim=1)
        A_eq_p = torch.cat([torch.zeros((T, N_g * T), dtype=torch.float64), I, I], dim=1)


        row1 = torch.cat([I_N_g, torch.zeros((N_g * T, T), dtype=torch.float64), torch.zeros((N_g * T, T), dtype=torch.float64)], dim=1)
        row2 = torch.cat([Ramp_matrix_combined, torch.zeros((N_g * (T - 1), T), dtype=torch.float64), torch.zeros((N_g * (T - 1), T), dtype=torch.float64)], dim=1)
        row3 = torch.cat([-Ramp_matrix_combined, torch.zeros((N_g * (T - 1), T), dtype=torch.float64), torch.zeros((N_g * (T - 1), T), dtype=torch.float64)], dim=1)

        x_max_list = torch.tensor([x_max for i in range(N_g * T)], dtype=torch.float64)
        r_pos_list = torch.tensor([r_pos for i in range(N_g * (T - 1))], dtype=torch.float64)
        r_neg_list = torch.tensor([r_neg for i in range(N_g * (T - 1))], dtype=torch.float64)
        A_uq_x = torch.cat([row1, row2, row3], dim=0)
        b_uq_x = torch.cat([x_max_list, r_pos_list, r_neg_list])
        w_cur_max_cons_A = torch.cat([torch.zeros((T, N_g * T), dtype=torch.float64), I, torch.zeros((T, T), dtype=torch.float64)], dim=1)
        w_cur_max_cons_b = p_for

        row1 = torch.cat([I_N_g, torch.zeros((N_g * T, N_g * T), dtype=torch.float64)], dim=1)
        row2 = torch.cat([torch.zeros((N_g * T, N_g * T), dtype=torch.float64), I_N_g], dim=1)
        z_pos_max_list = []
        z_neg_maxlist = []
        for n in range(N_g):
            z_pos_max_list += [args.z_pos_max[n] for i in range(T)]
            z_neg_maxlist += [args.z_neg_max[n] for i in range(T)]
        z_pos_max_list = torch.tensor(z_pos_max_list, dtype=torch.float64)
        z_neg_maxlist = torch.tensor(z_neg_maxlist, dtype=torch.float64)
        A_uq_z = torch.cat([row1, row2], dim=0)
        b_uq_z = torch.cat([z_pos_max_list, z_neg_maxlist])

        F_eq_r = torch.cat([I, -I], dim=1)
        F_eq_z = torch.cat([I_combined, -I_combined], dim=1)
        G_eq_p = I
        G_eq_l = I
        h_eq = torch.zeros(T, dtype=torch.float64)
        row1 = torch.cat([I_N_g, torch.zeros((N_g * T, N_g * T), dtype=torch.float64)], dim=1)
        row2 = torch.cat([torch.zeros((N_g * T, N_g * T), dtype=torch.float64), I_N_g], dim=1)
        F_uq = torch.cat([row1, row2], dim=0)
        
        row_1=torch.cat([I_N_g, torch.zeros((N_g * T, T), dtype=torch.float64), torch.zeros((N_g * T, T), dtype=torch.float64)], dim=1)
        row_2=torch.cat([I_N_g, torch.zeros((N_g * T, N_g * T), dtype=torch.float64)], dim=1)
        row_3=torch.cat([torch.zeros((N_g * T, N_g * T), dtype=torch.float64), -I_N_g], dim=1)
        H_uq_x = torch.cat([row_1, -row_1], dim=0)
        H_uq_z = torch.cat([row_2, row_3], dim=0)
        h_uq = torch.cat([x_max_list, torch.tensor([0 for i in range(N_g * T)], dtype=torch.float64)])
       

        constraints = []
        

        constraints += [
            F_uq @ z_for_xi_min <= z_max,
            F_uq @ z_for_xi_max <= z_max,
            F_eq_z @ z_for_xi_min - G_eq_p @ xi_p_min + G_eq_l @ xi_l_min + F_eq_r@ r_min == h_eq,
            F_eq_z @ z_for_xi_max - G_eq_p @ xi_p_max + G_eq_l @ xi_l_max + F_eq_r@ r_max == h_eq,
        ]

        for i in range(N):  
            s_l = samples_l[i]
            s_p = samples_p[i]
            constraints += [rho_z @ z[:, i] + rho_r @ r[:, i] <= phi[i]]
            constraints += [rho_z @ z_for_xi_max + rho_r @ r_max - gamma_p * torch.ones(T, dtype=torch.float64) @ (xi_p_max - s_p) - gamma_l * torch.ones(T, dtype=torch.float64) @ (xi_l_max - s_l) <= phi[i]]
            constraints += [rho_z @ z_for_xi_min + rho_r @ r_min + gamma_p * torch.ones(T, dtype=torch.float64) @ (xi_p_min - s_p) + gamma_l * torch.ones(T, dtype=torch.float64) @ (xi_l_min - s_l) <= phi[i]]
            constraints += [F_uq @ z[:, i] <= z_max]
            constraints += [F_eq_z @ z[:, i] - G_eq_p @ s_p + G_eq_l @ s_l + F_eq_r @ r[:,i] == h_eq]

        constraints += [gamma_l >= 0]
        constraints += [gamma_p >= 0]
        constraints += [A_eq_l @ x == l_for]
        constraints += [A_eq_p @ x == p_for]
        constraints += [H_uq_x @ x +H_uq_z@z_max <= h_uq]
        constraints += [w_cur_max_cons_A @ x <= w_cur_max_cons_b]
        constraints += [
            A_uq_x @ x <= b_uq_x,
            A_uq_z @ z_max <= b_uq_z,
        ]

        objective = cp.Minimize(alpha @ x[0:N_g * T] + alpha_z @ z_max + torch.ones(N, dtype=torch.float64) @ phi / N + gamma_l * epsion_l + gamma_p * epsion_p)
        
        # 定义并求解问题
        problem = cp.Problem(objective, constraints)
        self.cvxpyLayer = CvxpyLayer(problem, parameters=[p_for, l_for, xi_p_min, xi_p_max, xi_l_min, xi_l_max, samples_p, samples_l, epsion_p, epsion_l,alpha,alpha_z,rho_z,rho_r],  
                                        variables=[x, z_max, z, z_for_xi_max, z_for_xi_min, r, r_max,r_min, phi, gamma_l, gamma_p])
                    

class Combined_deterministic_non_parameter_dynamic_price(torch.nn.Module):
    def __init__(self, Determinisitc_layer,Intra_schdule_layer,model_pv, model_load,epsion_pv=1e-2, epsion_load=1e-2):
        super(Combined_deterministic_non_parameter_dynamic_price, self).__init__()
        self.model_pv = model_pv
        self.model_load = model_load
        self.Deterministic_layer = Determinisitc_layer.cvxpyLayer
        self.epsion_pv = epsion_pv
        self.epsion_load = epsion_load
        self.Intra_schdule_layer = Intra_schdule_layer.cvxpyLayer
        
    def model_pv_forward(self, x, scaler_y):
        mean_torch = torch.tensor(scaler_y.mean_, dtype=torch.float64).to(x.device)
        scale_torch = torch.tensor(scaler_y.scale_, dtype=torch.float64).to(x.device)

        predictions=self.model_pv(x)
        predictions=(predictions.reshape(-1,1)* scale_torch + mean_torch).reshape(-1,13,predictions.shape[-1])
        predictions = F.pad(predictions, (0, 0, 6, 5), "constant", 0)

        return predictions

    def model_load_forward(self, x, scaler_y):
        mean_torch = torch.tensor(scaler_y.mean_, dtype=torch.float64).to(x.device)
        scale_torch = torch.tensor(scaler_y.scale_, dtype=torch.float64).to(x.device)

        predictions=self.model_load(x)
        predictions=(predictions.reshape(-1,1)* scale_torch + mean_torch).reshape(-1,24,predictions.shape[-1])
        return predictions
    
    def switch_solution_to_dict_ahead(self,solution):
        solution_dict={}
        variables_name=['x', 'z_max']
        for name_index in range(len(variables_name)):
            solution_dict[variables_name[name_index]]=solution[name_index]
        return solution_dict

    def switch_solution_to_dict_intra(self,solution):
        solution_dict={}
        variables_name=['z', 'r']
        for name_index in range(len(variables_name)):
            solution_dict[variables_name[name_index]]=solution[name_index]
        return solution_dict

    def forward(self,args, p_quantile_list,l_quantile_list, targets_pv, targets_load, alpha,alpha_z, rho_z,rho_r, quantiles):
        N = args.N
        T = args.T
        if isinstance(p_quantile_list, dict):
            forecasts_pv = p_quantile_list[0.5].reshape(-1,24)
            forecasts_load = l_quantile_list[0.5].reshape(-1,24)
        else:
                forecasts_pv = targets_pv
                forecasts_load = targets_load

        forecasts_pv[forecasts_pv < 0] = 0

        alpha=torch.tensor(alpha,dtype=torch.float64).to(args.device)
        alpha_z=torch.tensor(alpha_z,dtype=torch.float64).to(args.device)
        rho_z=torch.tensor(rho_z,dtype=torch.float64).to(args.device)
        rho_r=torch.tensor(rho_r,dtype=torch.float64).to(args.device)
        
        while True:
            try:
                solution_ahead = self.Deterministic_layer(forecasts_pv, forecasts_load,alpha,alpha_z, solver_args={'solve_method':'ECOS'})
                solution_dict_ahead =self.switch_solution_to_dict_ahead(solution_ahead)
                break  # 如果没有报错，则跳出
            except:
                solution_ahead = self.Deterministic_layer(forecasts_pv, forecasts_load,alpha,alpha_z, solver_args={'solve_method':'SCS','max_iters': 5000})
                solution_dict_ahead =self.switch_solution_to_dict_ahead(solution_ahead)
                break
        while True:
            try:
                solution_intra = self.Intra_schdule_layer(solution_dict_ahead['z_max'],forecasts_pv,forecasts_load, targets_pv, targets_load,rho_z,rho_r, solver_args={'solve_method':'ECOS'})
                solution_dict_intra=self.switch_solution_to_dict_intra(solution_intra)
                break
            except:
                solution_intra = self.Intra_schdule_layer(solution_dict_ahead['z_max'],forecasts_pv,forecasts_load, targets_pv, targets_load,rho_z,rho_r, solver_args={'solve_method':'SCS','max_iters': 5000})
                solution_dict_intra=self.switch_solution_to_dict_intra(solution_intra)
                break

        obj = torch.sum(solution_dict_ahead['x'][:,0:args.N_g*args.T] * alpha, dim=1)+torch.sum(solution_dict_ahead['z_max'] * alpha_z, dim=1)\
            +torch.sum(solution_dict_intra['z'] * rho_z, dim=1)+torch.sum(solution_dict_intra['r'] * rho_r, dim=1)
        
        return solution_dict_ahead,solution_dict_intra, obj


class Combined_model_non_parameter_dynamic_price(torch.nn.Module):
    def __init__(self, DRO_layer_dynamic_price,Intra_schdule_layer,model_pv, model_load,epsion_pv=1e-2, epsion_load=1e-2):
        super(Combined_model_non_parameter_dynamic_price, self).__init__()
        self.model_pv = model_pv
        self.model_load = model_load
        self.DRO_layer_dynamic_price = DRO_layer_dynamic_price.cvxpyLayer
        self.epsion_pv = epsion_pv
        self.epsion_load = epsion_load
        self.Intra_schdule_layer = Intra_schdule_layer.cvxpyLayer
        
    def model_pv_forward(self, x, scaler_y):
        mean_torch = torch.tensor(scaler_y.mean_, dtype=torch.float64).to(x.device)
        scale_torch = torch.tensor(scaler_y.scale_, dtype=torch.float64).to(x.device)

        predictions=self.model_pv(x)
        predictions=(predictions.reshape(-1,1)* scale_torch + mean_torch).reshape(-1,13,predictions.shape[-1])
        predictions = F.pad(predictions, (0, 0, 6, 5), "constant", 0)

        return predictions

    def model_load_forward(self, x, scaler_y):
        mean_torch = torch.tensor(scaler_y.mean_, dtype=torch.float64).to(x.device)
        scale_torch = torch.tensor(scaler_y.scale_, dtype=torch.float64).to(x.device)

        predictions=self.model_load(x)
        predictions=(predictions.reshape(-1,1)* scale_torch + mean_torch).reshape(-1,24,predictions.shape[-1])
        return predictions
    
    def switch_solution_to_dict_ahead(self,solution):
        solution_dict={}
        variables_name=['x', 'z_max']
        for name_index in range(len(variables_name)):
            solution_dict[variables_name[name_index]]=solution[name_index]
        return solution_dict

    def switch_solution_to_dict_intra(self,solution):
        solution_dict={}
        variables_name=['z', 'r']
        for name_index in range(len(variables_name)):
            solution_dict[variables_name[name_index]]=solution[name_index]
        return solution_dict

    def forward(self,args, p_quantile_list,l_quantile_list, targets_pv, targets_load, alpha,alpha_z, rho_z,rho_r, quantiles):
        N = args.N
        T = args.T
        if isinstance(p_quantile_list, dict):
            forecasts_pv = p_quantile_list[0.5].reshape(-1,24)
            forecasts_load = l_quantile_list[0.5].reshape(-1,24)
        else:
                forecasts_pv = targets_pv
                forecasts_load = targets_load

        forecasts_pv[forecasts_pv < 0] = 0
    
        l_quantile_list[0]=l_quantile_list[quantiles[0]]-(l_quantile_list[quantiles[1]]-l_quantile_list[quantiles[0]])/(quantiles[1]-quantiles[0])*quantiles[0]
        l_quantile_list[1]=l_quantile_list[quantiles[-1]]+(l_quantile_list[quantiles[-1]]-l_quantile_list[quantiles[-2]])/(quantiles[-1]-quantiles[-2])*(1-quantiles[-1])
        p_quantile_list[0]=p_quantile_list[quantiles[0]]-(p_quantile_list[quantiles[1]]-p_quantile_list[quantiles[0]])/(quantiles[1]-quantiles[0])*quantiles[0]
        p_quantile_list[1]=p_quantile_list[quantiles[-1]]+(p_quantile_list[quantiles[-1]]-p_quantile_list[quantiles[-2]])/(quantiles[-1]-quantiles[-2])*(1-quantiles[-1])

        quantiles_extend=[0]+args.quantiles+[1]
        
        quantile_load_tensor=torch.stack([l_quantile_list[i].reshape(-1,24) for i in quantiles_extend])
        quantile_pv_tensor=torch.stack([p_quantile_list[i].reshape(-1,24) for i in quantiles_extend])
        quantiles_extend=torch.tensor(quantiles_extend)
        
        xi_load_min,xi_load_max,samples_load = self.samples_generation(args, quantile_load_tensor, targets_load, quantiles_extend)
        xi_pv_min,xi_pv_max,samples_pv= self.samples_generation(args, quantile_pv_tensor, targets_pv, quantiles_extend)
        
        for i in range(args.N):
            for j in range(args.T):
                samples_pv[:, i, j] = torch.where(forecasts_pv[:, j] <= samples_pv[:, i, j], forecasts_pv[:, j], samples_pv[:, i, j])

        for j in range(args.T):
            xi_pv_max[:,j] = torch.where(forecasts_pv[:,j] <= xi_pv_max[:,j], forecasts_pv[:,j], xi_pv_max[:,j])
            xi_pv_min[:,j] = torch.where(forecasts_pv[:,j] <= xi_pv_min[:,j], forecasts_pv[:,j], xi_pv_min[:,j])

        alpha=torch.tensor(alpha,dtype=torch.float64).to(args.device)
        alpha_z=torch.tensor(alpha_z,dtype=torch.float64).to(args.device)
        rho_z=torch.tensor(rho_z,dtype=torch.float64).to(args.device)
        rho_r=torch.tensor(rho_r,dtype=torch.float64).to(args.device)
        epsion_p=torch.tensor(args.epsion_p,dtype=torch.float64).to(args.device)
        epsion_l=torch.tensor(args.epsion_l,dtype=torch.float64).to(args.device)
        
        while True:
            try:
                solution_ahead = self.DRO_layer_dynamic_price(forecasts_pv, forecasts_load, 
                                                            xi_pv_min, xi_pv_max, xi_load_min, xi_load_max, 
                                                            samples_pv, samples_load, epsion_p, epsion_l, 
                                                            alpha, alpha_z, rho_z, rho_r, solver_args={'solve_method': 'ECOS'})
                solution_dict_ahead=self.switch_solution_to_dict_ahead(solution_ahead)
                break  # 如果没有报错，则跳出循环
            except Exception as e:
                    solution_ahead = self.DRO_layer_dynamic_price(forecasts_pv, forecasts_load, 
                                                            xi_pv_min, xi_pv_max, xi_load_min, xi_load_max, 
                                                            samples_pv, samples_load, epsion_p, epsion_l, 
                                                            alpha, alpha_z, rho_z, rho_r, solver_args={'solve_method': 'SCS', 'max_iters': 5000})
                    solution_dict_ahead=self.switch_solution_to_dict_ahead(solution_ahead)
                    break
    
        while True:
            try:
                solution_intra = self.Intra_schdule_layer(solution_dict_ahead['z_max'],forecasts_pv,forecasts_load, targets_pv, targets_load,rho_z,rho_r, solver_args={'solve_method':'ECOS'})
                solution_dict_intra=self.switch_solution_to_dict_intra(solution_intra)
                break  # 如果没有报错，则跳出循环
            except Exception as e:
                    solution_intra = self.Intra_schdule_layer(solution_dict_ahead['z_max'],forecasts_pv,forecasts_load, targets_pv, targets_load,rho_z,rho_r, solver_args={'solve_method': 'SCS', 'max_iters': 5000})
                    solution_dict_intra=self.switch_solution_to_dict_intra(solution_intra)
                    break  # 捕获异常后处理完毕，跳出循环
        
        obj = torch.sum(solution_dict_ahead['x'][:,0:args.N_g*args.T] * alpha, dim=1)+torch.sum(solution_dict_ahead['z_max'] * alpha_z, dim=1)\
            +torch.sum(solution_dict_intra['z'] * rho_z, dim=1)+torch.sum(solution_dict_intra['r'] * rho_r, dim=1)

        return solution_dict_ahead,solution_dict_intra, obj

    def samples_generation(self,args,quantile_tensor,targets,quantiles):
        samples=torch.rand((args.N,args.T))
        quantile_tensor=quantile_tensor.permute(1, 0, 2)
        indices = torch.searchsorted(torch.tensor(quantiles),samples)

        samples_result=torch.zeros((quantile_tensor.shape[0],args.N,args.T)).to(args.device)
        for i in range(quantile_tensor.shape[0]):
            quantile_tensor_i = quantile_tensor[i]#.to(args.device)
            y0 = quantile_tensor_i[indices - 1, torch.arange(args.T)]#.to(args.device)
            y1 = quantile_tensor_i[indices, torch.arange(args.T)]#.to(args.device)
            x0 = torch.tensor(quantiles)[indices - 1].to(args.device)
            x1 = torch.tensor(quantiles)[indices].to(args.device)
            samples=samples.to(args.device)

            samples=samples.to(args.device)
            samples_result[i] = y0 + (samples - x0) * (y1 - y0) / (x1 - x0)

        samples_result=samples_result-targets.unsqueeze(1)
        xi_min = quantile_tensor[:,1]-targets
        xi_max = quantile_tensor[:,-2]-targets
        return xi_min, xi_max, samples_result