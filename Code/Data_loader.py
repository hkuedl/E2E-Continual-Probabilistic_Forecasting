from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
class Dataset_wind(Dataset):
    def __init__(self,  root_path='../Data/GFC12/', flag='train', size=[96,0,24], train_length=16800,
                data_path='wf1.csv', target='target', scale=True, inverse=True):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.root_path = root_path
        self.data_path = data_path
        #self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.flag = flag
        self.train_length = train_length
        print(self.data_path)
        if isinstance(self.data_path, list):
            self.X_nor, self.y_nor= self.process_datasets(self.data_path)
        else:
            self.X_nor, self.y_nor = self.__read_data__(self.data_path)

    def __read_data__(self,data_path):
        self.scaler_x = StandardScaler() #MinMaxScaler()# 
        self.scaler_y = StandardScaler() #MinMaxScaler()# 
        file_name=data_path
        wind_data=pd.read_csv(self.root_path+data_path+'.csv')
        wind_data = wind_data.interpolate(method='cubic', limit_direction='both')

        del wind_data['hors']
        wind_data['target_date']=wind_data['date'].shift(-self.pred_len+1)
        wind_data['target_date'] = pd.to_datetime(wind_data['target_date'])
        wind_data['target_month']=wind_data['target_date'].dt.month
        wind_data['target_week']=wind_data['target_date'].dt.weekday
        wind_data['target_hour']=wind_data['target_date'].dt.hour
        wind_data['target_year']=wind_data['target_date'].dt.year
        for i in range(0,self.pred_len): 
            wind_data[self.target+'+'+str(i)]=wind_data[file_name].shift(-i)
        for i in range(1,self.pred_len): 
            wind_data['pred_u'+'+'+str(i)]=wind_data['u'].shift(-i)
        for i in range(1,self.pred_len): 
            wind_data['pred_v'+'+'+str(i)]=wind_data['v'].shift(-i)

        wind_data.index=range(len(wind_data))

        for i in range(self.seq_len):
            wind_data['u_'+str(i+1)]=wind_data['u'].shift(i+1)

        for i in range(self.seq_len):
            wind_data['v_'+str(i+1)]=wind_data['v'].shift(i+1)

        for i in range(self.seq_len):
            wind_data['wf_'+str(i+1)]=wind_data[file_name].shift(i+1)

        u_features = ['u_'+str(i+1) if i != 0 else 'u' for i in range(self.seq_len)]
        v_features = ['v_'+str(i+1) if i != 0 else 'v' for i in range(self.seq_len)]
        wf_features = ['wf_'+str(i+1) for i in range(self.seq_len)]
        u_features_pred = ['pred_u'+'+'+str(i) for i in range(1,self.pred_len)]
        v_features_pred = ['pred_v'+'+'+str(i) for i in range(1,self.pred_len)]
        features_cal=['target_year','target_month','target_week']

        self.X_features_name = u_features + v_features + wf_features + features_cal + u_features_pred + v_features_pred
        self.y_features_name = [self.target+'+'+str(i) for i in range(self.pred_len)]
        
        wind_data = wind_data.reindex(columns=self.X_features_name+self.y_features_name)
        wind_data.dropna(inplace=True)

        # 划分训练和测试数据
        train_data = wind_data[0:self.train_length]
        test_data = wind_data[self.train_length:]
        # 提取训练数据的特征和目标
        X_train_before_split = train_data[self.X_features_name]
        y_train_before_split = train_data[self.y_features_name]
        
        X_test = test_data[self.X_features_name]
        y_test = test_data[self.y_features_name]

        # 划分训练集和验证集

        if self.scale:
            self.scaler_x.fit(X_train_before_split)
            self.scaler_y.fit(y_train_before_split)
            X_train_norm=self.scaler_x.transform(X_train_before_split)
            y_train_norm=self.scaler_y.transform(y_train_before_split)
            X_test_norm=self.scaler_x.transform(X_test)
            y_test_norm=self.scaler_y.transform(y_test)
            
            X_train_norm=X_train_norm
            y_train_norm=y_train_norm
            X_test_norm=X_test_norm[::24]
            y_test_norm=y_test_norm[::24]
            X_train, X_val, y_train, y_val = train_test_split(X_train_norm, y_train_norm, test_size=0.2, random_state=42)
        else:
            X_train, X_val, y_train, y_val = train_test_split(X_train_before_split, y_train_before_split, test_size=0.2, random_state=42)
        
        if self.flag == 'train':
            self.X = X_train
            self.y = y_train

        elif self.flag == 'val':
            self.X = X_val
            self.y = y_val
        else:
            self.X  = X_test_norm
            self.y = y_test_norm
        
        return self.X, self.y

    def process_datasets(self, path_lst):
        X_train_all = None
        y_train_all = None

        for path in path_lst:
            sub_X, sub_y = self.__read_data__(path)
            if X_train_all is None:
                X_train_all = sub_X
                y_train_all = sub_y
            else:
                X_train_all = np.concatenate([X_train_all, sub_X])
                y_train_all = np.concatenate([y_train_all, sub_y])

        return X_train_all, y_train_all


    def __getitem__(self, index):
        seq_x=self.X_nor[index]
        seq_y=self.y_nor[index]
        return seq_x,seq_y

    def __len__(self):
        return len(self.X)

    def inverse_transform(self, y_nor):
        return self.scaler_y.inverse_transform(y_nor)
    
class Dataset_pv(Dataset):
    def __init__(self,  data_path='../Data/PV/PV_1h.csv', flag='train', size=[96,0,24], train_length=2327, target='pv', scale=True, inverse=True):#4654
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.data_path = data_path
        #self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.flag = flag
        self.train_length = train_length
        print(self.data_path)
        self.X_nor, self.y_nor= self.process_datasets()
        
    def process_datasets(self):
        self.scaler_x = StandardScaler() #MinMaxScaler()# 
        self.scaler_y = StandardScaler() #MinMaxScaler()# 
        self.scaler_x_cal = StandardScaler() #MinMaxScaler()#
        data=pd.read_csv(self.data_path)
        #data=data[0::4]
        data = data.interpolate(method='cubic', limit_direction='both')
        
        for j in range(int(self.seq_len/24)):
            data['pv_'+str(j+1)+'_day_before']=data['value'].shift((j+1)*24)

        self.X_features_name = ['month','hour','irr','temp'] + ['pv_'+str(j+1)+'_day_before' for j in range(int(self.seq_len/24))]
        self.y_features_name = ['value']
        
        exclude_hours = [0, 1, 2, 3, 4, 5, 19, 20, 21, 22, 23]
        data = data[~data['hour'].isin(exclude_hours)]
        data = data.reindex(columns=self.X_features_name+self.y_features_name)
        data.dropna(inplace=True)

        # 划分训练和测试数据
        train_data = data[0:self.train_length]
        test_data = data[self.train_length:]
        
        # 提取训练数据的特征和目标
        X_train_before_split = train_data[self.X_features_name]
        y_train_before_split = train_data[self.y_features_name]
        
        X_test = test_data[self.X_features_name]
        y_test = test_data[self.y_features_name]

        # 划分训练集和验证集
        if self.scale:
            self.scaler_x.fit(X_train_before_split)
            self.scaler_y.fit(y_train_before_split)

            X_train_norm=self.scaler_x.transform(X_train_before_split)
            y_train_norm=self.scaler_y.transform(y_train_before_split)
            X_test_norm=self.scaler_x.transform(X_test)
            y_test_norm=self.scaler_y.transform(y_test)

            X_train_norm=X_train_norm.reshape((-1,24-len(exclude_hours),11))
            y_train_norm=y_train_norm.reshape(-1,24-len(exclude_hours))
            X_test_norm=X_test_norm.reshape((-1,24-len(exclude_hours),11))
            y_test_norm=y_test_norm.reshape(-1,24-len(exclude_hours)) 
            X_train, X_val, y_train, y_val = train_test_split(X_train_norm, y_train_norm, test_size=0.2, random_state=42)
        else:
            X_train, X_val, y_train, y_val = train_test_split(X_train_before_split, y_train_before_split, test_size=0.2, random_state=42)
            
        if self.flag == 'train':
            self.X = X_train
            self.y = y_train

        elif self.flag == 'val':
            self.X = X_val
            self.y = y_val
        else:
            self.X  = X_test_norm
            self.y = y_test_norm
        print(self.X.shape)
        print(self.y.shape)
        return self.X, self.y

    def __getitem__(self, index):
        seq_x=self.X_nor[index]
        seq_y=self.y_nor[index]
        return seq_x,seq_y

    def __len__(self):
        return len(self.X)

    def inverse_transform(self, y_nor):
        return self.scaler_y.inverse_transform(y_nor)

class Dataset_pv_seq2seq(Dataset):
    def __init__(self,  data_path='../Data/PV/PV_1h.csv', flag='train', size=[96,0,24], train_length=4296, target='pv', scale=True, inverse=True):#8592
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.data_path = data_path
        #self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.flag = flag
        self.train_length = train_length
        print(self.data_path)
        self.X_nor, self.y_nor= self.process_datasets()
        
    def process_datasets(self):
        self.scaler_x = StandardScaler() #MinMaxScaler()# 
        self.scaler_y = StandardScaler() #MinMaxScaler()# 
        data=pd.read_csv(self.data_path)
        #data=data[0::4]
        data = data.interpolate(method='cubic', limit_direction='both')

        for i in range(0,self.pred_len): 
            data[self.target+'+'+str(i)]=data['value'].shift(-i)
        for i in range(1,self.pred_len): 
            data['temp'+'+'+str(i)]=data['temp'].shift(-i)
        for i in range(1,self.pred_len): 
            data['irr'+'+'+str(i)]=data['irr'].shift(-i)

        data.index=range(len(data))

        for i in range(self.seq_len):
            data['pv_his_'+str(i+1)]=data['value'].shift(i+1)

        history_features = ['pv_his_'+str(i+1) for i in range(self.seq_len)]
        temp_features = ['temp'+'+'+str(i) for i in range(1,self.pred_len)]
        irr_features = ['irr'+'+'+str(i) for i in range(1,self.pred_len)]
        features_cal=['month','hour']

        self.X_features_name = features_cal + history_features + temp_features + irr_features
        self.y_features_name = [self.target+'+'+str(i) for i in range(self.pred_len)]
        
        data = data.reindex(columns=self.X_features_name+self.y_features_name)
        data.dropna(inplace=True)
        # 划分训练和测试数据
        train_data = data[0:self.train_length]
        test_data = data[self.train_length:]
        # 提取训练数据的特征和目标
        X_train_before_split = train_data[self.X_features_name]
        y_train_before_split = train_data[self.y_features_name]
        X_test = test_data[self.X_features_name]
        y_test = test_data[self.y_features_name]


        if self.scale:
            self.scaler_x.fit(X_train_before_split)
            self.scaler_y.fit(y_train_before_split)
            X_train_norm=self.scaler_x.transform(X_train_before_split)
            y_train_norm=self.scaler_y.transform(y_train_before_split)
            X_test_norm=self.scaler_x.transform(X_test)
            y_test_norm=self.scaler_y.transform(y_test)

            X_train_norm=X_train_norm#[::24]
            y_train_norm=y_train_norm#[::24]
            X_test_norm=X_test_norm[::24]
            y_test_norm=y_test_norm[::24]

            X_train, X_val, y_train, y_val = train_test_split(X_train_norm, y_train_norm, test_size=0.2, random_state=42)
        else:
            X_train, X_val, y_train, y_val = train_test_split(X_train_before_split, y_train_before_split, test_size=0.2, random_state=42)
        
        if self.flag == 'train':
            self.X = X_train
            self.y = y_train

        elif self.flag == 'val':
            self.X = X_val
            self.y = y_val
        else:
            self.X  = X_test_norm
            self.y = y_test_norm

        return self.X, self.y

    def __getitem__(self, index):
        seq_x=self.X_nor[index]
        seq_y=self.y_nor[index]
        return seq_x,seq_y

    def __len__(self):
        return len(self.X)

    def inverse_transform(self, y_nor):
        return self.scaler_y.inverse_transform(y_nor)
    

class Dataset_load_seq2seq(Dataset):
    def __init__(self,  data_path='../Data/GEF_data/data.csv', flag='train', size=[96,0,24], train_length=8592, target='load', scale=True, inverse=True):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.data_path = data_path
        #self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.flag = flag
        self.train_length = train_length
        print(self.data_path)
        self.X_nor, self.y_nor= self.process_datasets()
        
    def process_datasets(self):
        self.scaler_x = StandardScaler() #MinMaxScaler()# 
        self.scaler_y = StandardScaler() #MinMaxScaler()# 
        data=pd.read_csv(self.data_path)
        data = data.interpolate(method='cubic', limit_direction='both')

        for i in range(0,self.pred_len): 
            data[self.target+'+'+str(i)]=data['value'].shift(-i)
        for i in range(1,self.pred_len): 
            data['temp'+'+'+str(i)]=data['temp'].shift(-i)

        data.index=range(len(data))

        for i in range(self.seq_len):
            data['load_his_'+str(i+1)]=data['value'].shift(i+1)

        history_features = ['load_his_'+str(i+1) for i in range(self.seq_len)]
        T_features = ['temp'+'+'+str(i) for i in range(1,self.pred_len)]
        features_cal=['year','month','weekday']

        self.X_features_name = features_cal + history_features + T_features
        self.y_features_name = [self.target+'+'+str(i) for i in range(self.pred_len)]
        
        data = data.reindex(columns=self.X_features_name+self.y_features_name)
        data.dropna(inplace=True)
        # 划分训练和测试数据
        train_data = data[0:self.train_length]
        test_data = data[self.train_length:]
        # 提取训练数据的特征和目标
        X_train_before_split = train_data[self.X_features_name]
        y_train_before_split = train_data[self.y_features_name]
        
        X_test = test_data[self.X_features_name]
        y_test = test_data[self.y_features_name]

        # 划分训练集和验证集

        if self.scale:
            self.scaler_x.fit(X_train_before_split)
            self.scaler_y.fit(y_train_before_split)
            X_train_norm=self.scaler_x.transform(X_train_before_split)
            y_train_norm=self.scaler_y.transform(y_train_before_split)
            X_test_norm=self.scaler_x.transform(X_test)
            y_test_norm=self.scaler_y.transform(y_test)
            
            X_train_norm=X_train_norm[::24]
            y_train_norm=y_train_norm[::24]
            X_test_norm=X_test_norm[::24]
            y_test_norm=y_test_norm[::24]
            X_train, X_val, y_train, y_val = train_test_split(X_train_norm, y_train_norm, test_size=0.2, random_state=42)
        else:
            X_train, X_val, y_train, y_val = train_test_split(X_train_before_split, y_train_before_split, test_size=0.2, random_state=42)
        
        if self.flag == 'train':
            self.X = X_train
            self.y = y_train

        elif self.flag == 'val':
            self.X = X_val
            self.y = y_val
        else:
            self.X  = X_test_norm
            self.y = y_test_norm
        
        return self.X, self.y

    def __getitem__(self, index):
        seq_x=self.X_nor[index]
        seq_y=self.y_nor[index]
        return seq_x,seq_y

    def __len__(self):
        return len(self.X)

    def inverse_transform(self, y_nor):
        return self.scaler_y.inverse_transform(y_nor)
    

class Dataset_load(Dataset):
    def __init__(self,  data_path='../Data/GEF_data/data.csv', flag='train', size=[96,0,24], train_length=4296, target='load', scale=True, inverse=True):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.data_path = data_path
        #self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.flag = flag
        self.train_length = train_length
        print(self.data_path)
        self.X_nor, self.y_nor= self.process_datasets()
        
    def process_datasets(self):
        self.scaler_x = StandardScaler() #MinMaxScaler()# 
        self.scaler_y = StandardScaler() #MinMaxScaler()# 
        self.scaler_x_cal = StandardScaler() #MinMaxScaler()#
        data=pd.read_csv(self.data_path)
        data = data.interpolate(method='cubic', limit_direction='both')

        for j in range(int(self.seq_len/24)):
            data['load_'+str(j+1)+'_day_before']=data['value'].shift((j+1)*24)
        
        self.X_features_name = ['month','weekday','hour','temp'] + ['load_'+str(j+1)+'_day_before' for j in range(int(self.seq_len/24))]
        self.y_features_name = ['value']

        data = data.reindex(columns=self.X_features_name+self.y_features_name)
        data.dropna(inplace=True)

        # 划分训练和测试数据
        train_data = data[0:self.train_length]
        test_data = data[self.train_length:]
        
        # 提取训练数据的特征和目标
        X_train_before_split = train_data[self.X_features_name]
        y_train_before_split = train_data[self.y_features_name]
        
        X_test = test_data[self.X_features_name]
        y_test = test_data[self.y_features_name]

        # 划分训练集和验证集
        if self.scale:
            self.scaler_x.fit(X_train_before_split)
            self.scaler_y.fit(y_train_before_split)

            X_train_norm=self.scaler_x.transform(X_train_before_split)
            y_train_norm=self.scaler_y.transform(y_train_before_split)
            X_test_norm=self.scaler_x.transform(X_test)
            y_test_norm=self.scaler_y.transform(y_test)

            X_train_norm=X_train_norm.reshape((-1,24,11))
            y_train_norm=y_train_norm.reshape(-1,24)
            X_test_norm=X_test_norm.reshape((-1,24,11))
            y_test_norm=y_test_norm.reshape(-1,24) 
            X_train, X_val, y_train, y_val = train_test_split(X_train_norm, y_train_norm, test_size=0.2, random_state=42)
        else:
            X_train, X_val, y_train, y_val = train_test_split(X_train_before_split, y_train_before_split, test_size=0.2, random_state=42)
            
        if self.flag == 'train':
            self.X = X_train
            self.y = y_train

        elif self.flag == 'val':
            self.X = X_val
            self.y = y_val
        else:
            self.X  = X_test_norm
            self.y = y_test_norm
        print(self.X.shape)
        print(self.y.shape)
        return self.X, self.y

    def __getitem__(self, index):
        seq_x=self.X_nor[index]
        seq_y=self.y_nor[index]
        return seq_x,seq_y

    def __len__(self):
        return len(self.X)

    def inverse_transform(self, y_nor):
        return self.scaler_y.inverse_transform(y_nor)