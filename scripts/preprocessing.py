import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def oneHotEncoder(dataframe,columns_onehot):
        df_OHE = dataframe.copy()
        df_OHE= pd.get_dummies(data=df_OHE, prefix='OHE', prefix_sep='_',
                       columns=columns_onehot, drop_first=True, dtype='int8')
        return df_OHE


def labelEncoder(dataframe, columns_label):
    # if method == 'labelEncoder':      
        df_lbl = dataframe.copy()
        for col in columns_label:
            label = LabelEncoder()
            label.fit(list(dataframe[col].values))
            df_lbl[col] = label.transform(df_lbl[col].values)
        return df_lbl

def Scaler(data, columns_scaler):          
        Standard = StandardScaler()
        df_standard = data.copy()
        df_standard[columns_scaler] = Standard.fit_transform(df_standard[columns_scaler])        
        return df_standard


