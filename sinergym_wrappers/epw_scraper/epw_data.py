import pickle
import numpy as np
class EPW_Data:
    def __init__(self, epw_df=None, transformed_df=None, pca=None, OU_mean=None, OU_std=None, 
                 OU_min=None, OU_max=None, weather_min=None, weather_max=None, base_OU=None) -> None:
        self.epw_df = epw_df
        self.transformed_df = transformed_df
        self.pca = pca
        self.OU_mean = OU_mean
        self.OU_std = OU_std
        self.OU_min = OU_min
        self.OU_max = OU_max
        self.base_OU = base_OU
        self.weather_min = weather_min
        self.weather_max = weather_max
    
    def read_OU_param(self, df, name):
        """
        Helper function to read the 3 OU parameters corresponding to a certain variable name in
        epw_df, transformed_df, OU_mean, or OU_std. 
        """
        return np.array([df[f"{name}_{i}"] for i in range(3)])
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(vars(self), f)
    
    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            
            variables =  pickle.load(f)
        return EPW_Data(**variables)