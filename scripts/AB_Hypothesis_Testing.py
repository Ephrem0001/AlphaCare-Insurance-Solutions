import pandas as pd
import numpy as np
from scipy.stats import ttest_ind


def gender_analysis(data):
    male_claims = data[data['Gender'] == 'Male']['TotalClaims']
    female_claims = data[data['Gender'] == 'Female']['TotalClaims']
    
    t_stat, p_value = ttest_ind(male_claims, female_claims, equal_var=False)
    return t_stat, p_value


def Province_vs_TotalPremium_analysis(data):
    male_claims = data[data['Gender'] == 'Male']['TotalClaims']
    female_claims = data[data['Gender'] == 'Female']['TotalClaims']
    
    t_stat, p_value = ttest_ind(male_claims, female_claims, equal_var=False)
    return t_stat, p_value


def gender_vs_zip_codes(data):
    male_claims = data[data['Gender'] == 'Male']['PostalCode']
    female_claims = data[data['Gender'] == 'Female']['PostalCode']
    
    t_stat, p_value = ttest_ind(male_claims, female_claims, equal_var=False)
    return t_stat, p_value


def calculate_margin(data):
    """Calculate the profit margin based on TotalPremium and TotalClaims."""
    data['ProfitMargin'] = ((data['TotalPremium'] - data['TotalClaims']) / data['TotalPremium']) * 100
    return data['ProfitMargin']


def PostalCode_analysis(data):
    male_claims = data[data['PostalCode'] == '2000']['ProfitMargin']
    female_claims = data[data['PostalCode'] == '122']['ProfitMargin']

    t_stat, p_value = ttest_ind(male_claims, female_claims, equal_var=False)
    return t_stat, p_value


