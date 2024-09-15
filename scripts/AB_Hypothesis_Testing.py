import pandas as pd
import numpy as np
from scipy.stats import ttest_ind


def gender_analysis(data):
    """Perform t-test analysis based on gender."""
    male_claims = data[data['Gender'] == 'Male']['TotalClaims']
    female_claims = data[data['Gender'] == 'Female']['TotalClaims']
    
    t_stat, p_value = ttest_ind(male_claims, female_claims, equal_var=False)
    return t_stat, p_value


