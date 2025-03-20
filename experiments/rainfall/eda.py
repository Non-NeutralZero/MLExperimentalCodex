import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

folder_path = 'eda'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)


def analyze_numerical_feature(feature, data, tag):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    sns.histplot(data[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    
    plt.subplot(1, 3, 2)
    sns.boxplot(x='rainfall', y=feature, data=data)
    plt.title(f'{feature} by Rainfall')
    
    plt.subplot(1, 3, 3)
    sns.kdeplot(data.loc[data['rainfall'] == 0, feature], label='No Rainfall')
    sns.kdeplot(data.loc[data['rainfall'] == 1, feature], label='Rainfall')
    plt.title(f'Density by Rainfall Status')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, f'{feature}_{tag}_analysis.png'))
    plt.show()
    plt.close()
