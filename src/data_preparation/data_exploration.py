# Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


def distribuition_plot(df, column):
    colors = ['#ff9999', '#66b3ff']
    fig = plt.figure(figsize=(6, 4))
    sns.countplot(data = df, x = column, hue = column, palette = colors)
    plt.xlabel('Sentimento')
    plt.ylabel('Contagem')
    plt.title('Avaliação do público', fontsize = 14, fontweight = 'bold')
    plt.legend(labels = ['Negativo', 'Positivo'])
    plt.show()


def load_data(path):
    """function to load a dataset

    Args:
        path (str): path to the dataset 
    
    Returns:
        DataFrame: a dataframe of the dataset
    """
    try:    
        data = pd.read_csv(path)
        return data
    except FileNotFoundError:
        return None
    
