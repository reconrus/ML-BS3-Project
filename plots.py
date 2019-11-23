import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE

def plot_accuracies(epochs, train_accuracies, source_test_accuracies, target_test_accuracies):
    df = pd.DataFrame({
        'epoch': range(1, epochs + 1),
        'train': train_accuracies,
        'svhn test': source_test_accuracies,
        'mnist test': target_test_accuracies
    })    

    sns.lineplot(x='epoch', y='accuracy', hue='variable', 
                data=pd.melt(df, ['epoch'], value_name='accuracy'))
    plt.show()


def plot_latent_space(s_feature_map, title, t_feature_map=None): 
    pass

