# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/06_plots.ipynb.

# %% auto 0
__all__ = ['plot_lr_distributions', 'plot_ROC_curve', 'plot_tippett', 'plot_cllr', 'plot_ece', 'plot_cllr_per_qualitydrop',
           'plot_cllr_per_common_attributes']

# %% ../nbs/06_plots.ipynb 3
from typing import Dict, List, Optional
from sklearn.metrics import roc_curve
from lir import Xy_to_Xn, metrics
from lir.ece import plot 

import os
import numpy as np 
import pandas as pd

from matplotlib import pyplot as plt 
import seaborn as sns


# %% ../nbs/06_plots.ipynb 5
def plot_lr_distributions(results:Dict, experiment_directory, save_plots:bool = True, show: Optional[bool] = False):
    """
    Plots the 10log LRs generated for the two hypotheses by the fitted system.
    """
    predicted_log_lrs = np.log10(results["lrs_predicted"])
    plt.figure(figsize=(10, 10), dpi=100)
    points0, points1 = Xy_to_Xn(predicted_log_lrs, np.array(results['y_test']))
    plt.hist(points0, bins=20, alpha=.25, density=True)
    plt.hist(points1, bins=20, alpha=.25, density=True)
    plt.xlabel(r'$log_{10}$ LR')
    if save_plots:
        savefig = os.path.join(experiment_directory, "lr_distributions")
        plt.savefig(savefig, dpi=600)
        plt.close()
    if show:
        plt.show()

# %% ../nbs/06_plots.ipynb 6
def plot_ROC_curve(results:Dict, experiment_directory, save_plots:bool = True, show: Optional[bool] = False):

    norm_distances = np.asarray(results["test_norm_distances"])
    fpr, tpr, thresholds = roc_curve(results['y_test'], 1 - norm_distances)
    plt.figure(figsize=(10, 10), dpi=100)
    plt.plot(fpr, fpr, linestyle='--', label='No Skill')
    plt.plot(fpr, tpr, color='r', label=r'ROC curve')
    plt.xlabel('False positive rate (1 - specificity)')
    plt.ylabel('True positive rate (sensitivity)')
    plt.title('ROC curve')
    plt.legend()
    if save_plots:
        savefig = os.path.join(experiment_directory, "ROC_curve")
        plt.savefig(savefig, dpi=600)
        plt.close()
    if show:
        plt.show()

# %% ../nbs/06_plots.ipynb 7
def plot_tippett(results:Dict, experiment_directory, save_plots:bool = True, show: Optional[bool] = False):
        
    """
    Plots the 10log LRs in a Tippett plot.
    """

    predicted_log_lrs = np.log10(results["lrs_predicted"])

    xplot = np.linspace(
        start=np.min(predicted_log_lrs),
        stop=np.max(predicted_log_lrs),
        num=100
    )

    lr_0, lr_1 = Xy_to_Xn(predicted_log_lrs, np.array(results["y_test"]))

    perc0 = (sum(i > xplot for i in lr_0) / len(lr_0)) * 100
    perc1 = (sum(i > xplot for i in lr_1) / len(lr_1)) * 100

    plt.figure(figsize=(10, 10), dpi=600)
    plt.plot(xplot, perc1, color='b', label=r'LRs given $\mathregular{H_1}$')
    plt.plot(xplot, perc0, color='r', label=r'LRs given $\mathregular{H_2}$')
    plt.axvline(x=0, color='k', linestyle='--')
    plt.xlabel('Log likelihood ratio')
    plt.ylabel('Cumulative proportion')
    plt.title('Tippett plot')
    plt.legend()

    if save_plots:
        savefig = os.path.join(experiment_directory, "tippet_plot")
        plt.savefig(savefig, dpi=600)
        plt.close()
    if show:
        plt.show()

# %% ../nbs/06_plots.ipynb 8
def plot_cllr(results:Dict, experiment_directory, enfsi_years: List[int], cllr_expert_per_year:Dict, 
    cllr_auto_per_year:Dict, embeddingModel, save_plots:bool = True, show: Optional[bool] = False):
    
    """
    Plots cllr value for ENFSI tests. It computes both cllr of automated systems with the cllrs from experts.
    If there is no ENFSI data, this graph does not show.

    # todo: save table with cllr results.
    """

    cllr_auto_df = pd.DataFrame(columns=['Year', 'Expert', 'Cllr'])
    cllr_exp_df = pd.DataFrame(columns=['Year', 'Expert', 'Cllr'])
    years = enfsi_years

    for year in years:
        for cllr_exp in cllr_expert_per_year[year]:
            cllr_exp_df = cllr_exp_df.append({'Year': str(year), 'LR Estimator': "Participant", 'Cllr': cllr_exp},
                                                ignore_index=True)

        cllr_auto = cllr_auto_per_year[year]
        cllr_auto_df = cllr_auto_df.append(
            {'Year': str(year), 'LR Estimator': embeddingModel, 'Cllr': cllr_auto},
            ignore_index=True)


    cllr_df = cllr_exp_df.append(cllr_auto_df)
    #Cada LR estimator va en un color
    paleta = ['orange', 'blue']

    # añadimos el cllr de las imagen promediadas
    if len(results['lrs_predicted_2015']):
        x = metrics.cllr(np.asarray(results['lrs_predicted_2015']), np.asarray(results['y_test_2015']))    
        cllr_df = cllr_df.append({'Year': str(2015), 'LR Estimator': 'Quality weighted Images', 'Cllr': x},
            ignore_index=True)
        paleta.append('red')

    sns.set_style("whitegrid")
    sc_plot = sns.catplot(data=cllr_df, x="Year", y="Cllr", hue="LR Estimator",
                            palette=sns.color_palette(paleta))

    # sc_plot.set_title("Cllrs for Automated system and ENFSI participants")
    # sc_plot.set(xticks=[map(str, years)])

    if save_plots:
        savefig = os.path.join(experiment_directory, "cllr_experts")
        plt.savefig(savefig, dpi=600)
        plt.close()
    if show:
        plt.show()

# %% ../nbs/06_plots.ipynb 9
def plot_ece(results:Dict, experiment_directory, save_plots:bool = True) -> object:
        savefig = os.path.join(experiment_directory, "ECE_plot")
        plot(np.asarray(results["lrs_predicted"]), np.asarray(results["y_test"]), path=savefig, kw_figure={'figsize': (10, 10), 'dpi': 600})

# %% ../nbs/06_plots.ipynb 11
def plot_cllr_per_qualitydrop(cllrs_2015:Dict[float,float], cllr_expert_per_year,
experiment_directory,
save_plots:bool = True, 
show: Optional[bool] = False):
    
    df = pd.DataFrame.from_dict(cllrs_2015, orient='index', columns=['Cllr'])    
    df = df.reset_index(drop = False)
    df.rename(columns={'index': 'Quality Drop'}, inplace=True)
    
    df['legend'] = 'Automatic System'
    
    #df.reset_index(inplace = True)
    # cllr es un valor promedio del error cometido en las observaciones. 
    # como cada observador tiene el mismo número de observaciones, 
    # el promedio global es igual al promedio de los valores obtenidos por cada observador.
    cllr_experts = np.mean(cllr_expert_per_year[2015])

    #para dibujar en la gráfica
    x1 = np.min(df['Quality Drop'])
    x2 = np.max(df['Quality Drop'])    

    df = df.append({'Quality Drop': x1, 'Cllr': cllr_experts, 'legend': 'Participants'}, ignore_index = True)
    df = df.append({'Quality Drop': x2, 'Cllr': cllr_experts, 'legend': 'Participants'}, ignore_index = True)

    #hay que ordenar os datos para que no aparezcan leyendas múltiples
    df.sort_values(by= ['legend', 'Quality Drop'], inplace = True)
    
    sns.set_style("whitegrid")
    sc_plot = sns.lineplot(data=df, x='Quality Drop', y="Cllr", hue = 'legend', marker='s')
    sc_plot.set(xscale="log")

    sc_plot.set_title("Cllrs for Automated system and ENFSI year 2015 according to quality drop")
    # sc_plot.set(xticks=[map(str, years)])

    if save_plots:
        savefig = os.path.join(experiment_directory, "cllr_2015_quality_drop")
        plt.savefig(savefig, dpi=600)
        plt.close()
    if show:
        plt.show()


def plot_cllr_per_common_attributes(results:Dict, cllr_expert_per_year,
experiment_directory,
save_plots:bool = True, 
show: Optional[bool] = False):

    df0 = pd.DataFrame({'lr':results['lrs_predicted'],\
        'y': results['y_test'],\
        'common_attributes':results['common_attributes'],\
        'drop':results['quality_drops']})
        
    
    df0 = df0.loc[(df0['drop']==1)]
    df0 = df0.reset_index(drop = True)
    df = pd.Dataframe()

    for n in np.unique(df0['common_attributes']):
        dfn = df0.loc[(df0['common_attributes'] == n)]
        x = metrics.cllr(np.asarray(dfn['lr']), np.asarray(dfn['y']))    
        df = df.append({'legend': 'Automatic System', 'Common Attributes': n, 'Cllr': x}, ignore_index=True)    

    df = df.reset_index(drop = False)
    
    
    #df.reset_index(inplace = True)
    # cllr es un valor promedio del error cometido en las observaciones. 
    # como cada observador tiene el mismo número de observaciones, 
    # el promedio global es igual al promedio de los valores obtenidos por cada observador.
    cllr_experts = np.mean(cllr_expert_per_year[2015])

    #para dibujar en la gráfica
    x1 = np.min(df['Common Attributes'])
    x2 = np.max(df['Common Attributes'])    

    df = df.append({'legend': 'Participants', 'Common Attributes': x1, 'Cllr': cllr_experts}, ignore_index = True)
    df = df.append({'Common Attributes': x2, 'Cllr': cllr_experts, 'legend': 'Experts'}, ignore_index = True)

    #hay que ordenar os datos para que no aparezcan leyendas múltiples
    df.sort_values(by= ['legend', 'Common Attributes'], inplace = True)
    
    sns.set_style("whitegrid")
    sc_plot = sns.lineplot(data=df, x='Common Attributes', y="Cllr", hue = 'legend', marker='s')
    #sc_plot.set(xscale="log")

    sc_plot.set_title("Cllrs for Automated system and ENFSI year 2015 according to number of matching attributes")
    # sc_plot.set(xticks=[map(str, years)])

    if save_plots:
        savefig = os.path.join(experiment_directory, "cllr_2015_common_atts")
        plt.savefig(savefig, dpi=600)
        plt.close()
    if show:
        plt.show()
