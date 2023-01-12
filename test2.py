import pandas as pd
import os
from matplotlib import pyplot as plt
from lr_video_face.plots import *

df0 = pd.read_pickle('datos2.pd')

if len(pd.unique(df0.Detector)) == 1 and len(pd.unique(df0.Calibrator)) == 1:

            rows = list(pd.unique(df0['Embedding Model']))
            cols = list(pd.unique(df0['Quality Model']))

            #generamos la gráfica con subplots
            fig,ax = plt.subplots(nrows = len(rows), ncols = len(cols), squeeze= False, figsize= (16,14))
            
            for index, df1 in df0.iterrows():

                row = rows.index(df1['Embedding Model'])
                col = cols.index(df1['Quality Model'])

                ax1 = ax[row][col]
                #ax1.ylabel = df1['Embedding Model']
                ax1.set_title ( f"Quality Model: {df1['Quality Model']}")

                subplot_new(ax1,df1.Results, df1.Cllr)
                ax1.set(ylabel= f"Embedding Model: {df1['Embedding Model']}\n Cllr")
            plt.suptitle('$C_{llr}$ Values')
            fig.tight_layout()
            #savefig = os.path.join(self.experiments.output_dir, f"cllr_summary_ESX{self.experiments.embedding_model_as_scorer}")
            savefig = os.path.join('.', 'multigrafica')
            plt.savefig(savefig, dpi=600)
            plt.close() 