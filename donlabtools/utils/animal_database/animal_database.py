import matplotlib
#
import matplotlib.pyplot as plt

import scipy
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

import os
os.chdir('/home/cat/code/manifolds/')

from calcium import calcium
from wheel import wheel
from visualize import visualize
from tqdm import trange

from scipy.io import loadmat
import umap

from sklearn.decomposition import PCA
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy import stats

#

#
np.set_printoptions(suppress=True)


class AnimalDatabase():

    def __init__(self):
        df = pd.read_excel('/media/cat/4TB/donato/CA3 Wheel Animals Database.xlsx')
        # print ("DF: ", df)

        self.df = df

    def load_sessions(self, animal_id):
        idx = np.where(self.df['Mouse_id'] == animal_id)[0].squeeze()
        P_start = int(self.df.iloc[idx]['Pday_start'])
        P_end = int(self.df.iloc[idx]['Pday_end'])
        #print("start: end: ", P_start, P_end)

        #
        session_ids = self.df.iloc[idx]['Session_ids'].split(',')
        session_ids = [x.strip(' ') for x in session_ids]

        self.sessions = session_ids