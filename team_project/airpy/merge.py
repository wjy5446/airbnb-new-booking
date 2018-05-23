
# coding: utf-8

# ### Import

# In[2]:


import numpy as np
import pandas as pd
from sklearn.metrics import *
from IPython.core.display import Image 
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
import io
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
import lightgbm as lgb
from scipy.stats import mode
import re
from datetime import datetime
from lightgbm import plot_importance
import warnings
warnings.filterwarnings('ignore')


# ---

# ## Sessions

# ### Sessions (groupby mode + secs_elapsed setting)

# In[3]:


def make_merged_sessions():
    
    sessions = pd.read_csv("sessions.csv")
    
    sessions["action"] = sessions["action"].fillna("show")
    sessions["action_type"] = sessions["action_type"].fillna("view")
    sessions["action_detail"] = sessions["action_detail"].fillna("view_search_results")
    
    id_groupby = sessions.groupby(sessions["user_id"]).agg(mode)
    
    device_type = []
    action = []
    action_type = []
    action_detail = []
    secs_elapsed = []

    for i in range(len(id_groupby.index)):
        device_type.append(id_groupby['device_type'][i][0])
        action.append(id_groupby['action'][i][0])
        action_type.append(id_groupby['action_type'][i][0])
        action_detail.append(id_groupby['action_detail'][i][0])
        secs_elapsed.append(id_groupby['secs_elapsed'][i][0])
    
    id_groupby_df = pd.DataFrame({"id":id_groupby.index ,
                                  "device_type":device_type ,
                                  "action":action,
                                  "action_type":action_type,
                                  "action_detail":action_detail,
                                  "secs_elapsed":secs_elapsed
                                  })
    
    ses = pd.read_csv("sessions.csv")
    ses = ses.filter(items=('user_id', 'secs_elapsed'))
    
    ses_groupby_sum = ses.groupby("user_id").agg(np.sum)
    ses_groupby_mean = ses.groupby("user_id").agg(np.mean)
    
    merge_ses_groupby = pd.merge(ses_groupby_sum, ses_groupby_mean, left_index=True, right_index=True, how="left")
    merge_ses_groupby = merge_ses_groupby.rename(columns={'secs_elapsed_x': 'secs_sum', 'secs_elapsed_y': 'secs_mean'})
    
    merged_sessions = pd.merge(id_groupby_df, merge_ses_groupby, left_on="id", right_index=True, how="left")
    
    merged_sessions['secs_elapsed'] = merged_sessions['secs_elapsed'].astype(float)
    
    merged_sessions['secs_mean'] = merged_sessions['secs_mean'].fillna(0)
    
    merged_sessions.to_csv("merged_sessions.csv", index=False)
    
    return merged_sessions


# ---

# ### Sessions (remove word)

# In[4]:


def remove_word():
    
    merged_sessions = pd.read_csv("merged_sessions.csv")

    def remove(word):
        word = re.sub("''", "", word)
        word = re.sub("\W", "", word)
        return word

    merged_sessions["action"] = merged_sessions["action"].apply(remove)
    merged_sessions["action_detail"] = merged_sessions["action_detail"].apply(remove)
    merged_sessions["action_type"] = merged_sessions["action_type"].apply(remove)
    merged_sessions["device_type"] = merged_sessions["device_type"].apply(remove)


    merged_sessions["action_detail"] = merged_sessions["action_detail"].replace({"['-unknown-']":"unknown"})
    merged_sessions["action_type"] = merged_sessions["action_type"].replace({"['-unknown-']":"unknown"})
    merged_sessions["device_type"] = merged_sessions["device_type"].replace({"['-unknown-']":"unknown",                                             "['Android App Unknown Phone/Tablet']": "Androd_unkown_phone"})

    merged_sessions = merged_sessions.to_csv("merged_sessions.csv", index=False)
    
    return merged_sessions


# ---

# ### Sessions (Action counts) - Last Session

# In[5]:


def sessions_detail_add():

    merged_sessions = pd.read_csv("merged_sessions.csv")
    sessions = pd.read_csv("sessions.csv")

    tmp = sessions.groupby(["user_id", "action_type"])["device_type"].count().unstack().fillna(0)
    sessions_at = pd.DataFrame(tmp)
    sessions_at.rename(columns = lambda x : "type__" + x, inplace = True)

    tmp = sessions.groupby(["user_id", "action"])["device_type"].count().unstack().fillna(0)
    sessions_a = pd.DataFrame(tmp)
    sessions_a.rename(columns = lambda x : "action__" + x, inplace = True)

    tmp = sessions.groupby(["user_id", "action_detail"])["device_type"].count().unstack().fillna(0)
    sessions_ad = pd.DataFrame(tmp)
    sessions_ad.rename(columns = lambda x : "detail__" + x, inplace = True)

    df_session_info = sessions_at.merge(sessions_a, how = "outer", left_index = True, right_index = True)
    df_session_info = df_session_info.merge(sessions_ad, how = "left", left_index = True, right_index = True)

    df_session_info.drop(["type__-unknown-", "detail__-unknown-"], axis = 1, inplace = True)
    df_session_info = df_session_info.fillna(0)

    last_merged_sessions = pd.merge(merged_sessions, df_session_info, left_on='id', right_index=True, how='left')

    merged_sessions = last_merged_sessions.to_csv("merged_sessions.csv", index=False)

    return merged_sessions
