
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

# ### Date read

# In[7]:

sessions = pd.read_csv("sessions.csv")
test_users = pd.read_csv("test_users.csv")
train_users_2 = pd.read_csv("train_users_2.csv")

# ---

# ### Date setting

# In[8]:


def pre_age_set_data(train_users_2, test_users):
    
    check = pd.concat([train_users_2, test_users], ignore_index=True)
    
    check["first_affiliate_tracked"] = check["first_affiliate_tracked"].replace(np.nan, "untracked")
    
    check["date_account_created"] = pd.to_datetime(check["date_account_created"], format = "%Y-%m-%d")
    check["timestamp_first_active"] = pd.to_datetime(check["timestamp_first_active"], format="%Y%m%d%H%M%S")

    s_lag = check["timestamp_first_active"] - check["date_account_created"]

    check["lag_days"] = s_lag.apply(lambda x : -1 * x.days)
    check["lag_seconds"] = s_lag.apply(lambda x : x.seconds)

    s_all_check = (check['age'] < 120) & (check['gender'] != '-unknown-')

    check['faithless_sign'] = s_all_check.apply(lambda x : 0 if x == True else 1)
    
    pre_age = check.drop("date_first_booking",axis = 1)
    
    pre_age['date_account_created_y'] = pre_age["date_account_created"].apply(lambda x : x.year)
    pre_age['date_account_created_m'] = pre_age["date_account_created"].apply(lambda x : x.month)
    pre_age['date_account_created_d'] = pre_age["date_account_created"].apply(lambda x : x.day)

    pre_age['timestamp_first_active_y'] = pre_age["timestamp_first_active"].apply(lambda x : x.year)
    pre_age['timestamp_first_active_m'] = pre_age["timestamp_first_active"].apply(lambda x : x.month)
    pre_age['timestamp_first_active_d'] = pre_age["timestamp_first_active"].apply(lambda x : x.day)

    pre_age = pre_age.drop("date_account_created" , axis=1)
    pre_age = pre_age.drop("timestamp_first_active" , axis=1)
    
    return check, pre_age


# ---

# # Gender

# ### Gender predict data set

# In[11]:


def pre_gen_predict_data(pre_age):
    
    pre_gen_sub = pre_age.filter(items = ['age', 'country_destination', 'id', 'gender'])
    pre_gen_dum = pre_age.filter(items = ['affiliate_channel', 'affiliate_provider',
                                       'first_affiliate_tracked', 'first_browser', 'first_device_type',
                                         'language', 'signup_app', 'signup_flow',
                                       'signup_method', 'date_account_created_y', 'date_account_created_m',
                                       'date_account_created_d', 'timestamp_first_active_y',
                                       'timestamp_first_active_m', 'timestamp_first_active_d'])


    pre_gen_dum = pd.get_dummies(pre_gen_dum)
    pre_gen_dum_con = pd.concat([pre_gen_dum, pre_gen_sub], axis=1)
    pre_gen_dum_con["gender"] = pre_gen_dum_con["gender"].replace(['-unknown-', 'OTHER'], np.nan)

    pre_gen_mission = pre_gen_dum_con[pre_gen_dum_con["gender"].isnull()].reset_index()
    pre_gen_train = pre_gen_dum_con[pre_gen_dum_con["gender"].notnull()].reset_index()

    pre_gen_mission_test = pre_gen_mission.drop("index", axis=1)
    pre_gen_train_test = pre_gen_train.drop("index", axis=1)

    pre_gen_mission_test_drop = pre_gen_mission_test.drop(['id', 'age', 'country_destination', "gender"], axis=1)
    pre_gen_train_test_drop = pre_gen_train_test.drop(['id', 'age', 'country_destination', "gender"], axis=1)
    
    return pre_gen_mission_test, pre_gen_train_test, pre_gen_mission, pre_gen_train,             pre_gen_mission_test_drop, pre_gen_train_test_drop


# ### Gender predict LightGBM

# In[12]:


def predict_gen_LightGBM(pre_gen_train_test_drop, pre_gen_train_test, pre_gen_mission_test_drop):

    X = pre_gen_train_test_drop
    y = pre_gen_train_test["gender"]
    
    model_gen_lgb = lgb.LGBMClassifier(nthread=3)
    model_gen_lgb.fit(X,y)

    print(classification_report(y, model_gen_lgb.predict(pre_gen_train_test_drop)))
    model_gen_lgb = model_gen_lgb.predict(pre_gen_mission_test_drop)
    model_gen_lgb = pd.DataFrame(model_gen_lgb)
    
    return model_gen_lgb


# ### Gender predict data make CSV

# ---

# # Age

# ### Age predict data set

# In[13]:


def pre_age_predict_data(pre_age):
    
    pre_age['age'] = pre_age['age'].fillna(-1)
    
    pre_age_sub = pre_age.filter(items = ['age', 'country_destination','id'])
    pre_age_dum = pre_age.filter(items = ['affiliate_channel', 'affiliate_provider',
                                       'first_affiliate_tracked', 'first_browser', 'first_device_type',
                                       'language', 'signup_app', 'signup_flow',
                                       'signup_method', 'date_account_created_y', 'date_account_created_m',
                                       'date_account_created_d', 'timestamp_first_active_y',
                                       'timestamp_first_active_m', 'timestamp_first_active_d'])
    
    pre_age_dum = pd.get_dummies(pre_age_dum)
    pre_age_dum_con = pd.concat([pre_age_dum, pre_age_sub], axis=1)
    pre_age_dum_con["age"] = pre_age_dum_con["age"].replace(-1, np.nan)
    
    pre_age_mission = pre_age_dum_con[pre_age_dum_con["age"].isnull()].reset_index()
    pre_age_train = pre_age_dum_con[pre_age_dum_con["age"].notnull()].reset_index()
    
    pre_age_mission_test = pre_age_mission.drop("index", axis=1)
    pre_age_train_test = pre_age_train.drop("index", axis=1)
    
    pre_age_mission_test_drop = pre_age_mission_test.drop(['id', 'age', 'country_destination'], axis=1)
    pre_age_train_test_drop = pre_age_train_test.drop(['id', 'age', 'country_destination'], axis=1)
    
    return pre_age_mission_test, pre_age_train_test, pre_age_mission, pre_age_train, pre_age_mission_test_drop, pre_age_train_test_drop


# In[14]:


def pre_age_predict_data_cat(pre_age_train):
    
    bins = [0, 15, 25, 35, 60, 9999]
    labels = ["미성년자", "청년", "중년", "장년", "노년"]
    cats = pd.cut(pre_age_train['age'], bins, labels=labels)
    cats = pd.DataFrame(cats)
    
    return cats


# ### Age predict LightGBM

# In[15]:


def predict_age_LightGBM(pre_age_train_test_drop, cats, pre_age_mission_test_drop):

    X = pre_age_train_test_drop
    y = cats
    
    model_age_lgb = lgb.LGBMClassifier(nthread=3)
    model_age_lgb.fit(X,y)

    print(classification_report(y, model_age_lgb.predict(pre_age_train_test_drop)))
    model_age_lgb = model_age_lgb.predict(pre_age_mission_test_drop)
    model_age_lgb = pd.DataFrame(model_age_lgb)
    
    return model_age_lgb


# ### Age predict data make CSV