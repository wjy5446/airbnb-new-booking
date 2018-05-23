# coding: utf-8

# ### Import

# In[1]:

from bs4 import BeautifulSoup 
import requests 
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

# In[12]:


sessions = pd.read_csv("sessions.csv")
test_users = pd.read_csv("test_users.csv")
train_users_2 = pd.read_csv("train_users_2.csv")
merged_sessions = pd.read_csv("merged_sessions.csv")


# ---

# ### Date setting - Base1

# In[13]:


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

# ### Date setting - Base2

# In[14]:


def pre_age_predict_data(pre_age):
    
    pre_age['age'] = pre_age['age'].fillna(-1)
    
    pre_age_sub = pre_age.filter(items = ['age', 'country_destination','id'])
    pre_age_dum = pre_age.filter(items = ['affiliate_channel', 'affiliate_provider',
                                       'first_affiliate_tracked', 'first_browser', 'first_device_type',
                                       'language', 'signup_app', 'signup_flow',
                                       'signup_method', 'date_account_created_y', 'date_account_created_m',
                                       'date_account_created_d', 'timestamp_first_active_y',
                                       'timestamp_first_active_m', 'timestamp_first_active_d'])
    
    
    pre_age_dum[['date_account_created_y', 'date_account_created_m', 'date_account_created_d',              'timestamp_first_active_y','timestamp_first_active_m',              'timestamp_first_active_d']] = pre_age_dum[['date_account_created_y', 'date_account_created_m',                                                          'date_account_created_d', 'timestamp_first_active_y',                                                           'timestamp_first_active_m',                                                          'timestamp_first_active_d']].astype(str)
    
    
    pre_age_dum = pd.get_dummies(pre_age_dum)
    pre_age_dum_con = pd.concat([pre_age_dum, pre_age_sub], axis=1)
    pre_age_dum_con["age"] = pre_age_dum_con["age"].replace(-1, np.nan)
    
    pre_age_mission = pre_age_dum_con[pre_age_dum_con["age"].isnull()].reset_index()
    pre_age_train = pre_age_dum_con[pre_age_dum_con["age"].notnull()].reset_index()
    
    pre_age_mission_test = pre_age_mission.drop("index", axis=1)
    pre_age_train_test = pre_age_train.drop("index", axis=1)
    
    pre_age_mission_test_drop = pre_age_mission_test.drop(['id', 'age', 'country_destination'], axis=1)
    pre_age_train_test_drop = pre_age_train_test.drop(['id', 'age', 'country_destination'], axis=1)
    
    return pre_age_mission_test, pre_age_train_test, pre_age_mission, pre_age_train,             pre_age_mission_test_drop, pre_age_train_test_drop


# In[15]:


def pre_age_predict_data_cat(pre_age_train):
    
    bins = [0, 15, 25, 35, 60, 9999]
    labels = ["미성년자", "청년", "중년", "장년", "노년"]
    cats = pd.cut(pre_age_train['age'], bins, labels=labels)
    cats = pd.DataFrame(cats)
    
    return cats


# ---

# ### Predict gender data setting - Only gender

# In[16]:




def add_gender(pre_age):
    
    pred_gen_data = pd.read_csv("model_gen_lgb.csv")
    
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

    pre_gen_mission_test_la = pd.concat([pre_gen_mission_test, pred_gen_data], axis=1)
    pre_gen_mission_test_la = pre_gen_mission_test_la.drop("gender", axis=1)
    pre_gen_mission_test_la = pre_gen_mission_test_la.rename(columns={"0": 'gender'})

    last_gen_add = pd.concat([pre_gen_mission_test_la, pre_gen_train_test])

    last_gen_add = last_gen_add.filter(items = ["id",'gender'])
    
    return last_gen_add


# ---

# ### Holiday, Weekend, Day of week data setting - Only Holiday

# In[17]:


def holiday(train_users_2, test_users):

    def get_holidays(year):
        response = requests.get("https://www.timeanddate.com/calendar/custom.html?year="+str(year)+"                                &country=1&cols=3&df=1&hol=25")
        dom = BeautifulSoup(response.content, "html.parser")

        trs = dom.select("table.cht.lpad tr")

        df = pd.DataFrame(columns=["date", "holiday"])
        for tr in trs:
            datestr = tr.select_one("td:nth-of-type(1)").text
            date = datetime.strptime("{} {}".format(year, datestr), '%Y %b %d')
            holiday = tr.select_one("td:nth-of-type(2)").text
            df.loc[len(df)] = {"date" : date, "holiday": 1}
        return df

    holiday_ls = []
    for year in range(2009, 2015):
        df = get_holidays(year)
        holiday_ls.append(df)
        holiday_df = pd.concat(holiday_ls)

    check = pd.concat([train_users_2, test_users], ignore_index=True)
    check["timestamp_first_active"] = check["timestamp_first_active"].apply(lambda x : str(x)[:8])

    pre_age_hol = check.filter(items=['id','timestamp_first_active'])

    pre_age_hol['week'] = pd.to_datetime(check["timestamp_first_active"], format="%Y-%m-%d")

    pre_age_hol["week"] = pre_age_hol['week'].dt.weekday
    pre_age_hol["weekend"] = pre_age_hol["week"].apply(lambda x : 1 if x>=5 else 0)
    pre_age_hol_dum = pd.get_dummies(pre_age_hol['week'])

    hdfd = pd.concat([pre_age_hol,pre_age_hol_dum],axis=1)
    hdfd = hdfd.drop("week",axis=1)

    hdfd = hdfd.rename(columns={0:"mon",1:"tue",2:"wed",3:"thur",4:"fri",5:"sat",6:"sun"})

    hdfd['timestamp_first_active'] = pd.to_datetime(hdfd["timestamp_first_active"])

    add_hol = pd.merge(hdfd, holiday_df, left_on='timestamp_first_active', right_on="date", how="left")

    add_hol = add_hol.drop(["timestamp_first_active",'date'],axis=1)
    add_hol = add_hol.fillna(0)

    return add_hol


# ---

# ### Predict age data setting - Merge (age+gender+holiday)

# In[8]:


# model_age_forest
# model_age_xg
# model_age_lgb

def predict_age_add(pre_age_mission_test, pre_age_train_test, last_gen_add, add_hol):
    
    pred_age_data = pd.read_csv("model_age_lgb.csv")
    
    pre_age_mission_test_la = pd.concat([pre_age_mission_test, pred_age_data], axis=1)
    pre_age_mission_test_la = pre_age_mission_test_la.drop("age", axis=1)
#     pre_age_mission_test_la["0"] = pre_age_mission_test_la["0"].replace({'age1':25,"age2":29,"age3":34,\
#                                                                          "age4":40,"age5":55})

    pre_age_mission_test_la["0"] = pre_age_mission_test_la["0"].replace({'미성년자':10,"청년":25,"중년":35,                                                                             "장년":45,"노년":60})
                                                                     
    pre_age_mission_test_la = pre_age_mission_test_la.rename(columns={"0": 'age'})
    
    pre_age_train_test_la = pre_age_train_test.drop("age", axis=1)
    pre_age_train_test_la['age'] = pre_age_train_test["age"]
    
    last_age_add = pd.concat([pre_age_mission_test_la, pre_age_train_test_la])
    
    train_set = train_users_2['id']
    train_set = pd.DataFrame(train_set)
    test_set = test_users['id']
    test_set = pd.DataFrame(test_set)
    
    last_gen_add_dum = pd.get_dummies(last_gen_add["gender"])
    last_gen_add_dum = pd.concat([last_gen_add['id'], last_gen_add_dum], axis=1)

    last_train_data = pd.merge(train_set, last_age_add, on="id", how="left")
    last_train_data = pd.merge(last_train_data, last_gen_add_dum, on="id", how="left")
    
    last_test_data = pd.merge(test_set, last_age_add, on="id", how="left")
    last_test_data = pd.merge(last_test_data, last_gen_add_dum, on="id", how="left")
    
    last_train_data = pd.merge(last_train_data, add_hol, on='id', how="left")
    last_test_data = pd.merge(last_test_data, add_hol, on='id', how="left")
    
    le = preprocessing.LabelEncoder()
    y_label = le.fit_transform(last_train_data["country_destination"]) 
    
    return last_train_data, last_test_data, y_label, le


# ---

# ### All data merge and make CSV - Last

# In[9]:


def last_data_setting(last_train_data, last_test_data):
    
    merged_sessions = pd.read_csv("merged_sessions.csv")
    merged_sessions_dum = merged_sessions.drop(['id','secs_elapsed','secs_sum','secs_mean'], axis=1)
    merged_sessions_dum = pd.get_dummies(merged_sessions_dum)
    ses_dum = pd.concat([merged_sessions_dum,merged_sessions[['id','secs_elapsed','secs_sum','secs_mean']]],axis=1)
    
    last_train_data_add = pd.merge(last_train_data, ses_dum, on="id", how="left")
    last_test_data_add = pd.merge(last_test_data, ses_dum, on="id", how="left")
    
    ## impute the missing value using median
    impute_list = last_test_data_add.columns.tolist()
    impute_list.remove("id")
    impute_list.remove("country_destination")

    imp = Imputer(missing_values='NaN', strategy='median', axis=0)

    last_train_data_add[impute_list] = imp.fit_transform(last_train_data_add[impute_list])
    last_test_data_add[impute_list] = imp.fit_transform(last_test_data_add[impute_list])

    last_train_data_add.to_csv("last_train_data_add.csv", index=False)
    last_test_data_add.to_csv("last_test_data_add.csv", index=False)
    
    return last_train_data_add, last_test_data_add
