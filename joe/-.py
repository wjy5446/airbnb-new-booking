def pre_setting():
    last_train_data_add = pd.read_csv("last_train_data_add.csv")
    last_test_data_add = pd.read_csv("last_test_data_add.csv")
    
    X = last_train_data_add
    y = last_test_data_add
    
    model_lgb = lgb.LGBMClassifier(nthread=3)
    model_qda = QuadraticDiscriminantAnalysis(store_covariance=True)
    model_lda = LinearDiscriminantAnalysis(n_components=3, solver="svd", store_covariance=True)
    model_xg = XGBClassifier(nthread=3)
    
    return X, y, model_lgb, model_qda, model_lda, model_xg

X, y, model_lgb, model_qda, model_lda, model_xg = pre_setting()