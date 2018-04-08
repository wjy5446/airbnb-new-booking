def submit_kaggle(df_train, df_test, model, path, user_id = "id", target = "country_destination"):
    
    le = LabelEncoder()

    y_train = le.fit_transform(df_train[target])
    X_train = df_train.drop([target, user_id], axis = 1)
    
    X_test_id = df_test[user_id]
    X_test = df_test.drop([user_id, target], axis = 1)
    

    print("model fitting 시작")
    model = model.fit(X_train, y_train)
    predic_proba = model.predict_proba(X_test)
    print("model fitting 종료")

    df_submit = pd.DataFrame(columns=["id", "country"])
    ids = []
    cts = []
    for i in range(len(X_test_id)):
        idx = X_test_id.iloc[i]
        ids += [idx] * 5
        cts += le.inverse_transform(np.argsort(predic_proba[i])[::-1])[:5].tolist()
    df_submit["id"] = ids
    df_submit["country"] = cts
    df_submit.to_csv(path, index = False)
    print("csv file 생성")
    !kaggle competitions submit -c airbnb-recruiting-new-user-bookings -f {path} -m "Message"