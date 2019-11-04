import pandas as pd
import logging
import os
import numpy as np
from sklearn.externals import joblib
pd.options.mode.chained_assignment = None 
# np.random.seed(0)

def main():


    logging.basicConfig(level=logging.DEBUG)

    logging.info("Loading data....\n")
    repo_path = os.path.dirname(os.path.abspath(__file__))
    df_dir = ("data")
    df_dir_path = os.path.join(repo_path, df_dir)
    
    csv_list = []

    
    for file in os.listdir(df_dir):
        if file.endswith(".csv"):
            logging.debug(os.path.join(df_dir_path, file))
            csv_list.append(os.path.join(df_dir_path, file))


    df = pd.DataFrame()
    logging.info("Creating Dateframe from CSV files ...\n")
    for csv_file in csv_list:
        
        if  csv_file[-5:]=="1.csv":
            temp_df = pd.read_csv(csv_file, sep='\t',low_memory=False)
            
        else:
            temp_df = pd.read_csv(csv_file, sep=",",low_memory=False)
        
        logging.info("\nCSV file:{}\nSize = {}\nShape ={}\nShape[0] x Shape[1] = {}".format(csv_file,temp_df.size, temp_df.shape, temp_df.shape[0]*temp_df.shape[1]))
        df = df.append(temp_df, ignore_index = True, sort=True)
    
    logging.info("\n\n-------Final dataframe-------:\nSize = {}\nShape ={}\nShape[0] x Shape[1] = {}".format(csv_file,df.size, df.shape, df.shape[0]*df.shape[1]))        

    # df.to_csv("pf.csv")
    
    logging.info("\n\nFactorize data ...")
    cols = ['age_cat', 'edcution_cat', 'sex','years_in_residence','reg_cd', 'prod_id']
    df[cols] = df[cols].apply(lambda x: pd.factorize(x)[0] + 1)
    logging.info("\nSubstitute prod_class any_class/prod-type-5-class ->  0/1 ....")
    mask = (df['prod_class'] == "prod-type-5-class")
    neg_mask = ~(df['prod_class'] == "prod-type-5-class")
    df['prod_class'][mask] = 1
    df['prod_class'][neg_mask] = 0
    df = df.drop(['user_id'], axis=1)
    df = df.drop(['prod_id'], axis=1)
    # df.to_csv("af.csv")

    X = df[["age_cat","car_ownership","credit_status_cd","edcution_cat","reg_cd","revenue_usd","sex","years_in_residence"]].copy()
    Y = df[["prod_class"]].copy()

    logging.info("\nDevide data to train/test with test size 0.2")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    
    logging.info("Size of train data X:{}, Y:{}".format(X_train.shape, y_train.shape))
    logging.info("Size of test data X:{}, Y:{}".format(X_test.shape, y_test.shape))
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(X_train)
    # save scaler
    scaler_file = "LR_scaler.sav"
    joblib.dump(scaler, scaler_file)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    y_train = y_train['prod_class'].values.tolist()
    y_test = y_test['prod_class'].values.tolist()

   
    TRAIN = False
    filename_LR = 'LR.sav'
    if TRAIN:
    
        print("Length of training data set {}".format(len(X_train)))
        print("Start of training LR....")
        clf = LogisticRegression(random_state=0, solver='liblinear', class_weight='balanced').fit(X_train, y_train)
        # save the model to disk

    
        joblib.dump(clf, filename_LR)
    else:
        logging.info("\n\nTraining is off, loading models...\n")
        clf = joblib.load(filename_LR)

    test_size = len(X_test)
    X_test = X_test[:test_size, :]
    y_test = y_test[:test_size]
    prediction = clf.predict(X_test)
    predict_proba = clf.predict_proba(X_test)
    score = clf.score(X_test, y_test)
    y_pred = clf.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    logging.info("LR Score:{}".format(score))
    logging.info("LR precision:{}".format(precision))
    logging.info("LR recall:{}".format(recall))


    filename = 'MLP.sav'    
    if TRAIN:    
        print("Start of training MLP....")
        
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(verbose=True)
        model.fit(X_train, y_train)
        filename = 'MLP.sav'
        joblib.dump(model, filename)
    else:
        model = joblib.load(filename)
    

    MLP_acc_score = accuracy_score(y_test, model.predict(X_test))
    y_pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    logging.info("MLP Score:{}".format(MLP_acc_score))
    logging.info("MLP precision:{}".format(precision))
    logging.info("MLP recall:{}".format(recall))


if __name__ == '__main__':
    main()
