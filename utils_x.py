import imp
import time
import datetime
import logging
import datetime
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import norm
from sklearn.model_selection import train_test_split
sns.set(style="darkgrid")


def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'corr', np.corrcoef(preds, labels)[0,1]


def get_bst_list_cccc(data, use_f_list, use_ratio_list, t1, t2, t3, t4, params, tree_nums): 
    num_rounds =1000
    early = 30   
    data_tr = data[ (data["date_time"] >= t1) & (data["date_time"] < t2)].copy()
    data_te = data[ (data["date_time"] >= t2) & (data["date_time"] < t3)].copy()
    data_fin = data[ (data["date_time"] >= t3) & (data["date_time"] < t4)].copy()
    bst_list = []
    model_f_list = []
    for i in range(tree_nums):
        use_f = []
        for i in range(len(use_f_list)):
            _, f = train_test_split(use_f_list[i], test_size=use_ratio_list[i])
            use_f = use_f + f
        model_f_list.append(use_f)
        print(use_f)
        print(len(use_f))
        X_train = data_tr[use_f]
        y_train = data_tr['label']
        X_test = data_te[use_f]
        y_test = data_te['label']
        
        bst = train_op(X_train, y_train, params, num_rounds, X_test, y_test, early)
        bst_list.append(bst)
    return bst_list, model_f_list

def factor_yhat_corr(bst, data, use_f, t1, t2):
    temp = get_preds(temp, use_f, bst)
    corr = {}
    for i in use_f:
        temp_factor = temp[~temp[i].isna()]
        corr[i] = np.corrcoef(temp_factor[i], temp_factor["preds_demean"])[0,1]
    return corr

def pnl_plot(tb, te, data, use_f, bst):
    plt.rcParams['figure.figsize'] = (4, 4)
    data_sim = data[ (data["date_time"] >= tb) & (data["date_time"] < te)].copy()
    data_sim = get_preds(data_sim, use_f, bst)    
    data_sim = data_sim.reset_index(drop=True)
    data_sim["preds"] = data_sim["preds_demean"]

    print(np.corrcoef(data_sim["preds"], data_sim["return"] - data_sim["mr"])[0,1])

    data_sim["preds+"] = data_sim["preds"].apply(lambda x: x if x>=0 else 0)
    plt.rcParams['figure.figsize'] = (8, 8)
    data_sim["pnl_byweightF1"] = data_sim["preds+"] * (data_sim["return"] - data_sim["future500_return"])
    data_sim["pnl_byweightI1"] = data_sim["preds+"] * (data_sim["return"] - data_sim["index500_return"])
    pnl_byweightF1 = data_sim.groupby(['date_time'])['pnl_byweightF1'].sum()*0.008
    pnl_byweightI1 = data_sim.groupby(['date_time'])['pnl_byweightI1'].sum()*0.008
    pnl_byweightF1.cumsum().plot(label = "future")
    pnl_byweightI1.cumsum().plot(label = "index")
    plt.legend()
    plt.show()

    data_sim["preds-"] = data_sim["preds"].apply(lambda x: x if x<=0 else 0)
    plt.rcParams['figure.figsize'] = (8, 8)
    data_sim["pnl_byweightI2"] = data_sim["preds-"] * (data_sim["return"] - data_sim["index500_return"])
    data_sim["pnl_byweightF2"] =  data_sim["preds-"] * (data_sim["return"] - data_sim["future500_return"])
    pnl_byweightF2 = data_sim.groupby(['date_time'])['pnl_byweightF2'].sum()*0.008
    pnl_byweightI2 = data_sim.groupby(['date_time'])['pnl_byweightI2'].sum()*0.008
    pnl_byweightF2.cumsum().plot(label = "future")
    pnl_byweightI2.cumsum().plot(label = "index")
    plt.legend()
    plt.show()
    
    
    pnl_byweightF1.cumsum().plot(label = "future+")
    pnl_byweightI1.cumsum().plot(label = "index+")
    pnl_byweightF2.cumsum().plot(label = "future-")
    pnl_byweightI2.cumsum().plot(label = "index-")
    plt.legend()
    plt.show()

    plt.rcParams['figure.figsize'] = (8, 8)
    data_sim["pnl_byweightF"] = data_sim["preds"] * (data_sim["return"] - data_sim["future500_return"])
    pnl_byweightF = data_sim.groupby(['date_time'])['pnl_byweightF'].sum()*0.008
    pnl_byweightF.cumsum().plot(label = "future")
    plt.legend()
    plt.show()
    
    
    data_sim.groupby(["date_time"])["preds"].std().rolling(10).mean().plot()
    plt.title("preds_std")
    plt.show()
    data_sim.groupby(["date_time"])["preds"].quantile(q=0.80).rolling(10).mean().plot()
    plt.title("preds_quantile 0.8")
    plt.show()
    data_sim["abs_preds"] = abs(data_sim["preds"])
    data_sim.groupby(["date_time"])["abs_preds"].mean().rolling(10).mean().plot()
    plt.title("preds_abs")
    plt.show()
    return data_sim


def pnl_plot_self(tb, te, data, use_f, bst):
    
    plt.rcParams['figure.figsize'] = (4, 4)
    data_sim = data[ (data["date_time"] >= tb) & (data["date_time"] < te)].copy()
    data_sim = get_preds(data_sim, use_f, bst)    
    data_sim = data_sim.reset_index(drop=True)
    data_sim["std"] = data_sim.groupby(["date_time"])["preds"].transform(lambda x: x.std())
    data_sim["preds"] = data_sim["preds"]  - data_sim["pred_mean"]
    gb = data_sim.groupby(['date_time'])['preds']

    print(np.corrcoef(data_sim["preds"], data_sim["return"] - data_sim["mr"])[0,1])

    data_sim["preds+"] = data_sim["preds"].apply(lambda x: x if x>=0 else 0)
    plt.rcParams['figure.figsize'] = (8, 8)
    data_sim["pnl_byweightF1"] = data_sim["preds+"] * (data_sim["return"])
    pnl_byweightF1 = data_sim.groupby(['date_time'])['pnl_byweightF1'].sum()*0.008
    pnl_byweightF1.cumsum().plot(label = "future")
    plt.legend()
    plt.show()

    data_sim["preds-"] = data_sim["preds"].apply(lambda x: x if x<=0 else 0)
    plt.rcParams['figure.figsize'] = (8, 8)
    data_sim["pnl_byweightI2"] = data_sim["preds-"] * (data_sim["return"])
    pnl_byweightI2 = data_sim.groupby(['date_time'])['pnl_byweightI2'].sum()*0.008
    pnl_byweightI2.cumsum().plot(label = "index")
    plt.legend()
    plt.show()
    
    return data_sim


def pos_neg_corr(data, use_f, t1, t2, bst):
    
    temp = data[ (data["date_time"] >= t1) & (data["date_time"] < t2)].copy()
    temp = get_preds(temp, use_f, bst)
    temp1 = temp[temp["preds_demean"]<=0]
    temp2 = temp[temp["preds_demean"]>0]
    corr = np.corrcoef(temp["preds_demean"], temp["return_demean"])[0, 1]
    neg_corr = np.corrcoef(temp1["preds_demean"], temp1["return_demean"])[0, 1]
    pos_corr = np.corrcoef(temp2["preds_demean"], temp2["return_demean"])[0, 1]
    plot_conditional_expectation(np.array(temp["preds_demean"]), np.array(temp["return"] - temp["y_index"]), 50)
    print("corr", corr)
    print("neg_corr", neg_corr)
    print("pos_corr", pos_corr)
    return 


def buffer(df):
    gb_date_time = df.groupby(['date_time'])
    df["rank"] = gb_date_time['preds'].transform( lambda x:pd.qcut(x, 10, labels=False))
    gb_unique_symbol = df.groupby(['unique_symbol'])
    df["rank_last"] = gb_unique_symbol["rank"].transform( lambda x: x.shift(1))
    df["pred_last"] = gb_unique_symbol["preds"].transform( lambda x: x.shift(1))
    df["pred_last"] = df["pred_last"].fillna(0)
    df["buffer_pred"] = df.apply(lambda x: x["pred_last"] if abs(x["rank_last"] - x["rank"])<=6 else x["preds"], axis = 1)
    return df

def rev_ema(x):
    x = list(x)
    x.reverse()
    df = pd.Series(x)
    df_ewm = df.ewm(alpha=0.98).mean()
    df_ewm = list(df_ewm)
    df_ewm.reverse()
    df_ewm = pd.Series(df_ewm)
    return df_ewm


def train_op(X_train, y_train, params, num_rounds, X_v, y_v, early=0):
    d_train = xgb.DMatrix(X_train, y_train)
    d_test = xgb.DMatrix(X_v, y_v)
    eval_set = [(d_train, 'train'), (d_test, 'eval')]
    if early!=0:
        bst = xgb.train(params, d_train, num_boost_round=num_rounds, evals=eval_set, early_stopping_rounds=early)
    else:
        bst = xgb.train(params, d_train, num_boost_round=num_rounds, evals=eval_set)
    return bst



def InverseNormalCDF(quantile):
    if quantile <1 and quantile>0:
        tag = 1
    else:
        return np.nan
    a = [2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637]
    b = [-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833]
    c = [0.3374754822726147, 0.9761690190917186, 0.1607979714918209, 0.0276438810333863, 0.0038405729373609, 0.0003951896511919, \
         0.0000321767881768, 0.0000002888167364, 0.0000003960315187]
    if (quantile >= 0.5) and (quantile <= 0.92):
        y = quantile - 0.5
        r = y*y
        num = 0 
        denom = 0
        for i in range(3,-1,-1):
            num = num*r + a[i]
            denom = denom*r + b[i]
        return (y*num / (denom*r+1))
    
    elif (quantile > 0.92) and (quantile < 1):
        num = 0
        r = np.log(-np.log(1-quantile))
        for i in range(8,-1,-1):
            num = num*r+c[i];
        return num
    else:
        return -1.0 * InverseNormalCDF(1 - quantile)
        
        
def time_series_desribe(x, shiftn, col):
    res = []
    for i in range(1,shiftn):
        res.append(x[col + str(i)])
    res = np.array(res)
    f = res.max() - res.min()
    return res
    
    
def plot_conditional_expectation(x, y, quantile_n = 20):
    step = 1 / quantile_n
    bin_cut = np.percentile(x, list(np.append(np.arange(0, 1, step), 1)*100) )
    bin_cut = np.unique(bin_cut)
    tmp = pd.DataFrame({'x':x.flatten(), 'y':y.flatten()})
    tmp['bin_cut'] = pd.cut(tmp['x'], bin_cut, include_lowest=True)
    groupby_mean = tmp.groupby('bin_cut').mean().reset_index()

    plt.figure(figsize=(12,4))
    plt.plot(groupby_mean['x'], groupby_mean['y'], '-o')
    plt.axvline(x=0, color='r', ls='-.')
    plt.axhline(y=0, color='r', ls='-.')
    plt.show()
        
    return groupby_mean


def plot_conditional_std(x, y, quantile_n = 20):
    step = 1 / quantile_n
    bin_cut = np.percentile(x, list(np.append(np.arange(0, 1, step), 1)*100) )
    bin_cut = np.unique(bin_cut)
    tmp = pd.DataFrame({'x':x.flatten(), 'y':y.flatten()})
    tmp['bin_cut'] = pd.cut(tmp['x'], bin_cut, include_lowest=True)
    groupby_mean = tmp.groupby('bin_cut').std().reset_index()

    plt.figure(figsize=(12,4))
    plt.plot(groupby_mean['x'], groupby_mean['y'], '-o')
    plt.axvline(x=0, color='r', ls='-.')
    plt.axhline(y=0, color='r', ls='-.')
    plt.show()       
    return groupby_mean


 
def combine_plus(result1, result2):
    factor_group_l = []
    for i in range(len(result1)):
        factor_group_l.append(result1[i]+result2[i])
    return factor_group_l
    
def combine_mult(result1, result2):
    factor_group_l = []
    for i in result1:
        for j in result2:
            factor_group_l.append(i+j)
    return factor_group_l


def get_preds(data, use_f, bst):
    tree_nums = bst.best_ntree_limit 
    preds = bst.predict(xgb.DMatrix(data[use_f]), ntree_limit=tree_nums)
    data["preds"] = list(preds)
    limit_tag = list(data["limit_tag"])
    minmax_preds = [0]
    for i in range(len(preds)):
        if limit_tag[i]>= 1.099:
            minmax_preds.append(min(preds[i], minmax_preds[i-1]))
        elif limit_tag[i]<= -1.099:
            minmax_preds.append(max(preds[i], minmax_preds[i-1]))
        else:
            minmax_preds.append(preds[i])
    data["preds"] = minmax_preds[1:]
    gb = data.groupby(['date_time'])['preds']
    data["pred_mean"] = gb.transform(lambda x: x.mean())
    data["preds_demean"] = data["preds"] - data["pred_mean"]
    data["return_demean"] = data["return"] - data["y_index"]
    return data
    
    

def single_train(data, use_f, t1, t2, t3, t4, params, early, num_rounds, split_type):
        
#     生成log文件
    logger = logging.getLogger(str(time.ctime()))
    logger.setLevel(level = logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler("log/batch_train:" + str(time.ctime()) + ".txt")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)
    
    logger.info(params)
    logger.info("factor:" + "|".join(use_f))

#     选择数据集
   
    if split_type == "random":
        tt = data[ (data["date_time"] >= t1) & (data["date_time"] < t3)].copy()
        data_tr, data_te, s2, s1 = train_test_split(tt, tt['label'], test_size=0.01)
        data_tr = data_tr.copy()
        data_te = data_te.copy()
    elif split_type == "time":
        data_tr = data[ (data["date_time"] >= t1) & (data["date_time"] < t2)].copy()
        data_te = data[ (data["date_time"] >= t2) & (data["date_time"] < t3)].copy()
    else:
        print("error split_type")
    
    data_fin = data[ (data["date_time"] >= t3) & (data["date_time"] < t4)].copy()
    X_train = data_tr[use_f]
    y_train = data_tr['label']
    X_test = data_te[use_f]
    y_test = data_te['label']        
    

#     训练
    print("-------------------- train -----------------------")
    bst = train_op(X_train, y_train, params, num_rounds, X_test, y_test, early)  
    data_fin = get_preds(data_fin, use_f, bst)
    data_te = get_preds(data_te, use_f, bst)
    data_tr = get_preds(data_tr, use_f, bst)

    logger.info("")
    for y in range(2015,2020):
        temp_df = data_fin[data_fin["date_time"].dt.year == y ]
        logger.info("test_set" + str(y) + ":"+ str(np.corrcoef(temp_df["return_demean"], temp_df["preds_demean"])[0,1]))
    logger.info("test_all_corrcoef:" + str(np.corrcoef(data_fin["return_demean"] , data_fin["preds_demean"])[0,1]))
    logger.info("test_all_corrcoef_Nodemean:" + str(np.corrcoef(data_fin["return_demean"] , data_fin["preds"])[0,1])) 
    logger.info("")

    logger.info("")
    for y in range(2014,2016):
        temp_df = data_te[data_te["date_time"].dt.year == y ]
        logger.info("val_set"+str(y) + ":" + str(np.corrcoef(temp_df["return_demean"], temp_df["preds_demean"])[0,1])) 
    logger.info("val__all_corrcoef:" + str(np.corrcoef(data_te["return_demean"] , data_te["preds_demean"])[0,1]))
    logger.info("val__all_corrcoef_Nodemean:" + str(np.corrcoef(data_te["return_demean"] , data_te["preds"])[0,1]))
    logger.info("")


    logger.info("")
    for y in range(2010,2015):
        temp_df = data_tr[data_tr["date_time"].dt.year == y ]
        logger.info("train_set"+str(y) + ":" + str(np.corrcoef(temp_df["return_demean"], temp_df["preds_demean"])[0,1]))    
    logger.info("train_corrcoef:" + str(np.corrcoef(data_tr["return_demean"] , data_tr["preds_demean"])[0,1]))
    logger.info("train_corrcoef_Nodemean:" + str(np.corrcoef(data_tr["return_demean"] , data_tr["preds"])[0,1]))
    logger.info("")
    tree_nums = bst.best_ntree_limit
    logger.info("tree_nums:" + str(tree_nums))

    
    for y in range(2015,2019):
        temp_df = data_fin[data_fin["date_time"].dt.year == y ]
        gb = temp_df.groupby(["unique_symbol"])
        auto_corr = []
        real_auto_corr = []
        for _,frame in gb:
            l1 = list(frame["preds"].shift(1))[1:]
            l2 = list(frame["preds"])[1:]
            r1 = list(frame["return"].shift(1))[1:]
            r2 = list(frame["return"])[1:]
            corr = np.corrcoef(l1, l2)[0,1]
            if np.isnan(corr):
#                 print(_)
                continue
            re_corr = np.corrcoef(r1, r2)[0,1]
            auto_corr.append(corr)
            real_auto_corr.append(re_corr)
        logger.info("pred auto corr"+ str(y) +":" + str(np.mean(auto_corr)))
        logger.info("return auto corr"+ str(y) +":" + str(np.mean(real_auto_corr)))
        

    gb = data_fin.groupby(["unique_symbol"])
    auto_corr = []
    real_auto_corr = []
    for _,frame in gb:
        l1 = list(frame["preds"].shift(1))[1:]
        l2 = list(frame["preds"])[1:]
        r1 = list(frame["return"].shift(1))[1:]
        r2 = list(frame["return"])[1:]
        corr = np.corrcoef(l1, l2)[0,1]
        if np.isnan(corr):
            print(_)
            continue
        re_corr = np.corrcoef(r1, r2)[0,1]
        auto_corr.append(corr)
        real_auto_corr.append(re_corr)
    logger.info("pred auto corr" +":" + str(np.mean(auto_corr)))
    logger.info("return auto corr"+":" + str(np.mean(real_auto_corr)))
        
    
    return bst



def bstlist2forecast(bst_list, model_f_list, data, use_f, t3, t4, params): 
    num_rounds =1000
    early = 30   
    logger = logging.getLogger(str(time.ctime()))
    logger.setLevel(level = logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler("log/batch_train:" + str(time.ctime()) + ".txt")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

#     记录params
    logger.info(params)
#     记录factor
    logger.info("factor:" + "|".join(use_f))

#     选择数据集
    data_fin = data[ (data["date_time"] >= t3) & (data["date_time"] < t4)].copy()

#     记录结果
    fin_pre = []
    te_pre = []
    tr_pre = []
    for i in range(len(bst_list)):
        bst = bst_list[i]
        t_use_f = model_f_list[i]
        tree_nums = bst.best_ntree_limit 
        tr_pre.append(bst.predict(xgb.DMatrix(data_fin[t_use_f]), ntree_limit=tree_nums))     
    tr_pre = np.array(tr_pre)          
    data_fin["preds"] = tr_pre.mean(axis = 0)
    
    gb = data_fin.groupby(['date_time'])['preds']
    data_fin["pred_mean"] = gb.transform(lambda x: x.mean())
    data_fin["preds_demean"] = data_fin["preds"] - data_fin["pred_mean"]
    data_fin["return_demean"] = data_fin["return"] - data_fin["mr"]
   
    plot_conditional_expectation(np.array(data_fin["preds_demean"]), np.array(data_fin["return_demean"]))
    logger.info("")
    for y in range(2015,2019):
        temp_df = data_fin[data_fin["date_time"].dt.year == y ]
        logger.info("test_set" + str(y) + ":"+ str(np.corrcoef(temp_df["return_demean"], temp_df["preds_demean"])[0,1]))
    logger.info("test_all_corrcoef:" + str(np.corrcoef(data_fin["return_demean"] , data_fin["preds_demean"])[0,1]))
    logger.info("test_all_corrcoef_Nodemean:" + str(np.corrcoef(data_fin["return_demean"] , data_fin["preds"])[0,1])) 
    logger.info("")

    for y in range(2015,2019):
        temp_df = data_fin[data_fin["date_time"].dt.year == y ]
        gb = temp_df.groupby(["unique_symbol"])
        auto_corr = []
        real_auto_corr = []
        for _,frame in gb:
            l1 = list(frame["preds"].shift(1))[1:]
            l2 = list(frame["preds"])[1:]
            r1 = list(frame["return"].shift(1))[1:]
            r2 = list(frame["return"])[1:]
            corr = np.corrcoef(l1, l2)[0,1]
            if np.isnan(corr):
#                 print(_)
                continue
            re_corr = np.corrcoef(r1, r2)[0,1]
            auto_corr.append(corr)
            real_auto_corr.append(re_corr)
        logger.info("pred auto corr"+ str(y) +":" + str(np.mean(auto_corr)))
        logger.info("return auto corr"+ str(y) +":" + str(np.mean(real_auto_corr)))
        
    gb = data_fin.groupby(["unique_symbol"])
    auto_corr = []
    real_auto_corr = []
    for _,frame in gb:
        l1 = list(frame["preds"].shift(1))[1:]
        l2 = list(frame["preds"])[1:]
        r1 = list(frame["return"].shift(1))[1:]
        r2 = list(frame["return"])[1:]
        corr = np.corrcoef(l1, l2)[0,1]
        if np.isnan(corr):
            print(_)
            continue
        re_corr = np.corrcoef(r1, r2)[0,1]
        auto_corr.append(corr)
        real_auto_corr.append(re_corr)
    logger.info("pred auto corr" +":" + str(np.mean(auto_corr)))
    logger.info("return auto corr"+":" + str(np.mean(real_auto_corr)))
    logger.info("tree_nums:" + str(-999))
    
    return fin_pre.mean(axis = 0)


        
        
def get_forecast(data_sim):
    
    data_sim["date_time"] = data_sim["date_time"] + datetime.timedelta(hours=9.5)
    data_sim["date_time"] = data_sim["date_time"].apply(lambda x:str(x))
    data_sim["unique_symbol"]  = data_sim["unique_symbol"].apply(lambda x: 104060000000 + x if x>=600000 else 104070000000 + x)
    data_sim["unique_symbol"]  = data_sim["unique_symbol"].apply(lambda x: str(x))
    data_sim = data_sim.reset_index(drop = True)
    symbol = sorted(list(set(data_sim["unique_symbol"])))
    timestamp = sorted(list(set(data_sim["date_time"])))
    forecast = []
    for s in range(len(symbol)):
        forecast.append(np.array([0]*len(timestamp)))
    forecast = np.array(forecast, dtype=np.float)
    
    
    for _, row in enumerate(data_sim.values):
        if _%5000 == 0:
            print(_, len(data_sim))
        for t in range(len(timestamp)):
            if timestamp[t] == row[0]:
                break
        for s in range(len(symbol)):
            if symbol[s] == row[2]:
                break
#         print(row[1])
        forecast[s][t] = row[1]
    forecast_df = pd.DataFrame()
    
    for i in range(len(symbol)):
        forecast_df[symbol[i]] = forecast[i][:-1]
        
    timestamp = timestamp[1:]
    forecast_df.index = timestamp
    forecast_df.to_csv( '/home/huandong/sim_data/forecast.csv',index=None, header=None)
    timestamp_df = pd.DataFrame({"timestamp":timestamp})
    timestamp_df["timestamp"] = timestamp_df["timestamp"].apply(lambda x:x.replace('-',""))
    timestamp_df.to_csv( '/home/huandong/sim_data/timestamp.csv',index=None, header=None)
    symbol_df = pd.DataFrame({"symbol":symbol})
    symbol_df.to_csv( '/home/huandong/sim_data/symbol.csv',index=None, header=None)
    
    return forecast_df, forecast
    
def get_return(data_sim):
    
    data_sim["date_time"] = data_sim["date_time"] + datetime.timedelta(hours=9.5)
    data_sim["date_time"] = data_sim["date_time"].apply(lambda x:str(x))
    data_sim["unique_symbol"] = data_sim["unique_symbol"].apply(lambda x:str(x))
    data_sim["unique_symbol"]  = data_sim["unique_symbol"].apply(lambda x: x if len(x)==6 else (6-len(x))*"0"+x)
    data_sim["unique_symbol"]  = data_sim["unique_symbol"].apply(lambda x: str(104060)+x)
    data_sim = data_sim.reset_index(drop = True)
    symbol = sorted(list(set(data_sim["unique_symbol"])))
    timestamp = sorted(list(set(data_sim["date_time"])))
    return_m = []
    for s in range(len(symbol)):
        return_m.append(np.array([0]*len(timestamp)))
    return_m = np.array(return_m, dtype=np.float)   
    
    for _, row in enumerate(data_sim.values):
        if _%5000 == 0:
            print(_, len(data_sim))
        for t in range(len(timestamp)):
            if timestamp[t] == row[0]:
                break
        for s in range(len(symbol)):
            if symbol[s] == row[2]:
                break
#         print(row[1])
        return_m[s][t] = row[1]
    return_df = pd.DataFrame()
    
    for i in range(len(symbol)):
        return_df[symbol[i]] = return_m[i][:-1]
        
    timestamp = timestamp[1:]
    return_df.index = timestamp
    return_df.to_csv( '/home/huandong/sim_data/return.csv',index=None, header=None)
    
    return return_df, return_m
