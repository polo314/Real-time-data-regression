from tools_data_download import *
from __future__ import division
import pandas as pd
import numpy as np
import math
import requests
import datetime

def main():


    loop_set=[3,5] #set of time windows for features computation

    now=datetime.datetime.now()
    year=now.year
    month=now.strftime('%m')
    today=now.strftime('%d')
    datafr=pd.DataFrame()
    dataframetotal=pd.DataFrame()
    windowrange=5
    hour_loading=' 10_00_00_.txt'
    for i in range(int(today)-windowrange,int(today)):
        address='folder address'+'minuteBTC_'+str(year)+'-'+str(month)+'-'+str(i).zfill(2)+ hour_loading
        datafr = pd.read_csv(address)
        dataframetotal=pd.concat([dataframetotal,datafr]).reset_index(drop=True)
    dataframetotal=dataframetotal.drop(['Unnamed: 0'], axis=1)
    file=open('folder address'+'name.txt','w')
    file.write(dataframetotal.to_csv())
    file.close()

    data = pd.read_csv('folder address'+'name.txt')
    data = pd.DataFrame(data)


    for i in range(0,len(loop_set)):
        short_window=loop_set[i]
        window1=2*loop_set[i]
        window2=4*loop_set[i]
        window3=6*loop_set[i]
        window4=10*loop_set[i]
        window5=15*loop_set[i]

        local_variance_window=int(round(1+short_window/2))
        slope_window=int(round(1+(short_window/2)))

        #Labels

        movavfun=total_movavg_predict(data,'close',short_window,window1,window2,window3,window4,window5)
        local_variance=local_msq(data,'close',movavfun,local_variance_window)
        data[local_variance]=data[local_variance].shift(-1)

                                    #4
        #Features

        short_window=int(round(1+short_window/2))
        window1=int(round(1+window1/2))
        window2=int(round(1+window2/2))
        window3=int(round(1+window3/2))
        window4=int(round(1+window4/2))
        window5=int(round(1+window5/2))
        local_variance_window=int(round(1+local_variance_window/2))
        slope_window=int(round(1+slope_window/2))

        avg_close=total_movavg(data,'close',short_window,window1,window2,window3,window4,window5)
        avg_close_root=movavg(data,'close',short_window)
        local_variance_close=local_msq(data,'close',avg_close,local_variance_window)
        msroot_close=msroot(data,'close',avg_close_root,short_window)
        entropy_close=entropy(data,'close',msroot_close,short_window)
        local_entropy_close=entropy(data,'close',local_variance_close,short_window)
        avg_entropy_close=movavg(data,entropy_close[1],short_window)
                                    #18
        slope_close=slope(data,'close',slope_window)
        avg_slope=total_movavg(data,slope_close,short_window,window1,window2,window3,window4,window5)
        avg_slope_root=movavg(data,slope_close,short_window)
        local_variance_slope=local_msq(data,slope_close,avg_slope,local_variance_window)
        msroot_slope=msroot(data,slope_close,avg_slope_root,short_window)
        entropy_slope=entropy(data,slope_close,msroot_slope,short_window)
        local_entropy_slope=entropy(data,slope_close,local_variance_slope,short_window)
        avg_entropy_slope=movavg(data,entropy_slope[1],short_window)
                                    #19
        data['high_close'+str(loop_set[i])]=data[avg_close]+data[local_variance_close]
        avg_high=total_movavg(data,'high_close'+str(loop_set[i]),short_window,window1,window2,window3,window4,window5)
        avg_high_root=movavg(data,'high_close'+str(loop_set[i]),short_window)
        local_variance_high=local_msq(data,'high_close'+str(loop_set[i]),avg_high,local_variance_window)
        msroot_high=msroot(data,'high_close'+str(loop_set[i]),avg_high_root,short_window)
        entropy_high=entropy(data,'high_close'+str(loop_set[i]),msroot_high,short_window)
        local_entropy_high=entropy(data,'high_close'+str(loop_set[i]),local_variance_high,short_window)
        avg_entropy_high=movavg(data,entropy_high[1],short_window)
                                    #19
        data['low_close'+str(loop_set[i])]=data[avg_close]-data[local_variance_close]
        avg_low=total_movavg(data,'low_close'+str(loop_set[i]),short_window,window1,window2,window3,window4,window5)
        avg_low_root=movavg(data,'low_close'+str(loop_set[i]),short_window)
        local_variance_low=local_msq(data,'low_close'+str(loop_set[i]),avg_high,local_variance_window)
        msroot_low=msroot(data,'low_close'+str(loop_set[i]),avg_low_root,short_window)
        entropy_low=entropy(data,'low_close'+str(loop_set[i]),msroot_low,short_window)
        local_entropy_low=entropy(data,'low_close'+str(loop_set[i]),local_variance_low,short_window)
        avg_entropy_low=movavg(data,entropy_low[1],short_window)
                                    #19
    file1=open('folder address'+'dataframe_featured.txt','w')
    file1.write(data.to_csv())
    file1.close()

    file1 = pd.read_csv('folder address'+'dataframe_featured.txt')
    data=pd.DataFrame(file1)

    data=data.drop(['date'], axis=1)
    data=data.drop(['time'], axis=1)
    data=data.drop(['Unnamed: 0'], axis=1)
    data=data.drop(['Unnamed: 0.1'], axis=1)
    data=data.dropna()
    data=data.reset_index()
    data=data.drop(['index'],axis=1)

    file2=open('folder address'+'dataframe_without_nan.txt','w')
    file2.write(data.to_csv())
    file2.close()

    data_cop=data
    data_cop.drop(data_cop.index[:0],inplace=True)
    data_cop=data_cop.reset_index()
    data_cop=data_cop.drop(['index'],axis=1)
    data1=data_cop.iloc[:int(4*len(data_cop)/5),:]
    data1_3=data_cop.iloc[int(4*len(data_cop)/5):,:]
    data1_3=data1_3.reset_index()
    data1_3=data1_3.drop(['index'],axis=1)
    data1_3_labels=pd.DataFrame()
    labels(data1_3_labels,data1_3,'close',loop_set)
    clean_labels(data1_3,'close',loop_set)


    a=splitting_cleaning(data1,'close','slope'+str(loop_set[0])+'_close',loop_set,0.2,0.3,4)


    test_lay=a[0]
    test_labels_lay=a[1]
    test_labels_lay=pd.DataFrame(test_labels_lay)
    test=a[2]
    test_labels=a[3]
    train={}
    train_lab={}

    for i in xrange(0,len(a)-4,2):
        p=int(i/2)
        train['{0}'.format(p)]=a[i+4]
        train_lab['{0}'.format(p)]=a[i+5]
        train_lab['{0}'.format(p)]=pd.DataFrame(train_lab['{0}'.format(p)])
        train_lab['{0}'.format(p)]=train_lab['{0}'.format(p)].reset_index()
        train_lab['{0}'.format(p)]=train_lab['{0}'.format(p)].drop(['index'],axis=1)

    index=int((len(a)-4)/2)

    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler()

    test_labels=pd.DataFrame(scaler.fit_transform(test_labels),columns=test_labels.columns)
    test_labels_lay=pd.DataFrame(scaler.fit_transform(test_labels_lay),columns=test_labels_lay.columns)
    for j in range(0,index):
        train_lab[str(j)]=pd.DataFrame(scaler.fit_transform(train_lab[str(j)]),columns=train_lab[str(j)].columns)

    test=pd.DataFrame(scaler.fit_transform(test),columns=test.columns)
    test_lay=pd.DataFrame(scaler.fit_transform(test_lay),columns=test_lay.columns)
    for j in range(0,index):
            train[str(j)]=pd.DataFrame(scaler.fit_transform(train[str(j)]),columns=train[str(j)].columns)

    train=pd.concat([train['0'],train['1'],train['2']])
    test=pd.concat([test,test_lay])
    train_lab=pd.concat([train_lab['0'],train_lab['1'],train_lab['2']])
    test_labels=pd.concat([test_labels,test_labels_lay])


    lista=list(test_labels.columns.values)

    #machine learning

    #RANDOM FOREST REGRESSOR?
    #Random forest

    for i in range(0,len(lista)):

        name=str(lista[i])
        test_labels_lay_sing=train_lab[name]
        test_labels_sing=test_labels[name]

    #Ada Boost

        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.tree import DecisionTreeRegressor
        ada_reg=AdaBoostRegressor(
            DecisionTreeRegressor(max_depth=8),n_estimators=70, learning_rate=0.3,)

        adamodel=ada_reg.fit(train,test_labels_lay_sing)

        from sklearn.externals import joblib
        joblib.dump(adamodel,'ada_'+name)

    #Gradient Boosting

        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.metrics import mean_squared_error
        from sklearn.model_selection import train_test_split

        Xtr,Xval,Ytr,Yval=train_test_split(train,test_labels_lay_sing,test_size=0.33,random_state=42)

        gbrt=GradientBoostingRegressor(max_depth=7,n_estimators=70,learning_rate=0.1)
        gbrt.fit(Xtr,Ytr)
        errors=[mean_squared_error(Yval,predictions)
            for predictions in gbrt.staged_predict(Xval)]
        bst_n_estimators=np.argmin(errors)
        gbrt_best=GradientBoostingRegressor(max_depth=7,n_estimators=bst_n_estimators)
        modelgbrt=gbrt_best.fit(Xtr,Ytr)

        from sklearn.externals import joblib
        joblib.dump(modelgbrt,'gbrt_'+name)

         #Stacking

        from sklearn.externals import joblib

        gbrt=joblib.load('gbrt_'+name)
        ada=joblib.load('ada_'+name)

        gbrt1=gbrt.predict(test)
        ada1=ada.predict(test)
        stacking_blen=pd.DataFrame({'gbrt':gbrt1,'ada':ada1})

        from sklearn.metrics import mean_squared_error

        mse_blen_gbrt=mean_squared_error(pd.DataFrame(stacking_blen['gbrt']),pd.DataFrame(test_labels_sing))
        rmse_blen_gbrt=np.sqrt(mse_blen_gbrt)
        print(rmse_blen_gbrt)
        mse_blen_ada=mean_squared_error(pd.DataFrame(stacking_blen['ada']),pd.DataFrame(test_labels_sing))
        rmse_blen_ada=np.sqrt(mse_blen_ada)
        print(rmse_blen_ada)

        from sklearn.model_selection import GridSearchCV
        from sklearn.ensemble import RandomForestRegressor

        stacking_blen=pd.DataFrame(stacking_blen)
        length_blen=int(len(stacking_blen.columns.values))
        param_grid_final = [
            {'n_estimators': [70], 'max_features':[length_blen],'max_depth':[8],},
            #{'bootstrap': [False],'n_estimators': [30,60,120], 'max_features': [10,20]},
            ]
        forest_reg_final = RandomForestRegressor()
        grid_search_final = GridSearchCV(forest_reg_final,param_grid_final,cv=5,scoring='neg_mean_squared_error')
        model_forest_final=grid_search_final.fit(stacking_blen,test_labels_sing)

        from sklearn.externals import joblib
        joblib.dump(model_forest_final,name)



    predictors_1=pd.DataFrame()
    predictors=pd.DataFrame()
    predictors_final=pd.DataFrame()

    true_features=data1_3.iloc[0:1,:]
    true_f=open('folder address'+'true_features.txt','w')
    true_f.write(true_features.to_csv())
    true_f.close()

    set_windows=[7,8,8.5]
    starters=[]
    for i in set_windows:
        starters.extend([len(data1_3)-int(i*len(data1_3)/9)])

    close_win=pd.DataFrame()
    for j in starters:
        start=j
        start0=starters[0]
        del predictors_1
        predictors_1=pd.DataFrame()
        for i in range(start,len(data1_3)+1):
            data_copy=data1_3.iloc[i-start:i,:]
            data_copy1=data1_3_labels.iloc[i-start:i,:]
            data_copy=pd.DataFrame(scaler.fit_transform(data_copy),columns=data_copy.columns)
            data_copy1=pd.DataFrame(scaler.fit_transform(data_copy1),columns=data_copy1.columns)
            data_copy=data_copy.iloc[start-1:start,:]
            data_copy1=data_copy1.iloc[start-1:start,:]
            predictors=blender1(data_copy,data_copy1,lista,str(starters.index(start)))
            predictors_1=pd.concat([predictors_1,predictors])
            predictors_1=predictors_1.reset_index()
            predictors_1=predictors_1.drop(['index'],axis=1)
            close_win.loc[i-start,'close_'+str(start)]=data_copy['close'][start-1]

        if j!= set_windows[0]:
            predictors_1=predictors_1.shift(-(start0-start))
            predictors_1=predictors_1.dropna()
            close_win['close_'+str(start)]=close_win['close_'+str(start)].shift(-(start0-start))

        predictors_final=pd.concat([predictors_final,predictors_1],axis=1)
    close_win=close_win.dropna()

    file3=open('folder address'+'predictors_final.txt','w')
    file3.write(predictors_final.to_csv())
    file3.close()

    data_pred1 = pd.read_csv('folder address'+'predictors_final.txt')
    data_pred1 = pd.DataFrame(data_pred1)
    data_pred1=data_pred1.drop(['Unnamed: 0'],axis=1)

    set2=[]

    for i in range(0,len(loop_set)):
        set2.extend(['totalmovavg_predictclose'+str(loop_set[i])+'_'+str(15*loop_set[i])+'0'])
        set2.extend(['totalmovavg_predictclose'+str(loop_set[i])+'_'+str(15*loop_set[i])+'1'])
        set2.extend(['totalmovavg_predictclose'+str(loop_set[i])+'_'+str(15*loop_set[i])+'2'])
    labe_train=data_pred1[set2[2]] #choose one among set2 elements

    for i in set2:
        del data_pred1[i]


    #Gradient Boosting

    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split

    Xtr,Xval,Ytr,Yval=train_test_split(data_pred1,labe_train,test_size=0.33,random_state=42)

    gbrt_final=GradientBoostingRegressor(max_depth=7,n_estimators=450,learning_rate=0.1)
    gbrt_final.fit(Xtr,Ytr)
    errors_final=[mean_squared_error(Yval,predictions)
        for predictions in gbrt_final.staged_predict(Xval)]
    bst_n_estimators_final=np.argmin(errors_final)
    print(bst_n_estimators_final)
    gbrt_best_final=GradientBoostingRegressor(max_depth=7,n_estimators=bst_n_estimators_final)
    modelgbrt_final=gbrt_best_final.fit(Xtr,Ytr)

    from sklearn.externals import joblib
    joblib.dump(modelgbrt_final,'gbrt_final_close')


if __name__=='__main__':
    main()
