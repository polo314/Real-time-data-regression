from clean2 import*
import pandas as pd
import matplotlib.pyplot as plt
import math
import datetime
import time


def main():


    loop_set=[3,5]

    set3=[] #labels scaled at different window sizes
    set4=[] #labels without scaling
    for i in range(0,len(loop_set)):
        set3.extend(['totalmovavg_predictclose'+str(loop_set[i])+'_'+str(15*loop_set[i])+'0'])
        set3.extend(['totalmovavg_predictclose'+str(loop_set[i])+'_'+str(15*loop_set[i])+'1'])
        set3.extend(['totalmovavg_predictclose'+str(loop_set[i])+'_'+str(15*loop_set[i])+'2'])

        set4.extend(['totalmovavg_predictclose'+str(loop_set[i])+'_'+str(15*loop_set[i])])

    data_window=pd.DataFrame()
    data_window_labels=pd.DataFrame()
    final_data=pd.DataFrame()
    predictors_1=pd.DataFrame()
    predictors=pd.DataFrame()
    predictors_final=pd.DataFrame()
    data_copy_labels=pd.DataFrame()
    data_predict=pd.DataFrame()
    close_win=pd.DataFrame()
    data=pd.DataFrame()
    data_copy=pd.DataFrame()
    labe_train=pd.DataFrame()
    labe_test=pd.DataFrame()
    data_la=pd.DataFrame()
    data_confr=pd.DataFrame()

    final_data.loc[0,'predicted_close']=0
    final_data.loc[0,'predicted_close_high']=0
    final_data.loc[0,'predicted_close_low']=0

    now=datetime.datetime.now()
    day=now.strftime('%d')
    hour=now.strftime('%H')
    now=now.strftime('%M')
    size0=1999 #a too small size0 can lead to insufficient data to be elaborated
    now1=int(day)*1440+int(hour)*60+int(now)+size0
    now=int(day)*1440+int(hour)*60+int(now)
    set_windows=[7,8,8.5]
    starters=[]
    size_=size0-15*loop_set[len(loop_set)-1]
    for i in set_windows:
        starters.extend([size_-int(i*size_/9)])

    delay_max_window=20
    count=0
    count1=0
    lab_tra=0

    x=[]
    y=[]
    yy=[]
    ya=[]
    yb=[]
    yc=[]

    x.extend([count1])

    plt.ion()
    fig=plt.figure()
    ax1=fig.add_subplot(1,1,1)

    from sklearn.externals import joblib
    gbrt_fin=joblib.load('gbrt_final_close')

    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler()



    while now1-now>0:
        size=now1-int(now)
        now=datetime.datetime.now()
        d=now.strftime('%d')
        h=now.strftime('%H')
        now=now.strftime('%M')
        now=int(d)*1440+int(h)*60+int(now)
        data_cycle=data_creator('BTC','EUR',size,1)
        data=data.shift(-(size))
        data.drop(data.tail(size+1).index,inplace=True)
        frame_cycle=[data,data_cycle]
        data=pd.concat(frame_cycle)
        data=data.reset_index()
        data=data.drop(['index'],axis=1)
        data_feat=pd.DataFrame()
        data_feat=data.copy()
        data_feat=data_feat.iloc[len(data_feat)-size0-1:,:]
        data_feat=data_feat.reset_index()
        data_feat=data_feat.drop(['index'],axis=1)
        last_data=size+1
        seconds=datetime.datetime.now()
        seconds=seconds.strftime('%S')

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

            if last_data == len(data_feat):
                movavfun=total_movavg_predict(data_feat,'close',short_window,window1,window2,window3,window4,window5)
                avg_close1=total_movavg(data_feat,'close',short_window,window1,window2,window3,window4,window5)


            #Features

                short_window=int(round(1+short_window/2))
                window1=int(round(1+window1/2))
                window2=int(round(1+window2/2))
                window3=int(round(1+window3/2))
                window4=int(round(1+window4/2))
                window5=int(round(1+window5/2))
                local_variance_window=int(round(1+local_variance_window/2))
                slope_window=int(round(1+slope_window/2))

                avg_close=total_movavg(data_feat,'close',short_window,window1,window2,window3,window4,window5)
                avg_close_root=movavg(data_feat,'close',short_window)
                local_variance_close=local_msq(data_feat,'close',avg_close,local_variance_window)
                msroot_close=msroot(data_feat,'close',avg_close_root,short_window)
                entropy_close=entropy(data_feat,'close',msroot_close,short_window,size)
                local_entropy_close=entropy(data_feat,'close',local_variance_close,short_window,size)
                avg_entropy_close=movavg(data_feat,entropy_close[1],short_window)
                slope_close=slope(data_feat,'close',slope_window)
                avg_slope=total_movavg(data_feat,slope_close,short_window,window1,window2,window3,window4,window5)

                avg_slope_root=movavg(data_feat,slope_close,short_window)
                local_variance_slope=local_msq(data_feat,slope_close,avg_slope,local_variance_window)
                msroot_slope=msroot(data_feat,slope_close,avg_slope_root,short_window)
                entropy_slope=entropy(data_feat,slope_close,msroot_slope,short_window,size)
                local_entropy_slope=entropy(data_feat,slope_close,local_variance_slope,short_window,size)
                avg_entropy_slope=movavg(data_feat,entropy_slope[1],short_window)

                data_feat['high_close'+str(loop_set[i])]=data_feat[avg_close]+data_feat[local_variance_close]
                avg_high=total_movavg(data_feat,'high_close'+str(loop_set[i]),short_window,window1,window2,window3,window4,window5)
                avg_high_root=movavg(data_feat,'high_close'+str(loop_set[i]),short_window)
                local_variance_high=local_msq(data_feat,'high_close'+str(loop_set[i]),avg_high,local_variance_window)
                msroot_high=msroot(data_feat,'high_close'+str(loop_set[i]),avg_high_root,short_window)
                entropy_high=entropy(data_feat,'high_close'+str(loop_set[i]),msroot_high,short_window,size)
                local_entropy_high=entropy(data_feat,'high_close'+str(loop_set[i]),local_variance_high,short_window,size)
                avg_entropy_high=movavg(data_feat,entropy_high[1],short_window)


                data_feat['low_close'+str(loop_set[i])]=data_feat[avg_close]-data_feat[local_variance_close]
                avg_low=total_movavg(data_feat,'low_close'+str(loop_set[i]),short_window,window1,window2,window3,window4,window5)
                avg_low_root=movavg(data_feat,'low_close'+str(loop_set[i]),short_window)
                local_variance_low=local_msq(data_feat,'low_close'+str(loop_set[i]),avg_high,local_variance_window)
                msroot_low=msroot(data_feat,'low_close'+str(loop_set[i]),avg_low_root,short_window)
                entropy_low=entropy(data_feat,'low_close'+str(loop_set[i]),msroot_low,short_window,size)
                local_entropy_low=entropy(data_feat,'low_close'+str(loop_set[i]),local_variance_low,short_window,size)
                avg_entropy_low=movavg(data_feat,entropy_low[1],short_window)


            else:

                #Labels

                movavfun=total_movavg_predict_cycle(data_feat,'close',last_data,short_window,window1,window2,window3,window4,window5)
                avg_close1=total_movavg_cycle(data_feat,'close',last_data,short_window,window1,window2,window3,window4,window5)

                #Features

                short_window=int(round(1+short_window/2))
                window1=int(round(1+window1/2))
                window2=int(round(1+window2/2))
                window3=int(round(1+window3/2))
                window4=int(round(1+window4/2))
                window5=int(round(1+window5/2))
                local_variance_window=int(round(1+local_variance_window/2))
                slope_window=int(round(1+slope_window/2))

                avg_close=total_movavg_cycle(data_feat,'close',last_data,short_window,window1,window2,window3,window4,window5)
                avg_close_root=movavg_cycle(data_feat,'close',short_window,last_data)
                local_variance_close=local_msq_cycle(data_feat,'close',avg_close,local_variance_window,last_data)
                msroot_close=msroot(data_feat,'close',avg_close_root,short_window)
                entropy_close=entropy(data_feat,'close',msroot_close,short_window,size)
                local_entropy_close=entropy(data_feat,'close',local_variance_close,short_window,size)
                avg_entropy_close=movavg_cycle(data_feat,entropy_close[1],short_window,last_data)
                slope_close=slope_cycle(data_feat,'close',slope_window,last_data)
                avg_slope=total_movavg_cycle(data_feat,slope_close,last_data,short_window,window1,window2,window3,window4,window5)

                avg_slope_root=movavg_cycle(data_feat,slope_close,short_window,last_data)
                local_variance_slope=local_msq_cycle(data_feat,slope_close,avg_slope,local_variance_window,last_data)
                msroot_slope=msroot(data_feat,slope_close,avg_slope_root,short_window)
                entropy_slope=entropy(data_feat,slope_close,msroot_slope,short_window,size)
                local_entropy_slope=entropy(data_feat,slope_close,local_variance_slope,short_window,size)
                avg_entropy_slope=movavg_cycle(data_feat,entropy_slope[1],short_window,last_data)

                data_feat['high_close'+str(loop_set[i])]=data_feat[avg_close]+data_feat[local_variance_close]
                avg_high=total_movavg_cycle(data_feat,'high_close'+str(loop_set[i]),last_data,short_window,window1,window2,window3,window4,window5)
                avg_high_root=movavg_cycle(data_feat,'high_close'+str(loop_set[i]),short_window,last_data)
                local_variance_high=local_msq_cycle(data_feat,'high_close'+str(loop_set[i]),avg_high,local_variance_window,last_data)
                msroot_high=msroot(data_feat,'high_close'+str(loop_set[i]),avg_high_root,short_window)
                entropy_high=entropy(data_feat,'high_close'+str(loop_set[i]),msroot_high,short_window,size)
                local_entropy_high=entropy(data_feat,'high_close'+str(loop_set[i]),local_variance_high,short_window,size)
                avg_entropy_high=movavg_cycle(data_feat,entropy_high[1],short_window,last_data)


                data_feat['low_close'+str(loop_set[i])]=data_feat[avg_close]-data_feat[local_variance_close]
                avg_low=total_movavg_cycle(data_feat,'low_close'+str(loop_set[i]),last_data,short_window,window1,window2,window3,window4,window5)
                avg_low_root=movavg_cycle(data_feat,'low_close'+str(loop_set[i]),short_window,last_data)
                local_variance_low=local_msq_cycle(data_feat,'low_close'+str(loop_set[i]),avg_high,local_variance_window,last_data)
                msroot_low=msroot(data_feat,'low_close'+str(loop_set[i]),avg_low_root,short_window)
                entropy_low=entropy(data_feat,'low_close'+str(loop_set[i]),msroot_low,short_window,size)
                local_entropy_low=entropy(data_feat,'low_close'+str(loop_set[i]),local_variance_low,short_window,size)
                avg_entropy_low=movavg_cycle(data_feat,entropy_low[1],short_window,last_data)

        if last_data == len(data_feat):
            data_labels=pd.DataFrame()
            labels(data_labels,data_feat,'close',loop_set)
            lista=list(data_labels.columns.values)

        quantity=int(round((loop_set[len(loop_set)-1]/2)+1))

        if last_data != len(data_feat):
            data_final=data_feat.iloc[len(data_feat)-(size+quantity+1):,:]
            data.drop(data.tail(quantity+size+1).index,inplace=True)
        else:
            data_final=data_feat.iloc[len(data_feat)-(size+1):,:]
            data.drop(data.tail(size+1).index,inplace=True)

        frame0=[data,data_final]
        data=pd.concat(frame0)

        now1=datetime.datetime.now()
        d1=now1.strftime('%d')
        h1=now1.strftime('%H')
        m1=now1.strftime('%M')
        seconds1=now1.strftime('%S')
        now1=int(d1)*1440+int(h1)*60+int(m1)
        size1=now1-int(now)
        difsec=int(seconds1)+60*size1-int(seconds)

        if size1==1 and 60-int(seconds1)<int(difsec/size):
            time.sleep(60-int(seconds1)+1)
            now1=datetime.datetime.now()
            d1=now1.strftime('%d')
            h1=now1.strftime('%H')
            m1=now1.strftime('%M')
            now1=int(d1)*1440+int(h1)*60+int(m1)
            print(now1)
            print('i waited a little')
            print(int(difsec/size))


        data_work=data.copy()
        data_copy_labels=pd.DataFrame()
        labels(data_copy_labels,data_work,'close',loop_set)
        clean_labels(data_work,'close',loop_set)
        lista=list(data_labels.columns.values)
        data_work=data_work.dropna()
        data_work=data_work.reset_index()
        data_work=data_work.drop(['index'],axis=1)
        len1=starters[0]+21+150
        data_work=data_work.iloc[len(data_work)-starters[0]-21-150:,:]
        data_copy_labels=data_copy_labels.iloc[len(data_copy_labels)-starters[0]-21-150:,:]
        data_work=data_work.reset_index()
        data_work=data_work.drop(['index'],axis=1)
        data_copy_labels=data_copy_labels.reset_index()
        data_copy_labels=data_copy_labels.drop(['index'],axis=1)
        len2=len(data_work)
        if len1 != len2:
            print('Warning, data_work length is varying!')

        data_confr['totalmovavgclose'+str(loop_set[0])+'_'+str(15*loop_set[0])]=data_work['totalmovavgclose'+str(loop_set[0])+'_'+str(15*loop_set[0])]
        data_work=data_work.drop(['totalmovavgclose'+str(loop_set[0])+'_'+str(15*loop_set[0])],axis=1)
        data_work=data_work.drop(['totalmovavgclose'+str(loop_set[1])+'_'+str(15*loop_set[1])],axis=1)
        data_work=data_work.drop(['date'],axis=1)
        data_work=data_work.drop(['time'],axis=1)
        data_iterator=pd.DataFrame()

        for q in starters:

            for h in range(0,len(lista)):
                    name=lista[h]
                    data_iterator.loc[0,'variance_pred'+str(starters.index(q))+name]=0
                    data_iterator.loc[0,'variance_gbrt'+str(starters.index(q))+name]=0
                    data_iterator.loc[0,'variance_ada'+str(starters.index(q))+name]=0
                    data_iterator.loc[0,'counter'+str(starters.index(q))+name]=0


        if close_win.empty:

            true_features = pd.read_csv('folder address'+'true_features.txt')
            true_features=true_features.drop(['Unnamed: 0'],axis=1)
        true_f=[]
        for l in true_features.columns.values:
            true_f.extend([l])
        set1=[]
        for k in data_work.columns.values:
            set1.extend([k])
        set2=set(set1) - set(true_f)
        set2=list(set2)

        for j in starters:
            start=j
            start0=starters[0]
            for i in range(start0,len(data_work)+1):
                data_copy=data_work['close'].iloc[i-start:i]
                data_copy=data_copy.values.reshape(-1,1)
                data_copy=pd.DataFrame(scaler.fit_transform(data_copy))
                close_win.loc[i-start0,'close_'+str(start)]=data_copy[0][len(data_copy)-1]
                del data_copy
                data_copy=pd.DataFrame()

    else:

        close_win.drop(close_win.tail(1).index,inplace=True)
        close_win=close_win.shift(-(size))
        close_win=close_win.dropna()
        for j in starters:
            start=j
            start0=starters[0]
            for i in range(len(data_work)-(size),len(data_work)+1):
                data_copy=data_work['close'].iloc[i-start:i]
                data_copy=data_copy.values.reshape(-1,1)
                data_copy=pd.DataFrame(scaler.fit_transform(data_copy))
                close_win.loc[i-start0,'close_'+str(start)]=data_copy[0][len(data_copy)-1]
                del data_copy
                data_copy=pd.DataFrame()

    if size1>=0 and count >0:

            data_work1=data_work
            for o in set2:
                data_work1=data_work1.drop([str(o)],axis=1)
            predictors_final=predictors_final.shift(-size)
            predictors_final=predictors_final.dropna()
            predictors_final=predictors_final.reset_index()
            predictors_final=predictors_final.drop(['index'],axis=1)

            for i in range(len(data_work1)-(size+quantity-1),len(data_work1)+1):

                del predictors_1
                predictors_1=pd.DataFrame()

                for j in starters:
                    start=j
                    start0=starters[0]
                    data_copy=data_work1.copy()
                    data_copy1=data_copy_labels.copy()
                    data_copy=data_work1.iloc[i-start:i,:]
                    data_copy1=data_copy_labels.iloc[i-start:i,:]
                    data_copy1=data_copy1.reset_index()
                    data_copy1=data_copy1.drop(['index'],axis=1)
                    for b in data_copy1.columns.values:
                        if data_copy1[b].isnull().values.any():
                            data_la.loc[0,b]=data_copy1[b][len(data_copy1)-1]

                        else:
                            data_copy_=data_copy1[b]
                            data_copy_=data_copy_.values.reshape(-1,1)
                            data_copy_=pd.DataFrame(scaler.fit_transform(data_copy_))
                            data_copy_=data_copy_.iloc[len(data_copy_)-1:len(data_copy_)]
                            data_copy_=data_copy_.rename(index=str,columns={data_copy_.columns.values[0]:b})
                            data_copy_=data_copy_.reset_index()
                            data_la.loc[0,b]=data_copy_[b][0]
                    data_copy=pd.DataFrame(scaler.fit_transform(data_copy),columns=data_copy.columns)
                    data_copy=data_copy.iloc[start-1:start,:]
                    predictors=blender1(data_copy,data_la,lista,str(starters.index(start)))
                    predictors_1=pd.concat([predictors_1,predictors],axis=1)
                predictors_final=pd.concat([predictors_final,predictors_1])
                predictors_final=predictors_final.reset_index()
                predictors_final=predictors_final.drop(['index'],axis=1)
            labe_train['label']=predictors_final[set3[2]] #one among set3 labels

            labe_train=labe_train.dropna()

            lab_tra=labe_train['label'][len(labe_train)-1]

            del labe_train['label']
            labe_train=pd.DataFrame()

            labe_test['label']=data_copy_labels[set4[0]] #one among set4 with the same loop_set value of the set3's one
            labe_test=labe_test.dropna()
            lab_tes=labe_test['label'][len(labe_test)-1]
            del labe_test['label']
            labe_test=pd.DataFrame()

            final_data.loc[count1-1,'close_to_be_predicted']=lab_tes
            y2=final_data['close_to_be_predicted'][count1-1]
            yy.extend([y2])
            line1=ax1.plot(x,yy,color='green',linewidth=2)
            fig.canvas.draw()

            predictors_final1=predictors_final.copy()

            predictors_final1=predictors_final1.iloc[len(predictors_final1)-1:len(predictors_final1),:]

            for i in set3:
                del predictors_final1[i]
            close_prediction=gbrt_fin.predict(predictors_final1)
            count1+=1

            if final_data['predicted_close'][0]==0:
                final_data.loc[0,'predicted_close']=lab_tes
                y1=final_data['predicted_close'][0]
                y.extend([y1])
                ya.extend([y1])
                yb.extend([y1])
                line=ax1.plot(x,y,color='black',linewidth=2)
                line2=ax1.plot(x,ya,color='red',linewidth=2)
                line3=ax1.plot(x,yb,color='blue',linewidth=2)
                fig.canvas.draw()

            h=lab_tra
            dif_relat=(close_prediction-h)
            dif_pred=dif_relat*(1/math.sqrt(6))* data_work['msq'+str(loop_set[0])+'_close'][len(data_work)-1]
            final_data.loc[count1,'predicted_close']=lab_tes+dif_pred#close_prediction
            final_data.loc[count1,'predicted_close_high']=lab_tes+dif_pred+(1/math.sqrt(6))*data_work['msq'+str(loop_set[0])+'_close'][len(data_work)-1]
            final_data.loc[count1,'predicted_close_low']=lab_tes+dif_pred-(1/math.sqrt(6))*data_work['msq'+str(loop_set[0])+'_close'][len(data_work)-1]
            y1=final_data['predicted_close'][count1]
            y2=final_data['predicted_close_high'][count1]
            y3=final_data['predicted_close_low'][count1]
            x.extend([count1])
            y.extend([y1])
            ya.extend([y2])
            yb.extend([y3])

            print(x)
            print(y)
            print(yy)
            print(ya)
            print(yb)

            line=ax1.plot(x,y,color='black',linewidth=2)
            line2=ax1.plot(x,ya,color='red',linewidth=2)
            line3=ax1.plot(x,yb,color='blue',linewidth=2)
            fig.canvas.draw()
            now4=datetime.datetime.now()
            d4=now4.strftime('%d')
            h4=now4.strftime('%H')
            m4=now4.strftime('%M')
            seconds4=now4.strftime('%S')
            now4=int(d4)*1440+int(h4)*60+int(m4)
            size4=now4-int(now)
            if size4>0:
                print('processing time higher than 1 minute')
            if size4<6:
                difference=6-size4
            else:
                print('too much time')
                difference=1
            time.sleep(difference*60-int(seconds4)+1)
            now1=datetime.datetime.now()
            d1=now1.strftime('%d')
            h1=now1.strftime('%H')
            m1=now1.strftime('%M')
            se=now1.strftime('%S')
            now1=int(d1)*1440+int(h1)*60+int(m1)
    if count==0:
        data_work1=data_work
        for o in set2:
            data_work1=data_work1.drop([str(o)],axis=1)
        for i in range(len(data_work1)-delay_max_window-1,len(data_work1)+1):
            del predictors_1
            predictors_1=pd.DataFrame()
            for j in starters:
                start=j
                start0=starters[0]
                data_copy=data_work1.copy()
                data_copy1=data_copy_labels.copy()
                data_copy=data_work1.iloc[i-start:i,:]
                data_copy1=data_copy_labels.iloc[i-start:i,:]
                data_copy1=data_copy1.reset_index()
                data_copy1=data_copy1.drop(['index'],axis=1)
                for b in data_copy1.columns.values:
                        if data_copy1[b].isnull().values.any():
                            data_la.loc[0,b]=data_copy1[b][len(data_copy1)-1]
                        else:
                            data_copy_=data_copy1[b]
                            data_copy_=data_copy_.values.reshape(-1,1)
                            data_copy_=pd.DataFrame(scaler.fit_transform(data_copy_))
                            data_copy_=data_copy_.iloc[len(data_copy_)-1:len(data_copy_)]
                            data_copy_=data_copy_.rename(index=str,columns={data_copy_.columns.values[0]:b})
                            data_copy_=data_copy_.reset_index()
                            data_la.loc[0,b]=data_copy_[b][0]
                data_copy=pd.DataFrame(scaler.fit_transform(data_copy),columns=data_copy.columns)
                data_copy=data_copy.iloc[start-1:start,:]
                predictors=blender1(data_copy,data_la,lista,str(starters.index(start)))
                predictors_1=pd.concat([predictors_1,predictors],axis=1)
            predictors_final=pd.concat([predictors_final,predictors_1])
            predictors_final=predictors_final.reset_index()
            predictors_final=predictors_final.drop(['index'],axis=1)
        count+=1
        now4=datetime.datetime.now()
        d4=now4.strftime('%d')
        h4=now4.strftime('%H')
        m4=now4.strftime('%M')
        seconds4=now4.strftime('%S')
        now4=int(d4)*1440+int(h4)*60+int(m4)
        size4=now4-int(now)
        if size4>0:
            print('processing time higher than 1 minute')
        time.sleep(60-int(seconds4)+1)
        now1=datetime.datetime.now()
        d1=now1.strftime('%d')
        h1=now1.strftime('%H')
        m1=now1.strftime('%M')
        se=now1.strftime('%S')
        now1=int(d1)*1440+int(h1)*60+int(m1)
        print(int(se))
        print(now1)
        print(now)
        print('end')

if __name__=='__main__':
    main()
