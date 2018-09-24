from __future__ import division
import pandas as pd
import numpy as np
import math
import requests
import datetime




def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')

def print_full_columns(x):
    pd.set_option('display.max_columns', len(x.columns.values))
    print(x)
    pd.reset_option('display.max_columns')

def slope(data,attribute,length):
    data['slope'+str(length)+'_'+str(attribute)]=pd.Series()
    ind=0
    while math.isnan(data[str(attribute)][ind])==True:
        ind+=1
    for i in range(0+length-1+ind,len(data)):
        ratio= (data[attribute][i]-data[attribute][i-length+1])/length
        data.loc[i,'slope'+str(length)+'_'+str(attribute)]= ratio
    return 'slope'+str(length)+'_'+str(attribute)

def slope_cycle(data,attribute,length,size):

    ind=0
    while math.isnan(data[str(attribute)][ind])==True:
        ind+=1
    if size != len(data):
            for i in range((len(data)-(size)),len(data)):
                ratio= (data[attribute][i]-data[attribute][i-length+1])/length
                data.loc[i,'slope'+str(length)+'_'+str(attribute)]= ratio
    return 'slope'+str(length)+'_'+str(attribute)

def msroot(data,attribute,movavg_attribute,movavg_window):
    data['dif'+str(movavg_window)+'_'+str(attribute)]=pd.Series()
    data['msq'+str(movavg_window)+'_'+str(attribute)]=pd.Series()
    a=int(movavg_window/2)
    total=0
    ind1=0
    while math.isnan(data[movavg_attribute][ind1])==True:
        ind1+=1
    ind2=0
    while math.isnan(data[attribute][ind2])==True:
        ind2+=1
    ind=max(ind1,ind2)
    for i in range(movavg_window+ind-1,len(data)):
        dif=data[str(movavg_attribute)][i]-data[attribute][i-a]
        data.loc[i,'dif'+str(movavg_window)+'_'+str(attribute)]=dif**2
    for j in range(movavg_window+ind-1,len(data)):
        total+=data['dif'+str(movavg_window)+'_'+str(attribute)][j]
        divid=j-movavg_window+2-ind
        data.loc[j,'msq'+str(movavg_window)+'_'+str(attribute)]=math.sqrt(total/divid)
    return 'msq'+str(movavg_window)+'_'+str(attribute)

def movavg(data,attribute,movavg_window):
    data['movavg'+str(movavg_window)+'_'+str(attribute)]=pd.Series()
    ind=0
    while math.isnan(data[attribute][ind])==True:
        ind+=1
    for i in range(movavg_window-1+ind,len(data)):
        counter=0
        for j in range(i-movavg_window+1,i+1):
            counter+=data[attribute][j]
        data.loc[i,'movavg'+str(movavg_window)+'_'+str(attribute)]=counter/movavg_window
    return 'movavg'+str(movavg_window)+'_'+str(attribute)

def movavg_cycle(data,attribute,movavg_window,size):
    if size !=len(data):
        for i in range(len(data)-size,len(data)):
            counter=0
            for j in range(i-movavg_window+1,i+1):
                counter+=data[attribute][j]
            data1=data['movavg'+str(movavg_window)+'_'+str(attribute)].copy()
            data1.loc[i]=counter/movavg_window
            data.loc[i,'movavg'+str(movavg_window)+'_'+str(attribute)]=data1[i]
    return 'movavg'+str(movavg_window)+'_'+str(attribute)

def movavg_retro_cycle(data,attribute,movavg_window,size):
    ind=0
    while math.isnan(data[attribute][ind])==True:
        ind+=1
    if size!=len(data):
        for i in range(len(data)-(size+1),len(data)-1):
            counter=0
            for j in range(i-movavg_window+1,i+1):
                counter+=data[attribute][j]
            data.loc[i+1,'movavg_retro'+str(movavg_window)+'_'+str(attribute)]=counter/movavg_window
    else:
        data['movavg_retro'+str(movavg_window)+'_'+str(attribute)]=pd.Series()
        for i in range(movavg_window-1+ind,len(data)-1):
            counter=0
            for j in range(i-movavg_window+1,i+1):
                counter+=data[attribute][j]
            data.loc[i+1,'movavg_retro'+str(movavg_window)+'_'+str(attribute)]=counter/movavg_window
    return 'movavg_retro'+str(movavg_window)+'_'+str(attribute)

def movavg_retro(data,attribute,movavg_window):
    data['movavg_retro'+str(movavg_window)+'_'+str(attribute)]=pd.Series()
    ind=0
    while math.isnan(data[attribute][ind])==True:
        ind+=1
    for i in range(movavg_window-1+ind,len(data)-1):
        counter=0
        for j in range(i-movavg_window+1,i+1):
            counter+=data[attribute][j]
        data.loc[i+1,'movavg_retro'+str(movavg_window)+'_'+str(attribute)]=counter/movavg_window
    return 'movavg_retro'+str(movavg_window)+'_'+str(attribute)

def local_msq(data,attribute,attribute1,msq_win):
    data['local_msq'+str(msq_win)+'_'+str(attribute)]=pd.Series()
    ind1=0
    while math.isnan(data[attribute1][ind1])==True:
        ind1+=1
    ind2=0
    while math.isnan(data[attribute][ind2])==True:
        ind2+=1
    ind=max(ind1,ind2)
    for i in range(ind,len(data)-msq_win+1):
        local_msq=0
        for j in range(i,i+msq_win):
            local_msq+=(data[str(attribute)][j]- data[str(attribute1)][j])**2
        data.loc[i+msq_win-1,'local_msq'+str(msq_win)+'_'+str(attribute)]=math.sqrt(local_msq/msq_win)
    return 'local_msq'+str(msq_win)+'_'+str(attribute)

def local_msq_cycle(data,attribute,attribute1,msq_win,size):

    ind1=0
    while math.isnan(data[attribute1][ind1])==True:
        ind1+=1
    ind2=0
    while math.isnan(data[attribute][ind2])==True:
        ind2+=1
    if size!= len(data):
        for i in range(len(data)-size-(msq_win+1),len(data)-msq_win+1):
            local_msq=0
            for j in range(i,i+msq_win):
                local_msq+=(data[str(attribute)][j]- data[str(attribute1)][j])**2
            data.loc[i+msq_win-1,'local_msq'+str(msq_win)+'_'+str(attribute)]=math.sqrt(local_msq/msq_win)
    return 'local_msq'+str(msq_win)+'_'+str(attribute)

def entropy(data,attribute,msq_attribute,movavg_window,size):
    if size == len(data):
        data['entropy_source'+str(movavg_window)+'_'+str(msq_attribute)+'_'+str(attribute)]=pd.Series()
        data['entropy'+str(movavg_window)+'_'+str(msq_attribute)+'_'+str(attribute)]=pd.Series()
        data['pr1_'+str(movavg_window)+'_'+str(msq_attribute)+'_'+str(attribute)]=pd.Series()
        data['pr2_'+str(movavg_window)+'_'+str(msq_attribute)+'_'+str(attribute)]=pd.Series()
        data['pr3_'+str(movavg_window)+'_'+str(msq_attribute)+'_'+str(attribute)]=pd.Series()
        data['variation'+'_'+'_'+str(msq_attribute)+str(attribute)]=pd.Series()
    base=math.e
    ind=0
    while math.isnan(data[str(msq_attribute)][ind])==True:
        ind+=1
    count1=math.log(3,base)/3
    count2=math.log(3,base)/3
    count3=math.log(3,base)/3
    for i in range(ind+1,len(data)):
        variation=data[str(attribute)][i]-data[str(attribute)][i-1]
        data.loc[i,'variation'+'_'+'_'+str(msq_attribute)+str(attribute)]=abs(variation)
        sum=count1+count2+count3
        normalization=1
        thr1=1*data[str(msq_attribute)][i]
        thr2=2*data[str(msq_attribute)][i]
        if abs(variation) <= thr1:
           count1+=1
           sum=count1+count2+count3
           pr1=float(count1)/sum
           pr2=float(count2)/sum
           pr3=float(count3)/sum
           entropy= -pr1*math.log(pr1,base)
           data.loc[i,'entropy'+str(movavg_window)+'_'+str(msq_attribute)+'_'+str(attribute)]= entropy
           data.loc[i,'pr1_'+str(movavg_window)+'_'+str(msq_attribute)+'_'+str(attribute)]=(-pr1*math.log(pr1,base)/normalization)
           data.loc[i,'pr2_'+str(movavg_window)+'_'+str(msq_attribute)+'_'+str(attribute)]=(-pr2*math.log(pr2,base)/normalization)
           data.loc[i,'pr3_'+str(movavg_window)+'_'+str(msq_attribute)+'_'+str(attribute)]=(-pr3*math.log(pr3,base)/normalization)
        elif thr1 < abs(variation) <= thr2:
           count2+=1
           sum=count1+count2+count3
           pr1=float(count1)/sum
           pr2=float(count2)/sum
           pr3=float(count3)/sum
           entropy= -pr2*math.log(pr2,base)
           data.loc[i,'entropy'+str(movavg_window)+'_'+str(msq_attribute)+'_'+str(attribute)]= entropy
           data.loc[i,'pr1_'+str(movavg_window)+'_'+str(msq_attribute)+'_'+str(attribute)]=(-pr1*math.log(pr1,base)/normalization)
           data.loc[i,'pr2_'+str(movavg_window)+'_'+str(msq_attribute)+'_'+str(attribute)]=(-pr2*math.log(pr2,base)/normalization)
           data.loc[i,'pr3_'+str(movavg_window)+'_'+str(msq_attribute)+'_'+str(attribute)]=(-pr3*math.log(pr3,base)/normalization)
        else:
           count3+=1
           sum=count1+count2+count3
           pr1=float(count1)/sum
           pr2=float(count2)/sum
           pr3=float(count3)/sum
           entropy= -pr3*math.log(pr3,base)
           data.loc[i,'entropy'+str(movavg_window)+'_'+str(msq_attribute)+'_'+str(attribute)]= entropy
           data.loc[i,'pr1_'+str(movavg_window)+'_'+str(msq_attribute)+'_'+str(attribute)]=(-pr1*math.log(pr1,base)/normalization)
           data.loc[i,'pr2_'+str(movavg_window)+'_'+str(msq_attribute)+'_'+str(attribute)]=(-pr2*math.log(pr2,base)/normalization)
           data.loc[i,'pr3_'+str(movavg_window)+'_'+str(msq_attribute)+'_'+str(attribute)]=(-pr3*math.log(pr3,base)/normalization)
        entropy_s=-pr1*math.log(pr1,base)-pr2*math.log(pr2,base)-pr3*math.log(pr3,base)
        data.loc[i,'entropy_source'+str(movavg_window)+'_'+str(msq_attribute)+'_'+str(attribute)]=entropy_s/3

    return ['entropy_source'+str(movavg_window)+'_'+str(msq_attribute)+'_'+str(attribute),
            'entropy'+str(movavg_window)+'_'+str(msq_attribute)+'_'+str(attribute),
            ]

def entropy_cycle(data,attribute,msq_attribute,movavg_window,size):
    data['entropy_source'+str(movavg_window)+'_'+str(msq_attribute)+'_'+str(attribute)]=pd.Series()
    data['entropy'+str(movavg_window)+'_'+str(msq_attribute)+'_'+str(attribute)]=pd.Series()
    data['pr1_'+str(movavg_window)+'_'+str(msq_attribute)+'_'+str(attribute)]=pd.Series()
    data['pr2_'+str(movavg_window)+'_'+str(msq_attribute)+'_'+str(attribute)]=pd.Series()
    data['pr3_'+str(movavg_window)+'_'+str(msq_attribute)+'_'+str(attribute)]=pd.Series()
    data['variation'+'_'+'_'+str(msq_attribute)+str(attribute)]=pd.Series()
    base=math.e
    ind=0
    while math.isnan(data[str(msq_attribute)][ind])==True:
        ind+=1
    count1=math.log(3,base)/3
    count2=math.log(3,base)/3
    count3=math.log(3,base)/3
    for i in range((len(data)-(size-1)),len(data)):
        variation=data[str(attribute)][i]-data[str(attribute)][i-1]
        data.loc[i,'variation'+'_'+'_'+str(msq_attribute)+str(attribute)]=abs(variation)
        sum=count1+count2+count3
        normalization=1#-math.log(pr1,base)-math.log(pr2,base)-math.log(pr3,base)
        thr1=1*data[str(msq_attribute)][i]
        thr2=2*data[str(msq_attribute)][i]
        if abs(variation) <= thr1:
           count1+=1
           sum=count1+count2+count3
           pr1=float(count1)/sum
           pr2=float(count2)/sum
           pr3=float(count3)/sum
           entropy= -pr1*math.log(pr1,base)
           data.loc[i,'entropy'+str(movavg_window)+'_'+str(msq_attribute)+'_'+str(attribute)]= entropy
           data.loc[i,'pr1_'+str(movavg_window)+'_'+str(msq_attribute)+'_'+str(attribute)]=(-pr1*math.log(pr1,base)/normalization)
           data.loc[i,'pr2_'+str(movavg_window)+'_'+str(msq_attribute)+'_'+str(attribute)]=(-pr2*math.log(pr2,base)/normalization)
           data.loc[i,'pr3_'+str(movavg_window)+'_'+str(msq_attribute)+'_'+str(attribute)]=(-pr3*math.log(pr3,base)/normalization)
        elif thr1 < abs(variation) <= thr2:
           count2+=1
           sum=count1+count2+count3
           pr1=float(count1)/sum
           pr2=float(count2)/sum
           pr3=float(count3)/sum
           entropy= -pr2*math.log(pr2,base)
           data.loc[i,'entropy'+str(movavg_window)+'_'+str(msq_attribute)+'_'+str(attribute)]= entropy
           data.loc[i,'pr1_'+str(movavg_window)+'_'+str(msq_attribute)+'_'+str(attribute)]=(-pr1*math.log(pr1,base)/normalization)
           data.loc[i,'pr2_'+str(movavg_window)+'_'+str(msq_attribute)+'_'+str(attribute)]=(-pr2*math.log(pr2,base)/normalization)
           data.loc[i,'pr3_'+str(movavg_window)+'_'+str(msq_attribute)+'_'+str(attribute)]=(-pr3*math.log(pr3,base)/normalization)
        else:
           count3+=1
           sum=count1+count2+count3
           pr1=float(count1)/sum
           pr2=float(count2)/sum
           pr3=float(count3)/sum
           entropy= -pr3*math.log(pr3,base)
           data.loc[i,'entropy'+str(movavg_window)+'_'+str(msq_attribute)+'_'+str(attribute)]= entropy
           data.loc[i,'pr1_'+str(movavg_window)+'_'+str(msq_attribute)+'_'+str(attribute)]=(-pr1*math.log(pr1,base)/normalization)
           data.loc[i,'pr2_'+str(movavg_window)+'_'+str(msq_attribute)+'_'+str(attribute)]=(-pr2*math.log(pr2,base)/normalization)
           data.loc[i,'pr3_'+str(movavg_window)+'_'+str(msq_attribute)+'_'+str(attribute)]=(-pr3*math.log(pr3,base)/normalization)
        entropy_s=-pr1*math.log(pr1,base)-pr2*math.log(pr2,base)-pr3*math.log(pr3,base)
        data.loc[i,'entropy_source'+str(movavg_window)+'_'+str(msq_attribute)+'_'+str(attribute)]=entropy_s/3

    return ['entropy_source'+str(movavg_window)+'_'+str(msq_attribute)+'_'+str(attribute),
            'entropy'+str(movavg_window)+'_'+str(msq_attribute)+'_'+str(attribute),
            ]

def total_movavg(data,attribute,*args):
    set=args
    dividend=0
    data['totalmovavg'+ str(attribute)+ str(set[0])+'_'+ str(set[len(set)-1])]=0
    for j in range(1,len(set)+1):
        dividend+=j
    for n in range(0,len(set)):
        average_function=movavg(data,attribute,set[n])
        data['totalmovavg'+ str(attribute)+ str(set[0])+'_'+ str(set[len(set)-1])]+=(len(set)-n)*data[average_function]
    data['totalmovavg'+ str(attribute)+ str(set[0])+'_'+ str(set[len(set)-1])]=data['totalmovavg'+ str(attribute)+ str(set[0])+'_'+ str(set[len(set)-1])]/dividend
    return 'totalmovavg' + str(attribute)+ str(set[0])+'_'+ str(set[len(set)-1])

def total_movavg_cycle(data,attribute,size,*args):
    set=args
    dividend=0
    for j in range(1,len(set)+1):
        dividend+=j
    for n in range(0,len(set)):
        average_function=movavg_cycle(data,attribute,set[n],size)
        for q in range(len(data)-(size),len(data)):
            if n==0:
                data.loc[q,'totalmovavg'+ str(attribute)+ str(set[0])+'_'+ str(set[len(set)-1])]=0
            data.loc[q,'totalmovavg'+ str(attribute)+ str(set[0])+'_'+ str(set[len(set)-1])]+=(len(set)-n)*data[average_function][q]
    for q in range(len(data)-(size),len(data)):
        data['totalmovavg'+ str(attribute)+ str(set[0])+'_'+ str(set[len(set)-1])][q]=data['totalmovavg'+ str(attribute)+ str(set[0])+'_'+ str(set[len(set)-1])][q]/dividend
    return 'totalmovavg' + str(attribute)+ str(set[0])+'_'+ str(set[len(set)-1])

def total_movavg_predict_cycle(data,attribute,size,*args):
    set=args
    dividend=0
    for j in range(1,len(set)+1):
        dividend+=j
    for n in range(0,len(set)):
        average_function=movavg_retro_cycle(data,attribute,set[n],size)
        for q in range(len(data)-(size),len(data)):
            if n==0:
                data.loc[q,'totalmovavg_predict'+ str(attribute)+ str(set[0])+'_'+ str(set[len(set)-1])]=0
            data.loc[q,'totalmovavg_predict'+ str(attribute)+ str(set[0])+'_'+ str(set[len(set)-1])]+=(len(set)-n)*data[average_function][q]
        del data[average_function]
    for q in range(len(data)-(size),len(data)):
        data['totalmovavg_predict'+ str(attribute)+ str(set[0])+'_'+ str(set[len(set)-1])][q]=data['totalmovavg_predict'+ str(attribute)+ str(set[0])+'_'+ str(set[len(set)-1])][q]/dividend
    for i in range(0-(size)+len(data)-int(round((set[0]/2)+1)),len(data)-int(round((set[0]/2)+1))):
        data.loc[i,'totalmovavg_predict'+ str(attribute)+ str(set[0])+'_'+ str(set[len(set)-1])]=data['totalmovavg_predict'+ str(attribute)+ str(set[0])+'_'+ str(set[len(set)-1])][i+int(round((set[0]/2)+1))]
    for i in range(len(data)-int(round((set[0]/2)+1)),len(data)):
        data.loc[i,'totalmovavg_predict'+ str(attribute)+ str(set[0])+'_'+ str(set[len(set)-1])]=np.nan
    return 'totalmovavg_predict' + str(attribute)+ str(set[0])+'_'+ str(set[len(set)-1])

def total_movavg_predict(data,attribute,*args):
    set=args
    dividend=0
    data['totalmovavg_predict'+ str(attribute)+ str(set[0])+'_'+ str(set[len(set)-1])]=0
    for j in range(1,len(set)+1):
        dividend+=j
    for n in range(0,len(set)):
        average_function=movavg_retro(data,attribute,set[n])
        data['totalmovavg_predict'+ str(attribute)+ str(set[0])+'_'+ str(set[len(set)-1])]+=(len(set)-n)*data[average_function]
        del data[average_function]
    data['totalmovavg_predict'+ str(attribute)+ str(set[0])+'_'+ str(set[len(set)-1])]=data['totalmovavg_predict'+ str(attribute)+ str(set[0])+'_'+ str(set[len(set)-1])]/dividend
    for i in range(0,len(data)-int(round((set[0]/2)+1))):
        data.loc[i,'totalmovavg_predict'+ str(attribute)+ str(set[0])+'_'+ str(set[len(set)-1])]=data['totalmovavg_predict'+ str(attribute)+ str(set[0])+'_'+ str(set[len(set)-1])][i+int(round((set[0]/2)+1))]
    for i in range(len(data)-int(round((set[0]/2)+1)),len(data)):
        data.loc[i,'totalmovavg_predict'+ str(attribute)+ str(set[0])+'_'+ str(set[len(set)-1])]=np.nan
    return 'totalmovavg_predict' + str(attribute)+ str(set[0])+'_'+ str(set[len(set)-1])

def clean_labels(set_copy,attribute,loopset):

    for i in range(0,len(loopset)):
        short_window=loopset[i]
        window5=15*loopset[i]
        del set_copy['totalmovavg_predict'+attribute+str(short_window)+'_'+str(window5)]

def labels(trainset_copy,trainset,attribute,loopset):

    for i in range(0,len(loopset)):
        short_window=loopset[i]
        window5=15*loopset[i]
        trainset_copy['totalmovavg_predict'+attribute+str(short_window)+'_'+str(window5)]=trainset['totalmovavg_predict'+attribute+str(short_window)+'_'+str(window5)]



def splitting_cleaning(data,attribute,ceil_attribute,loopset,test_size,test_sizes1,layers):
    data[ceil_attribute+'_category']=np.ceil((abs(data[ceil_attribute])+0.001)/0.76) #0.76
    data[ceil_attribute+'_category'].where(data[ceil_attribute+'_category']<6,6.0,inplace=True)
    print_full(data[ceil_attribute+'_category'])
    from sklearn.model_selection import StratifiedShuffleSplit
    split=StratifiedShuffleSplit(n_splits=1,test_size=test_size,random_state=42)
    for train_index, test_index in split.split(data,data[ceil_attribute+'_category']):
        strat_train_set=data.loc[train_index]
        strat_test_set=data.loc[test_index]
    for set_ in (strat_train_set,strat_test_set):
        set_.drop(ceil_attribute+'_category',axis=1,inplace=True)


    test=strat_test_set.copy()
    test_labels=pd.DataFrame()
    clean_labels(test,attribute,loopset)
    labels(test_labels,strat_test_set,attribute,loopset)

    strat_train_set_layer={}
    strat_test_set_layer={}
    train={}
    train_labels={}
    test_layer={}
    test_layer_labels={}
    i=0
    while i<(layers-1):
        if i==0:
            strat_train_set[ceil_attribute+'_category']=np.ceil((abs(strat_train_set[ceil_attribute])+0.001)/0.76) #0.76
            strat_train_set[ceil_attribute+'_category'].where(strat_train_set[ceil_attribute+'_category']<6,6.0,inplace=True)
            split=StratifiedShuffleSplit(n_splits=1,test_size=test_sizes1,random_state=42)
            for train_index, test_index in split.split(strat_train_set,strat_train_set[ceil_attribute+'_category']):

                strat_train_set_layer['{0}'.format(i)]=data.loc[train_index]
                strat_test_set_layer['{0}'.format(i)]=data.loc[test_index]
            for set_ in (strat_train_set_layer['{0}'.format(i)],strat_test_set_layer['{0}'.format(i)]):
                set_.drop(ceil_attribute+'_category',axis=1,inplace=True)
            train['{0}'.format(i)]=strat_train_set_layer['{0}'.format(i)].copy()
            clean_labels(train['{0}'.format(i)],attribute,loopset)
            train_labels['{0}'.format(i)]={}
            labels(train_labels['{0}'.format(i)],strat_train_set_layer['{0}'.format(i)],attribute,loopset)
            i+=1


        else:
            strat_test_set_layer['{0}'.format(i-1)][ceil_attribute+'_category']=np.ceil((abs(strat_test_set_layer['{0}'.format(i-1)][ceil_attribute])+0.001)/0.76)
            strat_test_set_layer['{0}'.format(i-1)][ceil_attribute+'_category'].where(strat_test_set_layer['{0}'.format(i-1)][ceil_attribute+'_category']<6,6.0,inplace=True)
            size=0.01*int(100*((layers-1)-i)/((layers-1)-i+1))
            split=StratifiedShuffleSplit(n_splits=1,test_size=size,random_state=42)
            for train_index, test_index in split.split(strat_test_set_layer['{0}'.format(i-1)],strat_test_set_layer['{0}'.format(i-1)][ceil_attribute+'_category']):
                strat_train_set_layer['{0}'.format(i)]=data.loc[train_index]
                strat_test_set_layer['{0}'.format(i)]=data.loc[test_index]
            for set_ in (strat_train_set_layer['{0}'.format(i)],strat_test_set_layer['{0}'.format(i)]):
                set_.drop(ceil_attribute+'_category',axis=1,inplace=True)

            train['{0}'.format(i)]=strat_train_set_layer['{0}'.format(i)].copy()
            clean_labels(train['{0}'.format(i)],attribute,loopset)
            train_labels['{0}'.format(i)]={}
            labels(train_labels['{0}'.format(i)],strat_train_set_layer['{0}'.format(i)],attribute,loopset)
            if i==(layers-2):
                test_layer=strat_test_set_layer['{0}'.format(i)].copy()
                clean_labels(test_layer,attribute,loopset)
                labels(test_layer_labels,strat_test_set_layer['{0}'.format(i)],attribute,loopset)
            i+=1

    ret=[test_layer,test_layer_labels,test,test_labels]
    for i in range(0,layers-1):
        ret.append(train[str(i)])
        ret.append(train_labels[str(i)])
    return ret


def blender1(train_data,train_labels,list,wind):
    dataframe1=pd.DataFrame()
    dataframe2=pd.DataFrame()
    for i in range(0,len(list)):
        name=list[i]

        from sklearn.externals import joblib

        gbrt=joblib.load('gbrt_'+name)
        ada=joblib.load('ada_'+name)

        gbrt1=gbrt.predict(train_data)
        ada1=ada.predict(train_data)

        stacking=pd.DataFrame({'gbrt':gbrt1,'ada':ada1})

        forest_final=joblib.load(name)
        predictions_final=forest_final.predict(stacking)

        train_labels1=train_labels.copy()
        train_labels1=train_labels1[name].values
        final0=pd.DataFrame(predictions_final)
        finalg=pd.DataFrame(gbrt1)
        finala=pd.DataFrame(ada1)
        final1=pd.DataFrame(train_labels1)
        final0=final0.reset_index()
        finalg=finalg.reset_index()
        finala=finala.reset_index()
        final1=final1.reset_index()
        dataframe=pd.concat([final1,final0],axis=1)
        del dataframe['index']
        dataframe=dataframe.values
        dataframe=pd.DataFrame(dataframe,columns=[name+wind,'predictions_'+name+wind])
        dataframe1=pd.concat([dataframe1,dataframe],axis=1)
        dataframe2=pd.concat([finalg,finala],axis=1)
        del dataframe2['index']
        dataframe2=dataframe2.values
        dataframe2=pd.DataFrame(dataframe2,columns=['predictions_gbrt_'+name+wind,'predictions_ada_'+name+wind])
        dataframe1=pd.concat([dataframe1,dataframe2],axis=1)


    return dataframe1

def minute_price_historical(symbol, comparison_symbol, limit, aggregate, exchange=''):
    url = 'https://min-api.cryptocompare.com/data/histominute?fsym={}&tsym={}&limit={}&aggregate={}'\
            .format(symbol.upper(), comparison_symbol.upper(), limit, aggregate)
    if exchange:
        url += '&e={}'.format(exchange)
    page = requests.get(url)
    data = page.json()['Data']
    df = pd.DataFrame(data)
    df['timestamp'] = [datetime.datetime.fromtimestamp(d) for d in df.time]
    return df




def datasets(ticker, comparison_symb, minute_number_of_data, minute_time_delta,write): #write==1 to save data

    mdf = minute_price_historical(ticker, comparison_symb, minute_number_of_data, minute_time_delta)

    m=[]
    for i in range(0,len(mdf["timestamp"])):
        m.append((str(mdf.timestamp[i])[0:4])+(str(mdf.timestamp[i])[5:7])+(str(mdf.timestamp[i])[8:10])+(str(mdf.timestamp[i])[11:13])+(str(mdf.timestamp[i])[14:16])+(str(mdf.timestamp[i])[17:19]))
    m=np.array(m)
    mdf1=pd.DataFrame({'close':mdf['close'],'high':mdf['high'],'low':mdf['low'],'open':mdf['open'],'time':mdf['time'],'volumefrom':mdf['volumefrom'],'volumeto':mdf['volumeto'],'date':m})
    if write==1:
        mtime=str(mdf.timestamp.max())
        name="minute"+ticker + "_"+mtime[:13]+ "_"+mtime[14:16]+ "_"+mtime[17:19]+"_"
        minute_file=open('folder address'+name+'.txt','w+')
        minute_file.write(mdf1.to_csv())
        minute_file.close()

    return mdf1

def data_creator(ticker,comparison_symb,minute_number_of_data,minute_time_delta):

    ticker='BTC'
    comparison_symb='EUR' #*****************['BTC', 'ETH', 'USD']
    minute_number_of_data=minute_number_of_data
    minute_time_delta = minute_time_delta # Bar width in minutes
    mdf=datasets(ticker, comparison_symb, minute_number_of_data, minute_time_delta,0)
    return mdf

def main():
    ticker='BTC' 
    comparison_symb='EUR' #['BTC', 'ETH', 'USD']
    minute_number_of_data=9999
    minute_time_delta = 1 # Bar width in minutes

    datasets(ticker,comparison_symb, minute_number_of_data, minute_time_delta,1)
    
if __name__=='__main__':
    main()
