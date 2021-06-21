import numpy as np
import pandas as pd

def action_row(row):
    if row['state']=='Released' and row['state_lag']=='Drag':
        return 'DD'
    elif row['state']=='Released' and row['state_lag']=='Pressed':
        return 'PC'
    
def delta_maker(data):
    
    data.loc[:,'action_begin']=data[(data['Action']=='DD') | (data['Action']=='PC') | (data['Action']=='MM')]['client timestamp'].shift()
    data.loc[data.index[0],'action_begin']=data['client timestamp'].values[0]
    data.loc[data[(data['Action']=='DD') | (data['Action']=='PC') | (data['Action']=='MM')]['client timestamp'].index[0],'action_begin']=data['client timestamp'].iloc[0]
    #data[(data['Action']=='DD') | (data['Action']=='PC')].reset_index(drop=True).loc[0,'action_begin']=data['client timestamp'].loc[0]
    data.loc[:,'delta_action']=data[(data['Action']=='DD') | (data['Action']=='PC') | (data['Action']=='MM')]['client timestamp']-data['action_begin']
    #data.loc[train_seg.index[0],'delta_action']=data.loc[train_seg.index[0],'client timestamp']
def post_seg_delta_maker(data):
    
    data.loc[:,'action_begin']=data[(data['Action_border']=='DD') | (data['Action_border']=='PC') | (data['Action_border']=='MM')]['client timestamp'].shift()
    data.loc[data[(data['Action_border']=='DD') | (data['Action_border']=='PC') | (data['Action_border']=='MM')]['client timestamp'].index[0],'action_begin']=data['client timestamp'].iloc[0]
    #data[(data['Action']=='DD') | (data['Action']=='PC')].reset_index(drop=True).loc[0,'action_begin']=data['client timestamp'].loc[0]
    data.loc[:,'delta_action']=data[(data['Action_border']=='DD') | (data['Action_border']=='PC') | (data['Action_border']=='MM')]['client timestamp']-data['action_begin']
    
def closest_timestamp(row,data,min_events):
    temp=(data['client timestamp']-row['action_corrector']).abs().sort_values().index[0]
    
    #Considerar minimo de eventos
    if row.name-temp > min_events :
        data.loc[temp,'Action']='MM'
        data.loc[row.name,'act_correc_index']=temp
def action_mm_finder(data,thresh,min_events):    
    # 1- Calcular time -resto(delta/10)
    data['action_corrector']=data['client timestamp']-data['delta_action']%thresh
    
    #-2 Função para para procurar índice da linha mais proxima do valor
    data[data['delta_action']>=thresh].apply(closest_timestamp,args=(data,min_events),axis=1)

def correct_small_actions(row):
    
    if row.Action_border!=None and row.Action_border_lag!=None :

        return row.Action_border_lag
    else:
        return row.Action_border

def correct_small_actions_2(row):
    
    if row.Action_border!=None and row.Action_border_antilag!=None and pd.notna(row.Action_border_antilag): 
        return None
    else:
        return row.Action_border

    
def segmentation(data2,thresh,min_events):
    data=data2.copy(deep=True)
    data['state_lag']=data['state'].shift()
    data.loc[:,'Action']=data.apply(action_row,axis=1)
    delta_maker(data)
    action_mm_finder(data,thresh,min_events)
    if data.Action.iloc[[-1]].item() != 'PC' and data.Action.iloc[[-1]].item() != 'DD':
        data.loc[data.index[-1],'Action']='MM'
    data.loc[data.index[-1],'Action_border']=data.Action.iloc[[-1]].values[0]
    data['Action_border']=data['Action']
    data['Action_border_lag']=data['Action_border'].shift()
    data['Action_border']=data.apply(correct_small_actions,axis=1)
    data['Action_border_antilag']=data['Action_border'].shift(-1)
    data['Action_border']=data.apply(correct_small_actions_2,axis=1)
    data['Action']=data.Action.fillna(method='bfill')
    
    data.drop(['Action_border_lag','Action_border_antilag'],axis=1,inplace=True)
    
    return data


#Função para tratar train_seg:
def segmentation_prep(data):
    data=data.drop(['state_lag','action_corrector','act_correc_index'],axis=1)
    post_seg_delta_maker(data)
    return data


