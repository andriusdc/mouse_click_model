import pandas as pd
import numpy as np


def Euclidean_Dist(df1, df2):
    return np.linalg.norm(df1.values - df2.values,
                   axis=1)
#Função distancia segmento/ponto
def lineseg_dists(p, a, b):
    """Cartesian distance from point to line segment

    Edited to support arguments as series, from:
    https://stackoverflow.com/a/54442561/11208892

    Args:
        - p: np.array of single point, shape (2,) or 2D array, shape (x, 2)
        - a: np.array of shape (x, 2)
        - b: np.array of shape (x, 2)
    """
    # normalized tangent vectors
    d_ba = b - a
    d = np.divide(d_ba, (np.hypot(d_ba[:, 0], d_ba[:, 1])
                           .reshape(-1, 1)))
    #np.nan_to_num(d,copy=False,nan=[0,0])
    #print(np.isnan(d).sum())

    # signed parallel distance components
    # rowwise dot products of 2D vectors
    s = np.multiply(a - p, d).sum(axis=1)
    t = np.multiply(p - b, d).sum(axis=1)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, np.zeros(len(s))])

    # perpendicular distance component
    # rowwise cross products of 2D vectors  
    d_pa = p - a
    #print(c)
    c = d_pa[:, 0] * d[:, 1] - d_pa[:, 1] * d[:, 0]

    return np.hypot(h, c)
def feat_eng(train_seg):
    temp=pd.DataFrame()
    #train_seg=train_seg[train_seg.session=='session_5265929106']
    temp=pd.DataFrame((train_seg[(train_seg.Action_border=='PC')| (train_seg.Action_border=='MM') | (train_seg.Action_border=='DD')].Action_border.index))
    temp['action_count']=temp.index
    temp['index']=temp.loc[:,0]
    temp.drop(0,axis=1,inplace= True)
    temp.set_index(temp['index'],inplace=True)
    temp.columns.name=None
    train_seg=train_seg.merge(temp,how='left',left_index=True, right_index=True)
    train_seg['action_count'].fillna(method='bfill',inplace=True)
    #train_seg.drop('index',axis=1,inplace=True)

    #2nd dataframe with action features

    df_action=train_seg[train_seg.Action_border.notna()].copy(deep=True)
    df_action['index']=train_seg[train_seg.Action_border.notna()].index
    df_action['begin_act_idx']=df_action['index'].shift()
    df_action['action_count']=train_seg['action_count']
    df_action.loc[df_action.index[0],'begin_act_idx']=train_seg.index[0]
    train_seg.loc[:,'begin_act_idx']=df_action['begin_act_idx']
    ###########################################################
    #############################################################
    train_seg.loc[:,'begin_act_idx'].fillna(method='bfill',inplace=True)
    train_seg.loc[:,'end_act_idx']=df_action['index']
    train_seg.loc[:,'end_act_idx'].fillna(method='bfill',inplace=True)
    df_action.loc[df_action.index.to_list()[0],'begin_act_idx']=0


    #delta x,y, t
    train_seg['delta_x']=train_seg['x']-train_seg['x'].shift()
    train_seg['delta_y']=train_seg['y']-train_seg['y'].shift()
    train_seg.loc[train_seg.index[0],'delta_x']=0
    train_seg.loc[train_seg.index[0],'delta_y']=0
    #print(train_seg.loc[:,'begin_act_idx'].values)
    #train_seg['delta_x_begin_act']=np.sqrt((train_seg['x'].values-train_seg.loc[train_seg.loc[:,'begin_act_idx'].values,'x'].values)**2+(train_seg['y'].values-train_seg.loc[train_seg.loc[:,'begin_act_idx'].values,'y'].values)**2)
    #train_seg['delta_x_end_act']=np.sqrt((train_seg['x'].values-train_seg.loc[train_seg.loc[:,'end_act_idx'].values,'x'].values)**2+(train_seg['y'].values-train_seg.loc[train_seg.loc[:,'end_act_idx'].values,'y'].values)**2)
    #print(np.array(list(zip(train_seg.loc[train_seg.loc[:,'begin_act_idx'].values,'x'],train_seg.loc[train_seg.loc[:,'begin_act_idx'].values,'y']))))
    train_seg['deviation']=lineseg_dists(p=np.array(train_seg[['x','y']]),
                                         a=np.array(list(zip(train_seg.loc[train_seg.loc[:,'begin_act_idx'].values,'x'],train_seg.loc[train_seg.loc[:,'begin_act_idx'].values,'y']))),
                                         b=np.array(list(zip(train_seg.loc[train_seg.loc[:,'end_act_idx'].values,'x'],train_seg.loc[train_seg.loc[:,'end_act_idx'].values,'y']))))  
    
    train_seg['dist_from_act_init']=Euclidean_Dist(train_seg.loc[train_seg['begin_act_idx'],['x','y']],
    train_seg[['x','y']])
    train_seg.deviation.fillna(train_seg['dist_from_act_init'],inplace=True)
    
    
    #Tempo
    train_seg['delta_t']=(train_seg['client timestamp']-train_seg['client timestamp'].shift())
    train_seg['delta_t'].replace(0, train_seg[train_seg['delta_t']!=0].delta_t.min()/2,inplace=True)
    train_seg.loc[train_seg.index[0],'delta_t']=0
    
    #Velo
    train_seg['vel_x']=train_seg['delta_x']/train_seg['delta_t']
    train_seg.loc[train_seg.index[0],'vel_x']=0    
    train_seg['vel_x_delta']=train_seg['vel_x']-train_seg['vel_x'].shift()
    train_seg.loc[train_seg.index[0],'vel_x_delta']=0    
    #velocidades 
    train_seg['vel_y']=train_seg['delta_y']/train_seg['delta_t']
    train_seg.loc[train_seg.index[0],'vel_y']=0    
    train_seg.loc[:,'vel_y'].replace([np.inf, -np.inf], np.nan, inplace=True)
    train_seg['vel_y_delta']=train_seg['vel_y']-train_seg['vel_y'].shift()
    train_seg.loc[train_seg.index[0],'vel_y_delta']=0    
    train_seg['vel_total']=np.sqrt(np.power(train_seg['vel_x'],2)+np.power(train_seg['vel_y'],2))
    train_seg.loc[train_seg.index[0],'vel_total']=0    
    train_seg['vel_tot_delta']=train_seg['vel_total']-train_seg['vel_total'].shift()
    train_seg.loc[train_seg.index[0],'vel_tot_delta']=0
    #theta coordenada angular
    train_seg['theta']=np.arctan(train_seg['delta_y']/train_seg['delta_x'])
    train_seg['theta'].replace(np.nan,0, inplace=True)
    train_seg.loc[train_seg.index[0],'theta']=0
    train_seg['theta_delta']=train_seg['theta']-train_seg['theta'].shift()
    train_seg.loc[train_seg.index[0],'theta_delta']=0 
    #extras
    train_seg['acc']=train_seg['vel_tot_delta']/train_seg['delta_t']
    train_seg.loc[train_seg.index[0],'acc']=0 
    train_seg['acc_delta']=train_seg['acc']-train_seg['acc'].shift()
    train_seg.loc[train_seg.index[0],'acc_delta']=0  
    train_seg['jerk']=train_seg['acc_delta']/train_seg['delta_t']
    train_seg.loc[train_seg.index[0],'jerk']=0
    #velocidade angular
    train_seg['vel_angular']=train_seg['theta_delta']/train_seg['delta_t']
    train_seg.loc[train_seg.index[0],'vel_angular']=0

    #Curvature
    train_seg['event_length']=np.sqrt(np.power(train_seg['delta_x'],2)+np.power(train_seg['delta_y'],2))
    train_seg['cum_length']=train_seg['event_length'].cumsum()  

    train_seg['cum_length_delta']=train_seg['cum_length']-train_seg['cum_length'].shift()
    train_seg.loc[train_seg.index[0],'cum_length_delta']=0   

    train_seg['curvat']=train_seg['theta_delta']/train_seg['delta_t']
    train_seg.loc[train_seg.index[0],'curvat']=0
    #Action Dataframe
    #df_action=train_seg[train_seg.Action_border.notna()].copy(deep=True)
    #df_action['begin_act_idx']=[0]+df_action.index.to_list()
    df_action['v_x_mean']=df_action.merge(train_seg.groupby(by='action_count')['vel_x'].mean(),left_on=['action_count'],right_index=True,how='left')['vel_x']
    df_action['v_x_std']=df_action.merge(train_seg.groupby(by='action_count')['vel_x'].std(),left_on=['action_count'],right_index=True,how='left')['vel_x']
    df_action['v_x_max']=df_action.merge(train_seg.groupby(by='action_count')['vel_x'].max(),left_on=['action_count'],right_index=True,how='left')['vel_x']
    df_action['v_x_min']=df_action.merge(train_seg.groupby(by='action_count')['vel_x'].min(),left_on=['action_count'],right_index=True,how='left')['vel_x']

    df_action['v_y_mean']=df_action.merge(train_seg.groupby(by='action_count')['vel_y'].mean(),left_on=['action_count'],right_index=True,how='left')['vel_y']
    df_action['v_y_std']=df_action.merge(train_seg.groupby(by='action_count')['vel_y'].std(),left_on=['action_count'],right_index=True,how='left')['vel_y']
    df_action['v_y_max']=df_action.merge(train_seg.groupby(by='action_count')['vel_y'].max(),left_on=['action_count'],right_index=True,how='left')['vel_y']
    df_action['v_y_min']=df_action.merge(train_seg.groupby(by='action_count')['vel_y'].min(),left_on=['action_count'],right_index=True,how='left')['vel_y']

    df_action['v_t_mean']=df_action.merge(train_seg.groupby(by='action_count')['vel_total'].mean(),left_on=['action_count'],right_index=True,how='left')['vel_total']
    df_action['v_t_std']=df_action.merge(train_seg.groupby(by='action_count')['vel_total'].std(),left_on=['action_count'],right_index=True,how='left')['vel_total']
    df_action['v_t_max']=df_action.merge(train_seg.groupby(by='action_count')['vel_total'].max(),left_on=['action_count'],right_index=True,how='left')['vel_total']
    df_action['v_t_min']=df_action.merge(train_seg.groupby(by='action_count')['vel_total'].min(),left_on=['action_count'],right_index=True,how='left')['vel_total'] 

    df_action['acc_mean']=df_action.merge(train_seg.groupby(by='action_count')['acc'].mean(),left_on=['action_count'],right_index=True,how='left')['acc']
    df_action['acc_std']=df_action.merge(train_seg.groupby(by='action_count')['acc'].std(),left_on=['action_count'],right_index=True,how='left')['acc']
    df_action['acc_max']=df_action.merge(train_seg.groupby(by='action_count')['acc'].max(),left_on=['action_count'],right_index=True,how='left')['acc']
    df_action['acc_min']=df_action.merge(train_seg.groupby(by='action_count')['acc'].min(),left_on=['action_count'],right_index=True,how='left')['acc']

    df_action['jerk_mean']=df_action.merge(train_seg.groupby(by='action_count')['jerk'].mean(),left_on=['action_count'],right_index=True,how='left')['jerk']
    df_action['jerk_std']=df_action.merge(train_seg.groupby(by='action_count')['jerk'].std(),left_on=['action_count'],right_index=True,how='left')['jerk']
    df_action['jerk_max']=df_action.merge(train_seg.groupby(by='action_count')['jerk'].max(),left_on=['action_count'],right_index=True,how='left')['jerk']
    df_action['jerk_min']=df_action.merge(train_seg.groupby(by='action_count')['jerk'].min(),left_on=['action_count'],right_index=True,how='left')['jerk']

    df_action['w_mean']=df_action.merge(train_seg.groupby(by='action_count')['vel_angular'].mean(),left_on=['action_count'],right_index=True,how='left')['vel_angular']
    df_action['w_std']=df_action.merge(train_seg.groupby(by='action_count')['vel_angular'].std(),left_on=['action_count'],right_index=True,how='left')['vel_angular']
    df_action['w_max']=df_action.merge(train_seg.groupby(by='action_count')['vel_angular'].max(),left_on=['action_count'],right_index=True,how='left')['vel_angular']
    df_action['w_min']=df_action.merge(train_seg.groupby(by='action_count')['vel_angular'].min(),left_on=['action_count'],right_index=True,how='left')['vel_angular']

    df_action['curv_mean']=df_action.merge(train_seg.groupby(by='action_count')['curvat'].mean(),left_on=['action_count'],right_index=True,how='left')['curvat']
    df_action['curv_std']=df_action.merge(train_seg.groupby(by='action_count')['curvat'].std(),left_on=['action_count'],right_index=True,how='left')['curvat']
    df_action['curv_max']=df_action.merge(train_seg.groupby(by='action_count')['curvat'].max(),left_on=['action_count'],right_index=True,how='left')['curvat']
    df_action['curv_min']=df_action.merge(train_seg.groupby(by='action_count')['curvat'].min(),left_on=['action_count'],right_index=True,how='left')['curvat']

    df_action['action_type']=train_seg[train_seg.Action_border.notna()]['Action_border']
    df_action['elapsed_time']=train_seg[train_seg.Action_border.notna()]['delta_action']
    df_action['s_action']=train_seg[train_seg.Action_border.notna()]['cum_length']
    df_action['dist_end_end']=Euclidean_Dist(train_seg.loc[train_seg['begin_act_idx'].unique(),['x','y']],
    train_seg.loc[train_seg['end_act_idx'].unique(),['x','y']])
    
    vector_i_f=pd.DataFrame()
    vector_i_f=(train_seg.loc[train_seg['end_act_idx'].unique(),['x','y']].reset_index()-train_seg.loc[train_seg['begin_act_idx'].unique(),['x','y']].reset_index())
    vector_i_f['norm']=np.sqrt(vector_i_f['x']**2+vector_i_f['y']**2)

    def get_angle(data):
       if data['x']>=0  and data['y']>=0:
          return np.arccos(np.divide(data['x'],data['norm']))
       if data['x']<0  and data['y']>=0:
          return np.arccos(np.divide(data['x'],data['norm']))
       if data['x']<0  and data['y']<0:
          return 2*np.pi-np.arccos(np.divide(data['x'],data['norm']))
       if data['x']>=0 and data['y']<0:
          return 2*np.pi-np.arccos(np.divide(data['x'],data['norm']))


    def angle_to_direction(angle):
       if angle <= np.radians(45):
          return 1
       elif angle <= np.radians(90):
          return 2
       elif angle <= np.radians(135):
          return 3
       elif angle <= np.radians(180):
          return 4
       elif angle <= np.radians(225):
          return 5
       elif angle <= np.radians(270):
          return 6
       elif angle <= np.radians(315):
          return 7
       else :
          return 8
    
    
  
    
    
    #usar merge no ID
    df_action['direction']=(vector_i_f.apply(get_angle,axis=1).fillna(0)).apply(angle_to_direction).values
    df_action['straightness']=df_action['dist_end_end']/df_action['s_action']
    df_action['num_points']=train_seg[train_seg.Action_border.notna()]['index']-train_seg[train_seg.Action_border.notna()]['index'].shift().fillna(train_seg[train_seg.Action_border.notna()]['index'].values[0])
    df_action['sum_angle']= df_action.merge(train_seg.groupby(by='action_count')['theta'].sum(),left_on=['action_count'],right_index=True,how='left')['theta']
    
    
    
    df_action['largest_deviation']=df_action.merge(train_seg.groupby(by='action_count')['deviation'].max(),left_on=['action_count'],right_index=True,how='left')['deviation']
     
    df_action['sharp angles']=df_action.merge(train_seg.groupby(by='action_count')['theta'].apply(lambda x: (x<=0.0005).sum()),left_on=['action_count'],right_index=True,how='left')['theta']
    
    #from inertia aceleration time
    df_action.drop(['Unnamed: 0', 'record timestamp', 'client timestamp', 'button', 'state',
       'x', 'y','Action', 'action_begin', 'delta_action',
       'Action_border', 'begin_act_idx'],inplace=True,axis=1)
    return df_action,train_seg

#obs:colocar essa função em novo notebook
#df_action,train_seg=feat_eng(train_seg)