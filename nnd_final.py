
"""

Created on Sat Dec  5 10:11:58 2020

@author: zahra
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#parameters
m_min=10
data_name1 = "z0m5e11.csv"




data_name_split =data_name1.split('_')
name = data_name_split[0]
halo_finder = data_name_split[1]
redshift = data_name_split[2]
L_box =int(data_name_split[3][2:-1])
if (halo_finder=='FOF'):
    m='mass'
if (halo_finder=='Rockstar'):
    m='Mvir'

#load data
#data_name = data_name1+'.csv';
data_name = './Simulations_data/'+data_name1+'.csv';
data = pd.read_csv(data_name)
data['log_mass'] = np.log10(data[m])
data = data[data['log_mass']>m_min]
data = data[[m,'x','y','z']].values

num_den = len(data)/(L_box**3)
r_star = (3/(4*np.pi*num_den))**(1/3)

radius = 10*r_star

print('loading data is finished!')
nnd = np.full((len(data),),-1).reshape(-1,1)
mass_nnd = np.full((len(data),),-1).reshape(-1,1)
data = np.hstack((data,nnd,mass_nnd))
data_r = np.random.uniform(0,L_box,(len(data),3))
data_r = np.hstack((data_r,nnd,mass_nnd))
print('Preparing array is finished!')
for i in range(len(data)):
    condition_x = (data[:,1]<data[i,1]+radius)&(data[:,1]>data[i,1]-radius)
    condition_y= (data[:,2]<data[i,2]+radius)&(data[:,2]>data[i,2]-radius)
    condition_z= (data[:,3]<data[i,3]+radius)&(data[:,3]>data[i,3]-radius)
    
    condition_xr = (data[:,1]<data_r[i,0]+radius)&(data[:,1]>data_r[i,0]-radius)
    condition_yr= (data[:,2]<data_r[i,1]+radius)&(data[:,2]>data_r[i,1]-radius)
    condition_zr= (data[:,3]<data_r[i,2]+radius)&(data[:,3]>data_r[i,2]-radius)
    
    
    a= ((data[condition_x&condition_y&condition_z][:,1]- data[i,1])**2 +
    (data[condition_x&condition_y&condition_z][:,2]- data[i,2])**2+ 
    (data[condition_x&condition_y&condition_z][:,3]- data[i,3])**2)**(0.5)
    
    try:
        data[i,4] = np.amin(a[a!=0])
        data[i,5] = data[condition_x&condition_y&condition_z][:,0][np.argwhere(a==np.amin(a[a!=0]))[0,0]]
        
    except ValueError:  
        pass
    
    
    a_r = ((data[condition_xr&condition_yr&condition_zr][:,1]- data_r[i,0])**2 +
    (data[condition_xr&condition_yr&condition_zr][:,2]- data_r[i,1])**2+ 
    (data[condition_xr&condition_yr&condition_zr][:,3]- data_r[i,2])**2)**(0.5)
    
 
        
       
    try:
        data_r[i,3] = np.amin(a_r)
        data_r[i,4] = data[condition_xr&condition_yr&condition_zr][:,0][np.argwhere(a_r==np.amin(a_r))[0,0]]
        
    except ValueError:  
        pass
    if(np.mod(i,10**(np.floor(np.log10(len(data)))-1))==0):
        print(len(data),i,(i/len(data))*100,'% is finished!')
    
       
    
  
    #saving data
address = './'+'sim['+data_name1+']_m_min['+str(m_min)+']/';

name_nnd =  address+'NND'+'_'+name+'_'+halo_finder+'_'+redshift+'_sample'+str([m_min])+'.txt'
name_nnd_r =  address+'NND_R'+'_'+name+'_'+halo_finder+'_'+redshift+'_sample'+str([m_min])+'.txt'
column_names_nnd = ['mass','x','y','z','nnd','mass[nn]']
column_names_nnd_r= ['x[random]','y[random]','z[random]','nnd','mass[nn]']
np.savetxt(name_nnd,data,header=','.join(column_names_nnd),comments='')
np.savetxt(name_nnd_r,data_r,header=','.join(column_names_nnd_r),comments='')



'''end'''
