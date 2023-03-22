# -*- coding: utf-8 -*-
"""
Created on Tue May  3 19:07:17 2022

@author: ALEXIS
"""
import os 
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import Data_driven_model

Data_driven_model.main()
#Verificar el directorio actual
directory= os.getcwd()
print(directory)

#Cambiar al directorio de trabajo
##INGRESAR SU CARPETA!!
os.chdir(r"D:\KU LEUVEN\Courses\Integrated Project\Conceptual")
#os.chdir(r"D:\KU LEUVEN\Courses\Integrated Project\WEAP_Inputs\Results")
#Uncertainty analysis
#Read csv file with time series of observed and model discharges (data)
pro_id = 'Jubones - Ecuador' # example

file_name="JubonesM_result_VAL.csv"
#lamda= float(input("Please insert value of lamda between 0 and 1: "))
lamda=0.20   #change lamba value

if file_name[-3:]=='csv':
    # Check decimal in .csv file (comma or dot)
    # Check headings
    data= pd.read_csv(file_name, decimal='.',index_col=0,na_values='-')
    
else: print("format incorrect")

if 0>lamda or lamda>1:
    print("value of lamda incorrect")

data[1:2]=data[1:2].astype(float)
#Calculate accumulatives flow
def cumulative(x):
    return data.cumsum(x)

def residual2(x):
    return (x)**2
#BC For observed and simulate  data
def BCO(x):
    return ((x**lamda)-1)/lamda


data['Obscumulative']=data['Obsflow_[m3/s]'].cumsum()
data['Simcumulative']=data['Simflow_[m3/s]'].cumsum()
data['residual']=data['Simflow_[m3/s]'] - data['Obsflow_[m3/s]']
data['residuals2']=data['residual'].apply(residual2)
data['BC_Obs']=data['Obsflow_[m3/s]'].apply(BCO)
data['BC_Sim']=data['Simflow_[m3/s]'].apply(BCO)
data['BCResidual']=data['BC_Sim'] - data['BC_Obs']
data['BIASres']= data['Simflow_[m3/s]']-data['Obsflow_[m3/s]']
print (data)

n=len(data)
pbias=data['BIASres'].sum()
pbias1=data['Obsflow_[m3/s]'].sum()
##STATISTICS CALCULATIONS
ME= data['residual'].mean() #Mean errors  [m3/s]
MSE=data['residuals2'].mean()  ##Mean squared errors [m3/s]
Var=data['residual'].var()   #variance of errors [m3/s]^2
Std=math.sqrt(Var)   #Standard deviation of Errors  [m3/s]
VarObs=data['Obsflow_[m3/s]'].var()  #Variance of observed flow [m3/s]^2
StdBC=data['BCResidual'].std()  #Standar deviation of the Box -Cox residuals Sigma
RMSE=math.sqrt(MSE)
EF=1-(MSE/VarObs)  #[-] Efficiency of the model

PBIAS=100*(1/pbias1)*pbias
print("MSE value is:" , MSE)
print("RMSE value is:" , RMSE)
print("Efficiency of the model is: ",EF)
print("PBIAS of the model is: ", PBIAS)

##Interval of confidence:
    #BC INTERVALS
#for 66% interval, sigma is 1*stdBC

data['BC+sigma']=data['BC_Sim'] + StdBC
data['BC-sigma']=data['BC_Sim'] - StdBC

#for 95% interval, sigma is 2*stdBC
data['BC+2sigma']=data['BC_Sim'] + 2*StdBC
data['BC-2sigma']=data['BC_Sim'] - 2*StdBC

##Non BC NTERVALS ("Normal intervals")
#Taking the inverse of BC
def inverseBC(x):
    return ((lamda*x)+1)**(1/lamda)
#for 66% interval
data['Sim+Sigma']=data['BC+sigma'].apply(inverseBC)
data['Sim-Sigma']=data['BC-sigma'].apply(inverseBC)
#for 95% interval
data['Sim+2Sigma']=data['BC+2sigma'].apply(inverseBC)
data['Sim-2Sigma']=data['BC-2sigma'].apply(inverseBC)

#%%PLOT OF RESULTS
##Cumulative flow
style.use('ggplot')
plt.title("Cummulative-Rippl diagram")
plt.plot(data['Obscumulative'],color='red', linewidth=0.5, label='Obscumulative')
plt.plot(data['Simcumulative'],color='blue', linewidth=0.5, linestyle='--',label='simcumulative')
plt.xticks([0,n/2,3*n/4,n])  #tick spacing, change according to size of dataframe
plt.xlabel('Time')
plt.ylabel('Q (m3/s)')
plt.legend()
plt.show()

#Heteroscedasticity
style.use('ggplot')
data.plot.scatter(x='Obsflow_[m3/s]',y='Simflow_[m3/s]',s=5) #s is for the size
x = y = plt.xlim()
plt.plot(x, y, linestyle='--', color='b', lw=2, scalex=False, scaley=False, label='Bisector')
plt.legend()
plt.title("Heteroscedasticity")
plt.show()
plt.clf()

##Homoscedaasticity Box - Cox plot
style.use('ggplot')
x=data.BC_Obs
y=data.BC_Sim
plt.scatter(x,y, s=5)
trend=np.polyfit(x,y,1)
poly1d_fn=np.poly1d(trend)
x = y = plt.xlim()
plt.plot(x, y, linestyle='--', color='b', lw=2, scalex=False, scaley=False, label='Bisector')
#plt.plot(x,poly1d_fn(x),'--',  color='m', lw=3,label='y=%.6fx+%.6f'% (trend[0],trend[1]))
plt.plot(x,poly1d_fn(x),'--',  color='m', lw=3,label='mean deviation')
plt.xlabel('Qobs (m3/s)')
plt.ylabel('Qsim (m3/s)')
plt.title("Homoscedasticity Box - Cox")
plt.legend()
plt.show()
plt.clf()

##Time series with 95% interval +2sigma
style.use('ggplot')
plt.title(pro_id)
plt.plot(data['Sim+2Sigma'],color='red', linewidth=0.25, linestyle='--', label='Upperlimit')
plt.plot(data['Sim-2Sigma'],color='red', linewidth=0.25, linestyle='--',label='Lowerlimit')
plt.plot(data['Obsflow_[m3/s]'], color='blue', linewidth=1, label='Obsflow')
plt.plot(data['Simflow_[m3/s]'],color='#FFA500', linewidth=1,label='Simflow')
plt.xticks([0,n/4,n/2,3*n/4,n])   #tick spacing, change according to size of dataframe
plt.xlabel('Time')
plt.ylabel('Q (m3/s)')
plt.legend()
plt.show()

##Time series scale
style.use('ggplot')
plt.title(pro_id)
plt.plot(data['Sim+2Sigma'],color='red', linewidth=0.25, linestyle='--', label='Upperlimit')
plt.plot(data['Sim-2Sigma'],color='red', linewidth=0.25, linestyle='--',label='Lowerlimit')
plt.plot(data['Obsflow_[m3/s]'], color='blue', linewidth=1, label='Obsflow')
plt.plot(data['Simflow_[m3/s]'],color='#FFA500', linewidth=1,label='Simflow')
plt.xticks([0,n/4,n/2,3*n/4,n])   #tick spacing, change according to size of dataframe
plt.xlim([500,1000])
plt.xlabel('Time')
plt.ylabel('Q (m3/s)')
plt.legend()
plt.show()

