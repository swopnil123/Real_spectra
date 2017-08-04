# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 12:58:41 2016

@author: HP
"""

import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy import integrate
from matplotlib import style 

style.use('ggplot')

from numba import jit 

"""This code reads earthquake ground motion data from folder and generates
elastic resposne spectra and computes the Median sa and Stdlog(sa) for 
the ensemble
"""
 
class real_spectra():
    
    def __init__(self,path): #path = inputfolder;      
        self.path = path 
        self.w = np.logspace(-1,2,100)
        
#method to get the path of the files stored in the directory    
    def file(self): 
        for filename in os.listdir(self.path):
            infilename = os.path.join(self.path,filename)
            if not os.path.isfile(infilename):
                continue 
            yield infilename 
            
#method to generate the list of files used in the analysis
    def records(self): 
        files = os.listdir(self.path)
        fid = [file.split('.')[0] for file in files]
        #df = pd.DataFrame(fid,columns = ['Records'])
        return fid
        
#method to extract timehistory values from the respective files        
    def data(self): 
        for file in self.file():
            fid = open(file,'r')
            for line in fid:
                text = line.strip()
                command = text.split(',')
                if len(command) > 1: 
                    if 'DT=' in command[1]:
                        # retrieve the value of constant time increment
                        dt = float('.'+command[1].split('.')[1].split(' ')[0])
                        break
            fid.close()
            df = pd.read_table(file,sep = '  ', skiprows = 4, header = None,\
                    engine = 'python') #python engine makes the warning disappear
            a = np.array(df)
            a = a.flatten(order = 'C')
            a = a[~np.isnan(a)] #removes the Nan objects from the array
            ag = np.array([0.])
            ag = np.append(ag,a)            
            t = np.ndarray(len(ag))
            t[0] = 0
            for i in range(1,len(t)):
                t[i] = t[i-1]+dt 
            yield t,ag
    
    @jit    
    def response_Newmark(self,wf,t,ag): # average acceleration method 
        Y, B = 1/2, 1/4
        u_t = np.ndarray(len(ag), dtype = 'float')    
        v_t = np.ndarray(len(ag), dtype = 'float')
        a_t = np.ndarray(len(ag), dtype = 'float')
        dt = t[1] - t[0]
        m = 1
        zi = 0.05
        w = 2*np.pi*wf
        k = w**2 * m 
        c = 2*m*w*zi
        #initial values
        u_t[0], v_t[0] = 0., 0.
        a_t[0] = (m*ag[0] - c*v_t[0] - k * u_t[0])/m
        k_ = k + Y * c/(B * dt) + m/(B*dt**2)
        a = m/(B*dt) + Y/B *c
        b = m/(2*B) + dt*(Y/(2*B) - 1) * c
    
    # main calculations start here 
        for i in range(1,len(ag)):
            delta_p = m * (ag[i]-ag[i-1]) + a * v_t[i-1] + b * a_t[i-1]
            delta_u = delta_p/k_
            delta_v = Y/(B*dt)*delta_u - Y/B*v_t[i-1] + dt * (1-Y/(2*B)) * a_t[i-1]
            delta_a = 1/(B*dt**2) * delta_u - v_t[i-1] / (B*dt) - a_t[i-1]/(2*B)
            u_t[i] = u_t[i-1] + delta_u            
            v_t[i] = v_t[i-1] + delta_v
            a_t[i] = a_t[i-1] + delta_a   
        return max(abs(u_t))*w**2      
        
    def psa(self): #generate the pseudo spectral acceleration 
        #vectorize the function to better and disregard the last files
        #w = np.logspace(-1,2,100)
        vfunc = np.vectorize(self.response_Newmark,excluded = [1,2]) 
        for t,ag in self.data():
            yield vfunc(self.w,t,ag)
            
# method to determine the strong motion duration of a record 
            
    def strong_motion(self,t,ag):        
        Ia = np.pi/(2*9.81)*integrate.trapz(ag**2,t)
        Ia_cum = np.pi/(2*9.81)*integrate.cumtrapz(ag**2,t,initial = 0)
        t1 = t[np.where(np.logical_and(Ia_cum/Ia>=0.05,Ia_cum/Ia<=0.95))][0]
        t2 = t[np.where(np.logical_and(Ia_cum/Ia>=0.05,Ia_cum/Ia<=0.95))][-1]
        Tsm = t2-t1
        return Tsm
    
# generate the list of Tsm for the dataset 
    def tsm(self):        
        for t,ag in self.data():
            yield self.strong_motion(t,ag)
                        
        
# method to create a database of response spectra 
    @jit        
    def database(self): 
        df = pd.DataFrame(self.w)
        for sa in self.psa():
            df = pd.concat([df,pd.DataFrame(sa)],axis = 1, ignore_index = True)
        
        for i in range(1,len(self.records())+1):
            df.rename(columns = {i: self.records()[i-1]},inplace = True)
        #initialize the mean,standard deviation, mean_std and mean-std     
        lnmean = np.mean(np.log(df.values[:,1:]),axis = 1)
        median = np.exp(lnmean)        
        lnstd = np.std(np.log(df.values[:,1:]),axis = 1)           
        mean_up = np.exp(lnmean+lnstd)
        mean_down = np.exp(lnmean-lnstd)        
        #insert the mean and standard deviation inside the Dataframe     
        dic = {'LnStd':lnstd,'Mean-Std':mean_down,'Mean+Std':mean_up,'Median':median}
        for keys, values in zip(dic.keys(),dic.values()):
            df.insert(1,keys,values)
        return df 
            
    def write_to_file(self,output): #output = output folder
        outputfile = 'RealSpectra.csv'        
        outputpath = os.path.join(output,outputfile)
        df = self.database()        
        df.to_csv(outputpath)
    
    def plotting(self):
        df = self.database()
        fig = plt.figure('Response Spectrum of Real Records')
        ax = fig.add_subplot(111)
        ax.loglog(self.w,df.values[:,5:])
        ax.loglog(self.w,df['Median'],linewidth = 4, label = "$\mu$" + " spectrum")
        ax.loglog(self.w,df['Mean+Std'],'--',linewidth = 3, label = \
            "$\mu$"+ " + " + "$\sigma$ "+'spectrum')
        ax.loglog(self.w,df['Mean-Std'],'--',linewidth = 3, label = \
            "$\mu$"+ " - " + "$\sigma$ "+'spectrum')
        ax.set_xlabel('Frequency(Hz)',fontsize = 13)
        ax.set_ylabel('Psa(g)', fontsize = 13)
        ax.set_title('Response Spectra Ensemble',fontsize = 15)  
        ax.grid(which = 'major')
        ax.grid(which = 'minor')
        ax.legend(loc = 4)        
        plt.show()
        plt.grid()
       
def main():       
    inputfolder = 'F:\Books\Thesis\PEERNGARecords_Unscaled\H2test'
    #outputfolder = 'F:\Books\Thesis\python scripts'        
    a = real_spectra(inputfolder)
    #return np.array(list(a.tsm()))    
    a.plotting()
    #a.write_to_file(outputfolder)
    
def time_historyplot(index):
    from scipy import integrate 
    b = real_spectra('F:\Books\Thesis\PEERNGARecords_Unscaled\H2test')
    values = list(b.data())
    recordnames = list(b.records())
    fig = plt.figure('Sample Time-history of recorded data')
    ax = fig.add_subplot(111)
    t = values[index][0]
    a = values[index][1]
    v = integrate.cumtrapz(a*9.81*1000,t,dx = None, initial = 0)
    ax.plot(t,v)
    ax.set_xlabel('Time(sec)',fontsize = 13)
    ax.set_ylabel('Acceleration(g)', fontsize = 13)
    ax.set_title('Timehistory of ' + str(recordnames[index]),fontsize = 15)  
    plt.show()
    
    
    
    
    
if __name__== '__main__':
    main()
   