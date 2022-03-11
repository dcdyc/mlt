# Find-S Algorithm

import pandas as pd 
import numpy as np
data = pd.read_csv("weather.csv")
x = np.array(data.iloc[:,0:-1])
y = np.array(data.iloc[:,-1])
print(data,'\n')

def finds(c,t) : 
    hyp = [None]*len(c[0])
    for i ,val in enumerate(c) : 
        if(t[i]=="yes") : 
            for x in range(len(hyp)) : 
                if hyp[x] == None :
                    hyp[x] = val[x]
                elif hyp[x] != val[x] :
                    hyp[x] = '?'
        print(hyp)
    return hyp


print("The final hypothesis is :",finds(x,y))