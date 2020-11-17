#!/usr/bin/env python
# coding: utf-8

# ## OPERATION RESEARCH
# The criteria for the pre-selection are:
# <ul>
# • Viscosity of the neat resin should be less than 1 Pa·s (1000 cps) to enable VARIM processing.
#     
# • Tg (glass transition temperature) of the neat resin should preferable be higher than 50-60°C.
# 
# • The process temperature should be lower than 230°C, so that low cost accessories can be used.
# 
# • The resin should have a long pot life.
# 
# • Cost of the resin should be affordable.
# 
# • Availability of basic knowledge and technology.
# </ul>
# 

# # Algorithm
# <ol>
# <li>Elimination Search according to constraints d</li>
# <li>Obtain the decision matrix and relative importance matrix a</li>
# <li>Normalize the decision matrix using the method from TOPSIS rij=xij/Σ(xij)^0.5</li>
# <li>Normalize the relative importance matrix using the Geometric Mean from AHP Method GMi=(πaij)^(1/n) by using AHP</li>
# <li>Eigen value calculation and multiplying decision matrix by weights to obtain normalized weighted matrix </li>
# <li>Obtain the best and worst solution from weighted normalized matrix using TOPSIS</li>
# <li>Obtain best soln using calculating the Euclidean distance and then ranking them according to RSI score.RSI=S_minus/S_plus+S_minus</li>
# </ol>
#    

# In[1]:


#Using the suitable libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go



# In[2]:


#Loading the Excel sheet containing the Decision matrix
df=pd.read_excel("C:\\Users\\SHUBHAM GUPTA.LAPTOP-TBS5E28D\\Desktop\\or.xlsx",sheet_name="Selected")
df


# In[3]:


d=df.drop(columns=df.columns[0])


# In[4]:


d


# In[5]:


#Normalizing the decision matrix
def den(i,j,d):
    sum_=0
    for i in range(0,d.shape[0]):
        sum_+=(d[i][j])**2
    return sum_
d=np.array(d)
r=np.zeros((d.shape[0],d.shape[1]))
for i in range(0,d.shape[0]):
    for j in range(0,d.shape[1]):
        r[i][j]=d[i][j]/(den(i,j,d))**0.5
        


# In[6]:


r


# In[7]:


#Loading the Excel sheet containing the realtive importance matrix
df1=pd.read_excel("C:\\Users\\SHUBHAM GUPTA.LAPTOP-TBS5E28D\\Desktop\\or.xlsx",sheet_name="Importance_matrix")
df1


# In[8]:


df1.drop(columns=['Unnamed: 0'],inplace=True)


# In[9]:


df1


# In[10]:


#To remove the bias, incorporating the AHP method for calcualtion of weights
def gm(df1):
    gmi=np.ones((df1.shape[0]))
    for i in range(0,df1.shape[0]):
        for j in range(0,df1.shape[1]):
            gmi[i]*=df1[i][j]
        gmi[i]=gmi[i]**(1/df1.shape[1])
    return gmi

df1_np=np.array(df1)
gmi=gm(df1_np)
w=np.zeros(gmi.shape[0])
for i in range(0,gmi.shape[0]):
    w[i]=gmi[i]/sum(gmi)

N2=np.transpose(w)
N3=np.dot(df1_np,N2)
N4=N3/N2
N2,N3,N4,gmi,df1_np,w


# In[11]:


#Calculating the eigen values using the relative importance matrix
eigen,righteigen=np.linalg.eig(df1_np)


# In[12]:


eigen,righteigen


# In[13]:


lambdamax=max(eigen)
lambdamax


# In[14]:


#Calculating the consistency ratio to ensure best solution. CR=CI/RI. CR<=1
#Random index values corresponding to different matrix sizes in array ri
ri=np.array([0.00,0.00,0.58,0.90,1.12,1.24,1.32,1.41,1.45,1.49])
ci=(lambdamax-df1_np.shape[0])/(df1_np.shape[0]-1)
consistency_ratio=ci/ri[df1_np.shape[0]-1]
consistency_ratio


# In[15]:


#Obtaining the weighted normalized matrix
V=r*w
V


# In[16]:


#Obtaining the best and worst system from matrix V
V_plus=np.zeros(V.shape[1])
V_minus=np.zeros(V.shape[1])
for j in range(0,V.shape[1]):
    V_plus[j]=np.max(V[:,[j]])
    V_minus[j]=np.min(V[:,[j]])
V_plus,V_minus


# In[17]:


#Caluculating the Euclidean distance from best and worst system of each entry in V. Then Ranking them according to RSI
#RSI=S_minus/S_plus+S_minus
S_plus=np.zeros(V.shape[0])
S_minus=np.zeros(V.shape[0])
RSI=np.zeros(V.shape[0])
for i in range(0,V.shape[0]):
    for j in range(0,V.shape[1]):
        S_plus[i]+=(V[i][j]-V_plus[j])**2
        S_minus[i]+=(V[i][j]-V_minus[j])**2
    S_plus[i]=S_plus[i]**0.5
    S_minus[i]=S_minus[i]**0.5
    RSI[i]=S_minus[i]/(S_plus[i]+S_minus[i])


# In[18]:


RSI


# In[19]:


#Sorting the items according to the RSI Score
sortedRSI=np.sort(RSI)[::-1]
sortedRSI_index=np.zeros(RSI.shape[0])
for i in range(0,RSI.shape[0]):
    sortedRSI_index[i]=int(np.where(RSI==sortedRSI[i])[0]+1)
sortedRSI_index=sortedRSI_index.astype(int)
RSI,sortedRSI,sortedRSI_index


# In[20]:


#Obtaining the list of items in best to worst order and obtaining the bar graph to show comparison b/w different items
names=list(df.columns)
items=list(df[names[0]])
for i in range(0,sortedRSI_index.shape[0]):
     print(items[sortedRSI_index[i]-1])
df_final = {names[0]:items, 'Score': RSI}
df_final= pd.DataFrame(data=df_final)
ax=df_final.plot(kind='bar')
ax.set_title("Overall Score",fontsize=16)
for p in ax.patches:
    ax.annotate("{:.2%}".format(p.get_height()),
                xy=(p.get_x()+0.02, p.get_height()+0.01))

ax.set_xticklabels(items)


# In[21]:


#For obtaining the spider diagram also known as the Radar plot
list_name = names
attributes= items
fig = go.Figure()

for i in range(0,V.shape[0]):
    fig.add_trace(go.Scatterpolar(
          r=V[i],
          theta=list_name,
          fill='toself',
          name=attributes[i]
    ))

fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[np.min(V),np.max(V)]
    )),
  showlegend=True
)

fig.show()


# In[ ]:




