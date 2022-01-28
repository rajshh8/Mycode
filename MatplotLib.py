#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt


# In[2]:


plt.plot([3,5,9])


# In[3]:


plt.plot([1,4,6,9],[10,20,30,15])


# In[4]:


import numpy as np


# In[5]:


x=np.arange(1,10)
x


# In[6]:


y=x**2
y


# In[7]:


plt.plot(x,y)
plt.title("Squares")
plt.xlabel("X-values")
plt.ylabel("Y-values")
plt.show()


# In[8]:


plt.plot(x,x,"r--")#for staright line through out the graph
plt.title("Linear graph")
plt.show()


# In[9]:


plt.plot(x,y,"gv")
plt.title("Squares")
plt.xlabel("X-values")
plt.ylabel("Y-values")
plt.show()


# In[10]:


plt.plot(x,y,"rx")
plt.title("Squares")
plt.xlabel("X-values")
plt.ylabel("Y-values")
plt.show()


# In[11]:


plt.plot(x,y,"go")
plt.title("Squares")
plt.xlabel("X-values")
plt.ylabel("Y-values")
plt.show()


# In[12]:


plt.plot(x,y,"g--")
plt.title("Squares")
plt.xlabel("X-values")
plt.ylabel("Y-values")
plt.xlim(-5,14)
plt.ylim(-20,100)
plt.show()


# In[13]:


plt.plot(x,y,"g",label="y=xˆ2")
plt.plot(x,x**3,"--",label="y=xˆ3")
plt.title("Squares and cube")
plt.xlabel("X-values")
plt.ylabel("Y-values")
plt.xlim(-1,30)
plt.ylim(0,1000)
plt.legend()
plt.show()


# In[14]:


plt.plot(x,y+100,"g",label="y=xˆ2")
plt.plot(x,x**2.5,"--",label="y=xˆ3")
plt.title("Squares and cube")
plt.xlabel("X-values")
plt.ylabel("Y-values")
plt.xlim(-1,30)
plt.ylim(0,300)
plt.legend()
plt.text(8,140,"(8,150)")
plt.show()


# In[15]:


plt.plot(x,y*2,"g",label="y=xˆ2")
plt.plot(y,x+20,"--",label="y=xˆ3")
plt.title("Squars and cube")
plt.xlabel("X-values")
plt.ylabel("Y-values")
plt.xlim(-1,30)
plt.ylim(0,200)
plt.legend()
plt.grid()

plt.show()


# In[16]:


x=np.arange(0.0001,100)
ln=np.log(x)
plt.plot(ln)
plt.title("Logarithmic graph")
plt.xlabel("X-axis")
plt.ylabel('Y-axis')
plt.show()


# In[17]:


import math
x=np.arange(0.0001,100)
x=list(map(lambda x:math.log(x),x))


# In[18]:


import math
x=np.arange(0.0001,100)
x=list(map(math.log,x))


# ## Vectorization

# In[19]:


np_log=np.vectorize(math.log)
x=np.arange(0.0001,100)


# In[20]:


plt.plot(np_log(x))
plt.show()


# # Bar Graph

# In[21]:


plt.bar([1,2,3],[1,4,2],width=0.4,color="g",label="Smoke")
plt.bar([1,2,3],[3,5,7],width=0.4,label="Death")
plt.title("Bar Graph")
plt.xlabel("X-value")
plt.ylabel("Y-value")
plt.legend()
plt.show()


# In[22]:


plt.bar([1,2,3],[1,4,2],width=0.4,color="g",label="Smoke")
plt.bar([1.2,2.2,3.2],[3,5,7],width=0.4,label="Death")
plt.title("Bar Graph")
plt.xlabel("X-value")
plt.ylabel("Y-value")
plt.legend()
plt.show()    #they overlapped so add 0.2


# In[23]:


plt.bar([1,2,3],[1,4,2],width=0.4,color="g",label="Smoke")
plt.bar(np.array([1,2,3])+0.4,[3,5,7],width=0.4,label="Death")
plt.title("Bar Graph")
plt.xlabel("X-value")
plt.ylabel("Y-value")
plt.legend()
plt.show()


# # Histogram ,,,,Frequency distribution

# In[24]:


val=[1,2,4,7,8,5,4,3,9,7,6,3,6,8,6,4,3,]
count,bins,patches=plt.hist(val)
plt.show()


# In[25]:


print(count)


# In[26]:


len(bins)


# In[27]:


bins


# In[28]:


val=np.random.rand(1000)


# In[29]:


plt.hist(val)
plt.show()


# ## Normal distribution 

# In[30]:


mu=100
sg=15
val=np.random.normal(mu,sg,size=1000000)


# In[31]:


plt.hist(val,bins=100)
plt.show()


# Most of the values are close to 100, mean=100
# 70% of the data is present under 1 standard deviatiom (100-15, 100+15)
# 98% of the data is present under 2 standard deviations (70, 130)
# 99.9% data lies under 3 std, (55, 145)

# # Scatter graph

# In[32]:


x=np.random.rand(50)
y=np.random.rand(50)
x2=np.random.rand(90)
y2=np.random.rand(90)


# In[33]:


plt.scatter(x,y)
plt.scatter(x2,y2)
plt.show()


# # Pie Chart

# In[34]:


plt.pie([1,3,4,6,3],labels=["Education","Travel","Health","Agriculture","Defence"],
       startangle=0,
       explode=[1,0,0,0,0],
        shadow=True)
        

plt.show()


# ## Sigmoid function

# In[35]:


# sg=1/1+eˆ-x

# x=(-100,100)


# In[36]:


lst=np.arange(-100,100)
x=list(map(lambda x:1/(1+ math.exp(-x)),lst))


# In[37]:


plt.plot(x)
plt.show()


# In[38]:


def sigmoid(x):
    return 1/(1+math.exp(-x))
sig=np.vectorize(sigmoid)

x=np.arange(-100,100)
y=sig(x)


# In[39]:


plt.plot(x,y)
plt.xlim(-15,15)
plt.ylim(-1,1.5)
plt.show()


# # Subplot

# In[40]:


x=np.arange(0,20,0.1)
sin=np.sin(x)
cos=np.cos(x)


# In[41]:


plt.plot(x,sin) 

plt.show()


# In[42]:


x=np.arange(0,20,0.1)
sin=np.sin(x)
cos=np.cos(x)
tan=np.tan(x)

plt.figure()

plt.subplot(2,3,1)
plt.plot(x,sin)

plt.subplot(2,3,3)
plt.plot(x,cos)

plt.subplot(2,3,5)
plt.plot(x,tan)

plt.show()


# In[43]:


fig = plt.figure()
ax231 = fig.add_subplot(231)
ax231.plot(x, sin)
ax233 = fig.add_subplot(233)
ax233.plot(x, tan)
ax235 = fig.add_subplot(235)
ax235.plot(x, cos)
ax231.set_title("TITLE 1")
plt.show()


# In[44]:


import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
np.random.seed(0) 
df = pd.DataFrame(data={'1':np.random.randint(0, 100, 10), '2':np.random.randint(0, 100, 10)})
plt.bar(df.index.values, df['1'])
plt.bar(df.index.values,df['2'],bottom=df['1'])

plt.legend(['1', '2'])
plt.show()


# # Another method of making graphs

# In[45]:


fig=plt.figure()
ax231=fig.add_subplot(231)
ax231.plot(x,sin)
ax235=fig.add_subplot(235)
ax235.plot(x,cos)
ax233=fig.add_subplot(233)
ax233.plot(x,tan)
plt.show()


# In[46]:


a=np.array([0,1,2])
b=np.array([0,1,2,3])


# In[47]:


a,b=np.meshgrid(a,b)


# In[48]:


a


# In[49]:


b


# In[58]:


c=np.array([[7]])
c


# In[62]:


fig=plt.figure()
ax=fig.add_subplot(projection="3d")
ax.plot_surface(a,b,c)
plt.show()


# In[54]:


type(fig)


# In[56]:


ax


# In[67]:


c=a**2+b**2


# In[73]:


x=np.arange(-1,1,0.0005)
y=x
x,y=np.meshgrid(x,y)
z=x**2 + y**2
fig=plt.figure()
ax=fig.add_subplot(projection="3d")
ax.plot_surface(x,y,z)
plt.show()


# In[ ]:




