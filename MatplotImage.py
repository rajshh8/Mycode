#!/usr/bin/env python
# coding: utf-8

# In[71]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[72]:


im=plt.imread('resized_nearest.jpeg')
plt.imshow(im)


# In[73]:


horz=np.zeros((20,355,3))
vert=np.zeros((420,20,3))


# In[74]:


mid=np.hstack([vert,im,vert])
final=np.vstack([horz,mid,horz])


# In[75]:


top.shape


# In[76]:


plt.imshow(final.astype("uint"))


# In[77]:


final=final.reshape(163300, 3)


# In[78]:


df=pd.DataFrame(final,columns=["c1","c2","c3"])


# In[79]:


df.to_csv("catOutput.csv",index=False)


# In[ ]:





# In[ ]:




