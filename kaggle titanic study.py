#!/usr/bin/env python
# coding: utf-8

# # 캐글 타이타닉

# In[1]:


import pandas as pd
#판다스 이용하기

train=pd.read_csv('train.csv') #train 파일 불러오기

train.head() #train 파일의 윗 5줄 보기


# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


# In[4]:


sns.relplot(data=train,x='Age',y='Survived')


# In[5]:


sns.relplot(data=train,x='Age',y='Survived',hue='Sex',aspect=3)


# In[6]:


sns.catplot(data=train, x='Pclass', y='Fare',aspect=4)
#차이를 직접 확인해봐라?

sns.catplot(data=train,x='Pclass',y='Fare',jitter=False, aspect=4)
sns.catplot(data=train,x='Pclass',y='Fare',aspect=4,hue='Sex')


# In[8]:


sns.catplot(data=train,x='Pclass',y='Fare',kind='box')
sns.catplot(data=train,x='Pclass',y='Fare',kind='box',hue='Sex')


# In[9]:


sns.pairplot(data=train,hue='Sex',
            x_vars=['Pclass','Age','Fare'],
            y_vars=['Survived','Age','Fare'],height=3)


# In[10]:


cnt=train[train['Sex']=='male'].shape[0]
print(cnt)


# In[12]:


cnt=train[train['Pclass']==1].shape[0]
print(cnt)


# In[5]:


import pandas as pd
train=pd.read_csv('train.csv')
train


# In[6]:


train=pd.read_csv('train.csv')
train

#사망자 수 세기
cnt=train[train['Survived']==0].shape[0]
print(cnt)

#생존자 수 세기
cnt=train[train['Survived']==1].shape[0]
print(cnt)

#matplotlib에서 bar chart를 그리는 명령어는? 실행후 plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Summary Statistics
## 사망자,생존자 수 
train_sum_by_survived = train.groupby('Survived').Survived.sum()

label = ['Survived','Death']
index = np.arange(len(label))

plt.bar(index, train_sum_by_survived)
plt.title('The number of Survived or Death', fontsize=20)
plt.xlabel('Survived', fontsize=18)
plt.ylabel('The number', fontsize=18)
plt.xticks(index, label, fontsize=15)
plt.show()



#2단계) 범례, 축 이름, 막대 색 설정할 때 쓰는 구문은?


# In[ ]:




