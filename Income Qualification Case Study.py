#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
from scipy.stats import chi2

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix,f1_score,accuracy_score,f1_score,classification_report
from sklearn.model_selection import KFold,cross_val_score


# In[2]:


train_data = pd.read_csv(r"C:\Users\Prabhjyot Kaur\Downloads\Case Study\train.csv")
test_data = pd.read_csv(r"C:\Users\Prabhjyot Kaur\Downloads\Case Study\test.csv")


# In[3]:


test_data.head()


# In[4]:


test_data.head()


# ### Data understanding

# In[5]:


pd.set_option('display.max_rows',None)


# In[6]:


#type of data.
train_data.info()


# In[7]:


train_data.isna().sum()


# In[8]:


train_data.dtypes


# In[9]:


list(train_data.columns.values)


# In[10]:


# we can see the datatypes in the train data are int64, float64 and object

# finding missing values in each datatype
na_counts = train_data.select_dtypes('int64').isnull().sum()
na_counts[na_counts > 0]


# In[11]:


na_counts = train_data.select_dtypes('object').isnull().sum()
na_counts[na_counts > 0]


# In[12]:


na_counts = train_data.select_dtypes('float64').isna().sum()
na_counts[na_counts>0]


# In[13]:


# we have 5 columns of float64 datatype with missing/null values


# ## Data Cleaning

# In[14]:


# Fixing columns with mixed values
# We can correct the variables using a mapping and convert to floats
# “yes” = 1 and “no” = 0


# In[15]:


mapping={'yes':1,'no':0}

for df in [train_data, test_data]:
    df['dependency'] =df['dependency'].replace(mapping)
    df['dependency'] = pd.to_numeric(df['dependency'])

    df['edjefe'] =df['edjefe'].replace(mapping)
    df['edjefe'] = pd.to_numeric(df['edjefe'])

    df['edjefa'] =df['edjefa'].replace(mapping)
    df['edjefa'] = pd.to_numeric(df['edjefa'])


# In[16]:


train_data.head(5)


# In[17]:


# According to the documentation for these columns with missing values:

#v2a1 (total nulls: 6860) : Monthly rent payment
#v18q1 (total nulls: 7342) : number of tablets household owns
#rez_esc (total nulls: 7928) : Years behind in school
#meaneduc (total nulls: 5) : average years of education for adults (18+)
#SQBmeaned (total nulls: 5) : square of the mean years of education of adults (>=18) in the household 142


# # 

# In[18]:


# handling missing values for column "v2a1" with highest number of missing values


# In[19]:


# Columns related to  Monthly rent payment :

# tipovivi1, =1 own and fully paid house
# tipovivi2, "=1 own,  paying in installments"
# tipovivi3, =1 rented
# tipovivi4, =1 precarious 
# tipovivi5, "=1 other(assigned,  borrowed)"


# In[20]:


data = train_data[train_data['v2a1'].isnull()].head() 


# In[21]:


columns=['tipovivi1','tipovivi2','tipovivi3','tipovivi4','tipovivi5']
data[columns]


# In[22]:


# Variables indicating home ownership
own_variables = [x for x in train_data if x.startswith('tipo')]


# In[23]:


# Visualisation of the home ownership variables for home missing rent payments
train_data.loc[train_data['v2a1'].isnull(), own_variables].sum().plot.bar(figsize = (8, 6),color = 'pink',edgecolor = 'k', linewidth = 3);

plt.xticks([0, 1, 2, 3, 4], ['Owns and Paid Off', 'Owns and Paying', 'Rented', 'Precarious', 'Other'], rotation = 22)

plt.title('Home Ownership Status for Households Missing Rent Payments', size = 15);


# In[24]:


#Looking at the above data we can assume that when the house is fully paid, there will be no monthly rent payment
#Adding for all the null values.
for df in [train_data, test_data]:
    df['v2a1'].fillna(value=0, inplace=True)


# In[25]:


train_data[['v2a1']].isnull().sum()


# # 

# In[26]:


# handling missing values for column "v18q1 "
# v18q1 (total nulls: 7342) : number of tablets household owns
# Columns related to "v18q1" = "v18q"


# In[27]:


# heads of households
head = train_data.loc[train_data['parentesco1'] == 1].copy()
head.groupby('v18q')['v18q1'].apply(lambda x: x.isnull().sum())


# In[28]:


# visualising "v18q1" column
plt.figure(figsize = (8, 6))
col='v18q1'
train_data[col].value_counts().sort_index().plot.bar(color = 'pink',edgecolor = 'k', linewidth = 2)
plt.xlabel(f'{col}'); plt.title(f'{col} Value Counts'); plt.ylabel('Count')
plt.show();


# In[29]:


# adding 0 for all the null values
for df in [train_data, test_data]:
    df['v18q1'].fillna(value=0, inplace=True)


# In[30]:


train_data[['v18q1']].isnull().sum()


# # 

# In[31]:


# handling missing values for "rez_esc" column
# rez_esc (total nulls: 7928) : Years behind in school


# In[32]:


for df in [train_data, test_data]:
    df['rez_esc'].fillna(value=0, inplace=True)


# In[33]:


train_data[['rez_esc']].isnull().sum()


# # 

# In[34]:


# handling missing values for "meaneduc" column
# meaneduc (total nulls: 5) : average years of education for adults (18+)

# Columns related to average years of education for adults (18+):
# 1. edjefe
# 2. edjefa
# 3. instlevel1
# 4. instlevel2


# In[35]:


data = train_data[train_data['meaneduc'].isnull()].head()


# In[36]:


columns=['edjefe','edjefa','instlevel1','instlevel2']
data[columns][data[columns]['instlevel1']>0].describe()


# In[37]:


#from the above data we find that meaneduc is null when no level of education is 0
for df in [train_data, test_data]:
    df['meaneduc'].fillna(value=0, inplace=True)


# In[38]:


train_data[['meaneduc']].isnull().sum()


# # 

# In[39]:


# handling missing values for "SQBmeaned" column
# SQBmeaned (total nulls: 5) : square of the mean years of education of adults (>=18) in the household 142

# Columns related to "SQBmeaned" :
# 1.edjefe
# 2.edjefa
# 3.instlevel1
# 4.instlevel2


# In[40]:


data = train_data[train_data['SQBmeaned'].isnull()].head()


# In[41]:


columns=['edjefe','edjefa','instlevel1','instlevel2']
data[columns][data[columns]['instlevel1']>0].describe()


# In[42]:


#from the above data we can see that SQBmeaned is null when no level of education is 0
for df in [train_data, test_data]:
    df['SQBmeaned'].fillna(value=0, inplace=True)


# In[43]:


train_data[['SQBmeaned']].isnull().sum()


# ### 

# ### Checking if there are any biases in our dataset

# In[44]:


# test 1
contingency_tab=pd.crosstab(train_data['tipovivi3'],train_data['v2a1'])
Observed_Values=contingency_tab.values
b=scipy.stats.chi2_contingency(contingency_tab)
Expected_Values = b[3]
no_of_rows=len(contingency_tab.iloc[0:2,0])
no_of_columns=len(contingency_tab.iloc[0,0:2])
df=(no_of_rows-1)*(no_of_columns-1)
print("Degree of Freedom:-",df)

chi_square=sum([(o-e)**2./e for o,e in zip(Observed_Values,Expected_Values)])
chi_square_statistic=chi_square[0]+chi_square[1]
print("chi-square statistic:-",chi_square_statistic)

alpha=0.05
critical_value=chi2.ppf(q=1-alpha,df=df)
print('critical_value:',critical_value)

p_value=1-chi2.cdf(x=chi_square_statistic,df=df)
print('p-value:',p_value)

print('Significance level: ',alpha)

print('Degree of Freedom: ',df)

print('chi-square statistic:',chi_square_statistic)

print('critical_value:',critical_value)

print('p-value:',p_value)

print('\n')
if chi_square_statistic>=critical_value:
    print("Reject H0,There is a relationship between 2 categorical variables")
else:
    print("Retain H0,There is no relationship between 2 categorical variables")
    
if p_value<=alpha:
    print("Reject H0,There is a relationship between 2 categorical variables")
else:
    print("Retain H0,There is no relationship between 2 categorical variables")


# ### Check if there is a house without a family head.

# In[45]:


train_data["parentesco1"].value_counts()


# In[46]:


pd.crosstab(train_data['edjefa'],train_data['edjefe'])


# In[47]:


# Interpretation : Above cross tab shows 0 male head and 0 female head which implies that there 
# are 435 families with no family head.


# ### Set poverty level of the members and the head of the house within a family
#  

# In[48]:


Poverty_level = train_data[train_data['v2a1'] !=0]
Poverty_level.shape


# In[49]:


p_level = Poverty_level.groupby('area1')['v2a1'].apply(np.median)
p_level


# In[50]:


def poverty(x):
    if x<8000:
        return('Below poverty level')
    elif x>140000:
        return('Above poverty level')
    elif x<140000:
        return('Below poverty level: Urban ; Above poverty level : Rural ')


# ### Checking whether all members of the house have the same poverty level

# In[51]:


tab = Poverty_level['v2a1'].apply(poverty)


# In[52]:


pd.crosstab(tab ,Poverty_level['area1'])


# In[53]:


# Interpretation :

#There are total 1242 people above poverty level independent of area
#Remaining 1111 people level depends on their area


# ### Count how many null values are existing in columns

# In[54]:


train_data.isna().sum().value_counts()


# In[55]:


train_data['Target'].isna().sum()


# In[56]:


# conclusion - There are zero null values in the Target variable and the columns


# ### Predict the accuracy using random forest classifier

# In[57]:


# delete 'Id', 'idhogar'
cols=['Id','idhogar']
for df in [train_data, test_data]:
    df.drop(columns = cols,inplace=True)


# In[58]:


train_data.head(5)


# In[59]:


train_data.iloc[:,0:-1]


# In[60]:


train_data.iloc[:,-1]


# In[61]:


x_features = train_data.iloc[:,0:-1] # feature without target
y_features = train_data.iloc[:,-1] # only target
print("x_features: ", x_features.shape)
print("y_features: ", y_features.shape)


# In[62]:


x_train,x_test,y_train,y_test = train_test_split(x_features,y_features,test_size=0.2,random_state=1)
rmclassifier = RandomForestClassifier()


# In[63]:


rmclassifier.fit(x_train,y_train)


# In[64]:


y_predict = rmclassifier.predict(x_test)


# In[65]:


print(accuracy_score(y_test,y_predict))


# In[66]:


print(confusion_matrix(y_test,y_predict))


# In[67]:


print(classification_report(y_test,y_predict))


# In[68]:


y_predict_test = rmclassifier.predict(test_data)


# In[69]:


y_predict_test


# ### Check the accuracy using random forest with cross validation. ¶

# In[70]:


seed = 7
kfold = KFold(n_splits=5,random_state=seed,shuffle=True)


# In[71]:


rmclassifier = RandomForestClassifier(random_state=10,n_jobs = -1)
print(cross_val_score(rmclassifier,x_features,y_features,cv=kfold,scoring='accuracy'))


# In[72]:


results = cross_val_score(rmclassifier,x_features,y_features,cv=kfold,scoring='accuracy')
print(results.mean()*100)


# In[73]:


num_trees= 100

rmclassifier = RandomForestClassifier(n_estimators=100, random_state=10,n_jobs = -1)
print(cross_val_score(rmclassifier,x_features,y_features,cv=kfold,scoring='accuracy'))


# In[74]:



results = cross_val_score(rmclassifier,x_features,y_features,cv=kfold,scoring='accuracy')
print(results.mean()*100)


# In[75]:


rmclassifier.fit(x_features,y_features)
labels = list(x_features)
feature_importances = pd.DataFrame({'feature': labels, 'importance': rmclassifier.feature_importances_})
feature_importances=feature_importances[feature_importances.importance>0.015]
feature_importances.head()


# In[76]:


y_predict_test = rmclassifier.predict(test_data)
y_predict_test


# In[77]:


feature_importances.sort_values(by=['importance'], ascending=True, inplace=True)
feature_importances['positive'] = feature_importances['importance'] > 0
feature_importances.set_index('feature',inplace=True)
feature_importances.head()

feature_importances.importance.plot(kind='barh', figsize=(14, 6),color = feature_importances.positive.map({True: 'blue', False: 'red'}))
plt.xlabel('Importance')


# In[78]:


# From the above figure, meaneduc,dependency,overcrowding has significant influence on the model.


# In[ ]:




