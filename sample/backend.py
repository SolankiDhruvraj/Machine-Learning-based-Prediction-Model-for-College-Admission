
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import os
import csv

def best_college(user_rank,user_c,user_round,user_q,user_p):
    dataframe=pd.read_csv('updated1.csv')
    dataframe=dataframe.drop(['id','Unnamed: 0'],axis=1)

    category=np.unique(dataframe['category'])
    code=[]
    for i in range(len(category)):
        code.append(i+1)
    dataframe['category']=dataframe['category'].replace(category,code)
    category_name=np.array(dataframe['category'])

    df=dataframe.copy()

    del df["year"]
    del df["institute_type"]
    del df["institute_short"]
    del df["program_name"]
    del df["program_duration"]
    del df["degree_short"]
    del df["is_preparatory"]
    del df["opening_rank"]

    college=np.unique(df['College'])
    code=[]
    for i in range(len(college)):
        code.append(i+1)
    df['College']=df['College'].replace(college,code)
    college_name=np.array(df['College'])

    quota=np.unique(df['quota'])
    code=[]
    for i in range(len(quota)):
        code.append(i+1)
    df['quota']=df['quota'].replace(quota,code)
    quota_name=np.array(df['quota'])

    pool=np.unique(df['pool'])
    code=[]
    for i in range(len(pool)):
        code.append(i+1)
    df['pool']=df['pool'].replace(pool,code)
    quota_name=np.array(df['pool'])



    global user_quota
    for i in range(len(quota)):
        if user_q==quota[i]:
            user_quota=i+1
        
    global user_pool
    for i in range(len(pool)):
        if user_p==pool[i]:
            user_pool=i+1

    global user_category
    for i in range(len(category)):
        if user_c==category[i]:
            user_category=i+1

    X=df.drop(['College','category'],axis=1)
    y=pd.DataFrame()
    y['College']=df['College']
    t1=dataframe['College']
    t2=df['College']

    dfcluster=pd.DataFrame()
    def Decision_Tree(df,dataframe,user_round,user_quota,user_pool,user_rank):
        X=df.drop(['College','category'],axis=1)
        y=pd.DataFrame()
        y['College']=df['College']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=51)
        classifier = DecisionTreeClassifier(criterion='gini')
        classifier.fit(X_train,y_train)
        accuracy=classifier.score(X_test,y_test)*100
        temp=np.array(X_test)
        t1=dataframe['College']
        t2=df['College']
        arr = np.array(t1)
        arr1 = np.array(t2)
        dict1={}
        global list1
        list1=[]
        for i in range(0,len(arr)):
            dict1[arr1[i]]=arr[i]
        global arr2
        arr2 = classifier.predict([[user_round,user_quota,user_pool,user_rank]])
        for i in arr2:
            list1.append(dict1[i])
        return df

    dataframe1=dataframe[dataframe.category==1]
    dataframe2=dataframe[dataframe.category==2]
    dataframe3=dataframe[dataframe.category==3]
    dataframe4=dataframe[dataframe.category==4]
    dataframe5=dataframe[dataframe.category==5]
    dataframe6=dataframe[dataframe.category==6]
    dataframe7=dataframe[dataframe.category==7]
    dataframe8=dataframe[dataframe.category==8]
    dataframe9=dataframe[dataframe.category==9]
    dataframe10=dataframe[dataframe.category==10]

    df1=df[df.category==1]
    df2=df[df.category==2]
    df3=df[df.category==3]
    df4=df[df.category==4]
    df5=df[df.category==5]
    df6=df[df.category==6]
    df7=df[df.category==7]
    df8=df[df.category==8]
    df9=df[df.category==9]
    df10=df[df.category==10]

    if (user_category==1):
        dfcluster = Decision_Tree(df1,dataframe1,user_round,user_quota,user_pool,user_rank)
    elif (user_category==2):
        dfcluster = Decision_Tree(df2,dataframe2,user_round,user_quota,user_pool,user_rank)
    elif (user_category==3):
        dfcluster = Decision_Tree(df3,dataframe3,user_round,user_quota,user_pool,user_rank)
    elif (user_category==4):
        dfcluster = Decision_Tree(df4,dataframe4,user_round,user_quota,user_pool,user_rank)
    elif (user_category==5):
        dfcluster = Decision_Tree(df5,dataframe5,user_round,user_quota,user_pool,user_rank)
    elif (user_category==6):
        dfcluster = Decision_Tree(df6,dataframe6,user_round,user_quota,user_pool,user_rank)
    elif (user_category==7):
        dfcluster = Decision_Tree(df7,dataframe7,user_round,user_quota,user_pool,user_rank)
    elif (user_category==8):
        dfcluster = Decision_Tree(df8,dataframe8,user_round,user_quota,user_pool,user_rank)
    elif (user_category==9):
        dfcluster = Decision_Tree(df9,dataframe9,user_round,user_quota,user_pool,user_rank)
    elif (user_category==10):
        dfcluster = Decision_Tree(df10,dataframe10,user_round,user_quota,user_pool,user_rank)
    
    X=dfcluster.drop(['College'],axis=1)
    y=dfcluster['College']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=51)
    kmeans = KMeans(n_clusters=100,random_state=51).fit(X_train)

    cluster1 = kmeans.predict(X)
    dfcluster['cluster']=cluster1
    dfcluster=dfcluster.reset_index()
    dict2={}
    for i in range(len(dfcluster)):
        dict2[dfcluster.College[i]]=dfcluster.cluster[i]
    tempdf = dfcluster.loc[dfcluster['cluster']==dict2[arr2[0]]].sort_values('closing_rank')

    tempdf1=tempdf.copy()

    code=[]
    for i in range(len(quota)):
        code.append(i+1)
    tempdf1['quota']=tempdf1['quota'].replace(code,quota)
    quota_name=np.array(tempdf1['quota'])

    code=[]
    for i in range(len(pool)):
        code.append(i+1)
    tempdf1['pool']=tempdf1['pool'].replace(code,pool)
    quota_name=np.array(tempdf1['pool'])

    code=[]
    for i in range(len(college)):
        code.append(i+1)
    tempdf1['College']=tempdf1['College'].replace(code,college)
    quota_name=np.array(tempdf1['pool'])

    code=[]
    for i in range(len(category)):
        code.append(i+1)
    tempdf1['category']=tempdf1['category'].replace(code,category)
    quota_name=np.array(tempdf1['category'])

    del tempdf1['index']
    final=pd.DataFrame(tempdf1['College'])
    final.drop_duplicates(keep="first",inplace=True)
    final_1=pd.DataFrame(final.head(10))
    del final_1['Unnamed: 0']
    
    
    final_1.to_csv('final.csv')
    
    df_final1 = pd.read_csv('final.csv')
    df_final1.to_html("html_output.html")
    
    