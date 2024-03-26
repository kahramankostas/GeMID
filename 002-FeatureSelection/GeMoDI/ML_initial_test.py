
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import pickle
import time
from tqdm import tqdm
import sklearn
import numpy as np
from tabulate import tabulate


from sklearn.tree import ExtraTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import OneClassSVM
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier    
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import  ComplementNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import MultinomialNB  
from sklearn.neighbors import NearestCentroid
from sklearn.svm import NuSVC
from sklearn.linear_model import Perceptron
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier

ml_list={"ExtraTreeClassifier":ExtraTreeClassifier(),
    "DecisionTreeClassifier":DecisionTreeClassifier(),
    #"OneClassSVM":OneClassSVM(),
    #"MLPClassifier":MLPClassifier(),
    "ComplementNB":ComplementNB(),
    "DummyClassifier":DummyClassifier(),         
    "RadiusNeighborsClassifier":RadiusNeighborsClassifier(),
    "KNeighborsClassifier":KNeighborsClassifier(),
    "ClassifierChain":ClassifierChain(base_estimator=DecisionTreeClassifier()),
    "MultiOutputClassifier":MultiOutputClassifier(estimator=DecisionTreeClassifier()),
    "OutputCodeClassifier":OutputCodeClassifier(estimator=DecisionTreeClassifier()),
    "OneVsOneClassifier":OneVsOneClassifier(estimator=DecisionTreeClassifier()),
    "OneVsRestClassifier":OneVsRestClassifier(estimator=DecisionTreeClassifier()),
    #"SGDClassifier":SGDClassifier(),
    "RidgeClassifierCV":RidgeClassifierCV(),
    "RidgeClassifier":RidgeClassifier(),
    "PassiveAggressiveClassifier    ":PassiveAggressiveClassifier    (),
    #"GaussianProcessClassifier":GaussianProcessClassifier(),
    "AdaBoostClassifier":AdaBoostClassifier(),
    #"GradientBoostingClassifier":GradientBoostingClassifier(),
    "BaggingClassifier":BaggingClassifier(),
    "ExtraTreesClassifier":ExtraTreesClassifier(),
    "RandomForestClassifier":RandomForestClassifier(),
    "BernoulliNB":BernoulliNB(),
    #"CalibratedClassifierCV":CalibratedClassifierCV(),
    "GaussianNB":GaussianNB(),
    "LabelPropagation":LabelPropagation(),
    "LabelSpreading":LabelSpreading(),
    "LinearDiscriminantAnalysis":LinearDiscriminantAnalysis(),
    #"LinearSVC":LinearSVC(),
    "LogisticRegression":LogisticRegression(),
    #"LogisticRegressionCV":LogisticRegressionCV(),
    "MultinomialNB  ":MultinomialNB  (),
    "NearestCentroid":NearestCentroid(),
    "NuSVC":NuSVC(),
    "Perceptron":Perceptron(),
    "QuadraticDiscriminantAnalysis":QuadraticDiscriminantAnalysis(),
    #"SVC":SVC(),
    #"HistGradientBoostingClassifier":HistGradientBoostingClassifier(),
    "CategoricalNB" : CategoricalNB()}
    


def ML(GA_input_features,train,vali,test,fold,ml_list=ml_list):
    
    
    df = pd.read_csv(train,usecols=GA_input_features)#,header=None )
    
    X_train =df[df.columns[0:-1]]
    #X_train=np.array(X_train)
    df[df.columns[-1]] = df[df.columns[-1]].astype('category')
    y_train=df[df.columns[-1]].cat.codes  
    
    df = pd.read_csv(vali,usecols=GA_input_features)#,header=None )
    X_vali =df[df.columns[0:-1]]
    #X_test=np.array(X_test)
    df[df.columns[-1]] = df[df.columns[-1]].astype('category')
    y_vali=df[df.columns[-1]].cat.codes  
    
    df = pd.read_csv(test,usecols=GA_input_features)#,header=None )
    X_test =df[df.columns[0:-1]]
    #X_test=np.array(X_test)
    df[df.columns[-1]] = df[df.columns[-1]].astype('category')
    y_test=df[df.columns[-1]].cat.codes  
    
    
    
 
    
    # In[6]:
    
    
    line=[["ML","Vali-ACC","Vali-F1","Vali-STD","Test-ACC","Test-F1","Test-STD","Total-TIME"]]

    flag=1
    for ii in tqdm(ml_list):
        second=time.time()
        logmodel = ml_list[ii]
        v_acc=[]
        t_acc=[]
    
        v_f1=[]
        t_f1=[]
        #print("Accuracy = "+ str(accuracy_score(y_test,predictions)))
        try:
            for i in range(fold):
                logmodel.fit(X_train,y_train)
                pred_v = logmodel.predict(X_vali)
                pred_t = logmodel.predict(X_test)
                v_f1.append(sklearn.metrics.f1_score(y_vali,pred_v,average= "macro"))
                t_f1.append(sklearn.metrics.f1_score(y_test,pred_t,average= "macro"))
                v_acc.append(sklearn.metrics.accuracy_score(y_vali,pred_v))
                t_acc.append(sklearn.metrics.accuracy_score(y_test,pred_t))
    
            result=[ii,round(np.mean(v_acc),4),round(np.mean(v_f1),4),round(np.std(v_f1),4),round(np.mean(t_acc),4),round(np.mean(t_f1),4),round(np.std(t_f1),4),round((time.time()-second),4)]
            line.append(result)
            #print ('%-30s %-30s %-30s %-30s' % (ii[30:],np.mean(v_results),np.mean(t_results),time.time()-second))
        except:
            #if flag:
                #print ('%-30s %-30s %-30s' % ("ML","F1 Score","TIME"))
                #flag=0
            #print ('%-30s %-30s %-30s' % (ii[:30],"ERROR",time.time()-second))
            #result=[ii,"ERROR","ERROR","ERROR","ERROR","ERROR","ERROR",round((time.time()-second),4)]
            #line.append(result)
            pass
    
    
    # In[8]:
    
    
    df = pd.DataFrame (line[1:], columns = line[0])
    df=df.sort_values(by=['Vali-F1'], ascending=False)
    
    print (tabulate(df, headers=list(df.columns)))
    return df
    

