# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 13:21:49 2021

@author: Yuting
"""
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

def plot_precision_recall_vs_threshold(precisions,recalls,thresholds):
    plt.plot(thresholds,precisions[:-1],"b--",label="Precision")
    plt.plot(thresholds,recalls[:-1],"g-",label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0,1])
    
    
def positive_pred_ratio(th, y_score):
    return np.array([(y_score >= x).sum() / len(y_score) for x in th])
    
def calculate_roc(ytrain, strain, ytest, stest):
    '''
    ---------------
        ytrain: train label
        strain: train score
        ytest: test label
        stest:test score
    --------------

    '''
    res_dict = {}
    fpr, tpr, th = metrics.roc_curve(ytrain, strain)
    fpr_, tpr_, th_ = metrics.roc_curve(ytest, stest)
    ppr = positive_pred_ratio(th, strain)
    ppr_ = positive_pred_ratio(th_, stest)
    freq_train = pd.Series(np.r_[0., np.diff(ppr)], index=pd.cut(fpr, np.linspace(0, 1, 21))).groupby(lambda x:x.mid).sum()
    freq_test = pd.Series(np.r_[0., np.diff(ppr_)], index=pd.cut(fpr_, np.linspace(0, 1, 21))).groupby(lambda x:x.mid).sum()
    freq = pd.DataFrame({'train':freq_train, 'test':freq_test})
    res_dict['train'] =  {'fpr': fpr, 'tpr': tpr, 'ppr': ppr, 'th': th, 'strain': strain}
    res_dict['test'] = {'fpr': fpr_, 'tpr': tpr_, 'ppr': ppr_, 'th': th_, 'stest': stest}
    res_dict['freq'] = freq
    return res_dict
    
def plot_roc_auc2(train_fpr,train_tpr,test_fpr,test_tpr):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(train_fpr, train_tpr, label='Train AUC: ' + str(auc_score_train))
    ax.plot(test_fpr, test_tpr, label='Test AUC: ' + str(auc_score_test))
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.legend(fontsize=12)
    plt.show()    
    
    
    
def calculate_roc_auc(model_, X_test_, y_test_):

    # Get the model predictions
    prediction_test_=model_.predict_proba(X_test_)[:,1]
    
    # Compute roc-auc
    fpr, tpr, thresholds=metrics.roc_curve(y_test_, prediction_test_)
    
    # Print the evaluation metrics as pandas dataframe
    score=pd.DataFrame({"ROC-AUC" : [metrics.auc(fpr, tpr)]})
    return fpr, tpr, score



def plot_roc_auc(fpr,tpr):
    
    # Initialize plot
    f, ax=plt.subplots(figsize=(14,8))
    
    # Plot ROC
    roc_auc=metrics.auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=2, alpha=0.3,
            label="AUC = %0.2f"% (roc_auc))
    
    # Plot the random line.
    plt.plot([0, 1], [0, 1], linestyle='--', lw=3, color='r',
             label="Random", alpha=.8)
    
    # Fine tune and show the plot.
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title("ROC-AUC")
    ax.legend(loc="lower right")
    plt.show()







    