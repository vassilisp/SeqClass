# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 16:44:47 2015

@author: vpan
"""
import LoadingTestData
import rebatcher
from pipelineC_T_SGD import EVALUATE_TEST
from sklearn.decomposition import TruncatedSVD, PCA, RandomizedPCA
import time
from masquerfold import MasquerFold

import matplotlib.pyplot as plt

import numpy as np

def identify(Ydiv):    
    dic = {}
    ident = np.zeros(len(Ydiv))
    cnt=0
    for i, user in enumerate(Ydiv):
        if user not in dic:
            dic.update({user:cnt})
            cnt+=1
            
        ident[i] = (dic[user])
    return ident, dic
    
def plot4(X, Ydiv=None, filename='', save=False):


    if Ydiv == None:
        print('provide either ident array or Ydiv')
        return -1
    else:
        ident, dic = identify(Ydiv)
    
    fig,ax = plt.subplots(2,2)
    ax[0,1].scatter(XX[:,0], XX[:,1], c = ident/len(dic), cmap=plt.cm.hsv)
    ax[0,1].set_xlabel('C1')
    ax[0,1].set_ylabel('C2')
    ax[1,0].scatter(XX[:,0], XX[:,2], c = ident/len(dic), cmap=plt.cm.hsv)
    ax[1,0].set_xlabel('C1')
    ax[1,0].set_ylabel('C3')
    #ax[1,1].scatter(XX[:,1], XX[:,2], c = ident)
    ax[1,1].scatter(XX[:,1], XX[:,2], c = ident/len(dic), cmap=plt.cm.hsv)
    ax[1,1].set_xlabel('C2')
    ax[1,1].set_ylabel('C3')
    
    
    for u in np.unique(Y):
       
       mask = Ydiv == u
    
       axx = ax[0,1]
       axx.text(XX[mask,0].mean(), XX[mask,1].mean(),u, ha='center', va='center', 
                bbox=dict(alpha=0.5, edgecolor='none',facecolor=plt.cm.hsv(dic[u]/len(dic))))
    
       axx = ax[1,0]
       axx.text(XX[mask,0].mean(), XX[mask,2].mean(),u, ha='center', va='center', 
                bbox=dict(alpha=0.5, edgecolor='none',facecolor=plt.cm.hsv(dic[u]/len(dic))))
    
       axx = ax[1,1]
       axx.text(XX[mask,1].mean(), XX[mask,2].mean(),u, ha='center', va='center', 
                bbox=dict(alpha=0.5,edgecolor='none', facecolor=plt.cm.hsv(dic[u]/len(dic))))
    
    ax[0,0].axis('off')
    from mpl_toolkits.mplot3d import Axes3D
    #fig = plt.figure( figsize=(20,20))
    ax = fig.add_subplot(221, projection='3d')
    
    xs = XX[:,0]
    ys = XX[:,1]
    zs = XX[:,2]

    ax.scatter(xs, ys, zs, c=ident/len(dic),cmap=plt.cm.hsv)
    
    ax.set_xlabel('C1')
    ax.set_ylabel('C2')
    ax.set_zlabel('C3')
    
    for u in Y:
        
        ax.text3D(XX[Ydiv == u, 0].mean(),
                  XX[Ydiv == u, 1].mean(),
                  XX[Ydiv == u, 2].mean(), u,
                  horizontalalignment='center',
                  bbox=dict(alpha=.5, edgecolor='none', facecolor=plt.cm.hsv(dic[u]/len(dic))))
    
    plt.tight_layout()
    fig.tight_layout()
    if save:
        if filename =='':
            filename = '4_3Dplot.svg'
        plt.savefig(filename, format='svg', dpi=600)


def plot3D(X, Ydiv=None, filename='', save=False):
    

    if Ydiv == None:
        print('provide Ydiv')
        return -1
    else:
        ident, dic = identify(Ydiv)
            
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure( figsize=(20,20))
    ax = fig.add_subplot(111, projection='3d')
    
    xs = XX[:,0]
    ys = XX[:,1]
    zs = XX[:,2]

    ax.scatter(xs, ys, zs, c=ident/len(dic),cmap=plt.cm.hsv)
    
    ax.set_xlabel('C1')
    ax.set_ylabel('C2')
    ax.set_zlabel('C3')
    
    for u in Y:
        
        ax.text3D(XX[Ydiv == u, 0].mean(),
                  XX[Ydiv == u, 1].mean(),
                  XX[Ydiv == u, 2].mean(), u,
                  horizontalalignment='center',
                  bbox=dict(alpha=.5, edgecolor='w', facecolor=plt.cm.hsv(dic[u]/len(dic))))
    
    plt.tight_layout()
    if save:
        if filename =='':
            filename = '3Dplot.svg'
        plt.savefig(filename, format='svg', dpi=600)

def findNexplain(decomp_method, thresh=0.9):
    cum = 0
    for i, comp in enumerate(decomp_method.explained_variance_ratio_):
        cum += comp
        if cum>thresh:
            break
    print(i, 'explains:', decomp_method.explained_variance_ratio_[:i].sum())
    return i

def findNincrement(decomp_method, thresh=0.0001):
    cum = 0.00000001
    for i, comp in enumerate(decomp_method.explained_variance_ratio_):
        incr = (cum+comp)/cum
        cum = cum+comp
        if incr<1 + thresh:
            break
    print(i, 'explains:', decomp_method.explained_variance_ratio_[:i].sum())
    return i
    
def findNdiff(decomp_method, thresh=0.001):
    for i, comp in enumerate(decomp_method.explained_variance_ratio_):
        if comp<thresh:
            break
    print('value:', decomp_method.explained_variance_ratio_[i])
    print(i, 'explains:', decomp_method.explained_variance_ratio_[:i].sum())
    return i

from sklearn.feature_extraction.text import TfidfVectorizer

#%%
scale = False
DR = 'svd'
nmax=3
nmin=3

proID = 'pro48937'
tokens = 2
X, Y = LoadingTestData.loadTestData(proID, 'clientId',tokens)

#%%
div = 200
Xdiv, Ydiv, rep = rebatcher.single_rebatcher(X,Y, div)

#%%
#Xdiv1, Ydiv1, rep = rebatcher.single_rebatcher2(X,Y, div)
#Xdiv2, Ydiv2, rep = rebatcher.single_rebatcher2(X,Y, div, acc=False)
#%%
#Xdiv3, Ydiv3, rep = rebatcher.single_rebatcher2(X,Y, div, acc=True)
#Xdiv4, Ydiv4, rep = rebatcher.single_rebatcher2(X,Y, 25, acc=True, min_div=150)
Xdiv5, Ydiv5, inter, rep = rebatcher.single_rebatcher2(X,Y, 25, acc=True, min_div=150, max_div=300, get_inter=True)

#%%
Xdiv = Xdiv5
Ydiv = Ydiv5
#%%
ngrams = (nmin,nmax)

tfidf = TfidfVectorizer(ngram_range=ngrams, sublinear_tf=True)

t0 = time.time()
X_fv = tfidf.fit_transform(Xdiv)
print(time.time()-t0)
print(X_fv.shape)

#%%

if scale==True:
    #scale to -1, 1
    from sklearn.preprocessing import StandardScaler
    stdS = StandardScaler()
    X_s = stdS.fit_transform(X_fv.toarray())

#%%
X_s = X_fv
#%%
"""

pca = PCA()
t1 = time.time()
pca.fit(X_fv.toarray())
t1 = time.time() - t1
print(t1)
n_comp = findNexplain(pca)
"""
#%%

if DR == 'rpca':
    rpca = RandomizedPCA(n_components=500)
    t1 = time.time()
    rpca.fit(X_fv.toarray())
    t1 = time.time() - t1
    print(t1)
    n_comp = findNexplain(rpca)

    XX = rpca.transform(X_fv.toarray())

else:
    svd = TruncatedSVD(n_components=1000)
    t2 = time.time()
    svd.fit(X_s)
    t2 = time.time() - t2
    print(t2)
    n_comp2 = findNexplain(svd)
    
    XX = svd.transform(X_fv)[:,:n_comp2]
    XX = svd.transform(X_fv)
#%%
#%%
from sklearn.manifold import MDS
from sklearn.manifold import Isomap
iso = Isomap(n_neighbors=len(np.unique(Y)), n_components=50)
t3 = time.time()
iso.fit(X_s.toarray())
t3 = time.time()-t3
print(t3)


#%%
res_iso = iso.transform(X_s.toarray())
XX= res_iso
#%%

mds = MDS(n_components=3, n_jobs=-1)
t4 = time.time()
res_mds = mds.fit_transform(X_s.toarray())
t4 = time.time()-t4
print(t4)
#%%
XX = res_mds


#%%
from sklearn.manifold import TSNE

tsne = TSNE(n_components=3)
t5 = time.time()
res_tsne = tsne.fit_transform(X_s)
t5 = time.time()-t5
print(t5)
#%%
XX = res_tsne
#%%

#%%    

#%%
'''
h = 10
plt.scatter(*svd.components_[h:h+2])
'''
#%%

plt.figure()
ident,dic = identify(Ydiv)
plt.scatter(XX[:,0], XX[:,1], c=ident, cmap=plt.cm.hsv)
for u in np.unique(Y):
    mask = Ydiv==u
    plt.text(XX[mask,0].mean(), XX[mask,1].mean(),u, ha='center', va='center', 
            bbox=dict(alpha=0.5, edgecolor='none',facecolor=plt.cm.hsv(dic[u]/len(dic))))
#%%
'''
for i in range(1000):
    yy = np.ones((len(XX[i])))*i
    xx = XX[i]
    cca = [int(ident[i])]*3
    plt.scatter(xx[:3],yy[:3], c=cca, cmap=plt.cm.hsv)
'''
#%%
plot4(XX,Ydiv,save=False)

#%%
plot3D(XX,Ydiv,save=False)

#%%
plt.figure(figsize=(10,10))
plt.scatter(XX[:,10], XX[:,111], c=ident, cmap= plt.cm.rainbow)


#%%
from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
import numpy as np

cv = StratifiedKFold(Ydiv, n_folds=4)
#%%
cv = MasquerFold(Y, inter, n_folds=4)
#%%
c = np.arange(-2,2,0.1 )
xx = np.ones(len(c))*2
xx = np.power(xx,c)
xx= [1,2]
params = {'C': xx}
gridS = GridSearchCV(LinearSVC(), n_jobs=-1, refit=False,
                                param_grid= params, cv=cv, scoring = 'accuracy',
                                verbose = 10)



#%%
XX =X_s
#%%
gridS.fit(XX,Ydiv)

#%%
#DO A SECOND VALIDATION WITH TRUNCATEDSVD IN THE PIPELINE AND COMPARE WITH MULTINOMIAL NAIVE BAYES
#%%
from sklearn.base import clone
best_params = gridS.best_params_
estim = gridS.estimator
bestim = clone(estim)
bestim.set_params(**best_params)

best_estimators = {}
best_estimators.update({'opt_linearSVC': bestim, 'linearSVC': LinearSVC()})



#%%
import Globals
method = 'T' + str(tokens) + '-' + str(div) +'-SVD-' +proID
exeTime = time.strftime('%d%m_%H%M')
path = Globals.getResultsPATH()

path = Globals.getProcessIDPath(method, exeTime)
#%%

EVALUATE_TEST(XX,Ydiv, best_estimators, path+str(div)+'/', method + '_' + str(div))     


#%%
import rebatcher
Xdiv2, Ydiv2, rep = rebatcher.single_rebatcher2(X,Y, 25, acc=True, min_div=150)
