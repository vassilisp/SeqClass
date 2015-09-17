# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 13:22:20 2015

@author: vpan
"""

from astroML.plotting import hist
import matplotlib.pyplot as plt

from stratified_metrics import savefig


def scoreHisto(y_true, y_scores, label, path, targetscore=None):
    
    if len(y_scores.shape)>=2 or len(y_true.shape)>=2:
        y_scores = y_scores.ravel()
        y_true = y_true.ravel()
    

    scores = y_scores
    labels = y_true
    positive_samples = scores[labels==1]
    negative_samples = scores[labels==0]

    if len(positive_samples)>len(negative_samples):
        palpha = 0.2
        nalpha = 0.9
    else:
        palpha = 0.9
        nalpha = 0.2
    fig, ax = plt.subplots(figsize=(9,6))
    #hist(x, 'freedman', ax=ax, color = 'r', alpha=0.9)

    count, bins,_ = hist(negative_samples, 'freedman', ax=ax, color = 'b', alpha=nalpha)
    
    #fig, ax = plt.subplots(figsize=(9,6))
    hist(positive_samples, bins=bins, ax=ax, color = 'r', alpha=palpha)

    #title
    ax.set_title('Score histogram of pos/neg samples')   
    
    bot,top = ax.get_ylim()

    if targetscore != None:
        if targetscore.__class__.__name__ == 'int':
            targetscore = [targetscore]
        for ttscore in targetscore:
            ax.plot([ttscore, ttscore],[bot,top], 'g--', lw=1.3 )
    
    if __name__ != '__main__':
        savefig(fig, path, label+'separation_hist')
    
    
    
if __name__ == "__main__":

    from sklearn.pipeline import make_pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.svm import LinearSVC    
    
    mnb = make_pipeline(TfidfVectorizer(ngram_range=(3,4)), MultinomialNB())
    

    svc01 = OneVsRestClassifier(LinearSVC(C=0.1), n_jobs=-1)

    
    psvc1 = make_pipeline(TfidfVectorizer(ngram_range=(3,4)), svc01)

    
    
    kept_estimators = {'naiveBayes':mnb, 
                       'pSVC': psvc1
                       #'rsvc': rsvc,
                       #'psvc0.1':psvc01
                       }
                       
    
    #%%
                       
    import LoadingTestData
    import rebatcher
    from sklearn.cross_validation import train_test_split    
    #X,Y = LoadingTestData.loadTestData('pro307653', 'clientId', 1)                       
    X,Y = LoadingTestData.loadTestData('pro48556', 'clientId', 1) 

    Xdiv, Ydiv,_ = rebatcher.single_rebatcher2(X,Y,150)
    
    
    x_train, x_test, y_train, y_test = train_test_split(Xdiv, Ydiv, train_size=0.7)

    mnb.fit(x_train, y_train)
    
    scores = mnb.predict_log_proba(x_test)

    from sklearn.preprocessing import LabelBinarizer
    lbin = LabelBinarizer()
    lbin.fit(y_train)
    
    y_true = lbin.transform(y_test)    
    
    scoreHisto(1-y_true, scores, 'TESTEST', 'TESTPATH',-2)    
    
    
    