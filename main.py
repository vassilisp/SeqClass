import time

import DBconnection
import Globals

import LoadingTestData
from statisticsGen import StatisticsGen

import rebatcher
from reporter import Reporter
from basicQueries import BasicQueries
from sklearn.cross_validation import train_test_split

import numpy as np

from pipelineC_T_SGD import getPipelines, EVALUATE_TEST, EVALUATE_TOPSCORES, savepickle, savejson, reportSCORES, findclf_name

from sklearn.base import clone
exeTime = time.strftime('%d%m_%H%M')
con = DBconnection.getConnection()

A = DBconnection.connectAndFetchArray(BasicQueries.getAllProcessIDs())
A = A[:,0]
A = [ 'pro48937']#'pro48556',#'pro307653'


#%% Setup env
generate_processID_stats = False

reporter = Reporter()
labels = []
results_estimators = []
results_topScores = []

cnt = 0
#%%
dividers = {100:'batch100', 200:'batch200'}#0:'full', 100:'batch100' , 200:'batch200', 300:'batch300', 400:'batch400'} 
def main():

    tokenList = {'2':2, '4':4} #{'1':1, '4':4}  #{'full':0, '1':1, '2':2, '3':3, '4':4}  ###REMEBMER TO TEST USE TARGET IF REFERER EXISTS RULE
    q_dividers = ['clientId'] #'clientId,subSession' -- if added - add also a FOR LOOP
    
    
    for proID in A:
            
        if __name__ == "__main__":
            
            
            reporter.new_report("REPORT OF PROCESS: " + proID)
            
            #create and get ProcessIDPath
            proID_path = Globals.getProcessIDPath(proID, exeTime)
    
            if(generate_processID_stats == True):
                #processID general statistics
                statGen = StatisticsGen(proID,'clientId',0 , exeTime)        
                statGen.transPerSequence();
                statGen.totalTransPerDay();      
                statGen.sequencesPerUser();
                statGen.totalTransPerUser();
                reporter.concat_report(statGen.getReport())
            
            for tok_desc, token in tokenList.items():
                X, Y = LoadingTestData.loadTestData(proID, q_dividers[0], token)
                
                token_path = proID_path + 'token' + str(token) + '/'
                
                #TODO graph number of unique transitions that change with each different token
                runTestwithDividers(X,Y,proID, proID_path, token_path, token)
            
            reporter.saveReport(proID_path, str(proID) + '_FULL_report.txt')
            
            
def runTestwithDividers(X,Y, proID, proID_path, token_path, token):
        
     #Held out sample for later validation
    #X_develop, X_validate , Y_develop, Y_validate = train_test_split(X,Y, test_size=0.3)
    
    for div, div_descr in dividers.items():
        

        div_path = Globals.mkdir_LR(token_path+div_descr)
        
        #X_develop, Y_develop, report_develop = rebatcher.single_rebatcher(X_develop, Y_develop, div)
        #X_validate, Y_validate, report_validate = rebatcher.single_rebatcher(X_validate, Y_validate, div)
        #report_develop.concat_report(report_validate)
        #report_develop.saveReport(div_path + 'report' + div + '.txt')            
        
        X, Y , report_batch = rebatcher.single_rebatcher(X,Y, div)
        
        sequencesPerUser(Y, proID, proID_path, token, div)
        
        #some stats on the new divided sets
        #report_develop.saveReport(div_path)
        #reporter.concat_report(report_develop)
        reporter.concat_report(report_batch)
        #running GridSearch on specific token div combination
        
                
        #GridSearch
        
        #score paint report save
        
        
        #reporter.saveReport(proID_path + 'full_report.txt')
        kept_all_estimators, kept_all_top_scores, report_search = run(X,Y, proID, token, div, div_path, SELECT_PIPELINE=3)
        
        reporter.concat_report(report_search)
        
        lab = str(proID) +'-'+ str(token) +'-'+  str(div) 
        labels.append(lab)
        results_estimators.append(kept_all_estimators)
        results_topScores.append(kept_all_top_scores)
        
        filename = lab + '_' + str(cnt)
        
        savepickle(results_topScores, proID_path + 'ALLfromALL/', filename + '_topscores.pickle')
        savepickle(results_estimators, proID_path +  'ALLfromALL/', filename + '_ALLestimators.pickle')
        
    reporter.saveReport(token_path, str(proID) + '_' + str(token) + 'dividers_report.txt')
    
            

           
def run(X_develop, Y_develop, proID, tokens, div, div_path, SELECT_PIPELINE=1):
    
    X = X_develop
    Y = Y_develop
    
    generated_pipelines = getPipelines(Y, SELECT_PIPELINE)
    
    kept_all_best_estimators = {}
    kept_all_best_params = {}
    kept_all_best_full_params = {}
    kept_all_top_scores = {}
    
    report = Reporter()
    """
    if len(generated_pipelines)==1:
        print("OK")
    else:
        print('Something went wrong,TOO MANY PIPES generating the pipelines')
        print('ONLY ONE MUST BE SET')
        exit
    """
    for pipe_desc, gen_pipes in generated_pipelines.items():
        if len(gen_pipes)<1:
            continue;

        print('\n'*10)
        print('##'*40)
        print('RUNNING PIPE :', pipe_desc)
        print('##'*40)
                
        report.new_report('RUNNING PIPE: ' + pipe_desc + ' / ' + str(proID) + '_' + str(tokens) + '_' + str(div))
        
        for cnt, estimator_pipe in enumerate(gen_pipes):        
            print('=='*40)
            print('RUNNING PIPE :', pipe_desc)
            print('__'*40)
            print(estimator_pipe)
            print('=='*40)            
            
            
            if __name__ == "__main__":
                t0 = time.time()
                estimator_pipe.fit(X,Y)
                t1 = time.time()-t0
            else:
                exit
            

            estim = clone(estimator_pipe.estimator)
            best_params = estimator_pipe.best_params_.copy()
            #the best estimator is the estimator of the pipe set with the best
            #parameters of the pipe
            best_estimator = estim.set_params(**best_params)

            clf_name = findclf_name(best_estimator)
            report.subreport('Tested estimator : ' + clf_name + ' in ' + pipe_desc + ' pipe')
            
            try:
                best_estimator.set_params(**{'clf__estimator__probability': True})
            except:
                pass
            
            if clf_name == 'SVC':
                try:
                    clf_name += '_' + best_estimator.named_steps['clf'].kernel
                except:
                    print('probably not an SVC')

            
            steps = best_estimator.named_steps
            dr_dic = {}            
            dr_name = ''            
            if 'dr' in steps:
                dr_dic = {'dr':str(steps['dr'])}
                dr_name = str(steps['dr'].__class__.__name__)
            
            best_full_params = best_params.copy()
            best_full_params.update(dr_dic)
            best_full_params.update({'clf':clf_name})
            

            pipe_path = div_path + pipe_desc + '_pipe_details/'
            filename_starter = str(proID) + '_' + str(tokens) + '_' + str(div) + '_'           
            filename_descr = pipe_desc + '_' +dr_name +'_'+ clf_name
            filename_full =  filename_starter + filename_descr
            
            #save estimator, best_params and best_full_params
            savejson(best_params, pipe_path + 'best_params/', filename_full + '_bestparams.txt')
            savejson(best_full_params, pipe_path + 'best_full_params/', filename_full + '_bestfullparams.txt')
            savepickle(best_estimator, pipe_path + 'estimators/', filename_full +'_bestestimator.pickle')            
            
            text, top_scores = reportSCORES(estimator_pipe.grid_scores_[:20],name=clf_name, pipe_de=pipe_desc, dr = dr_dic)
            savepickle(top_scores, pipe_path + 'topScores/', filename_full + '_topScores.pickle')
            
            clf_descr = clf_name
            if dr_name != '':
                clf_descr += ' (' + dr_name + ')'
            
            kept_all_best_params.update({clf_descr:best_params})
            kept_all_best_full_params.update({clf_descr:best_full_params})
            kept_all_best_estimators.update({clf_descr:best_estimator})
            kept_all_top_scores.update({clf_descr: top_scores})
            
            report.report('Found best parameters')
            report.report('-Execution time: ' + str(t1))
            report.report(str(best_full_params))
            report. report('-----scores and parameters------')
            report.report(str(text))
            report.saveReport(pipe_path +'/reports/', filename_starter + '_' + str(pipe_desc) + '_' + str(cnt) + '_report.txt')
            
        
        filename_starter = proID + '_' + str(tokens) + '_' + str(div) + '_' + str(pipe_desc) + '_pipe'     
        report.saveReport(div_path + 'reports/', filename_starter + '_report.txt')
    
        #Finished pipe execution -Evaluate results
    
        #save estimator, best_params and best_full_params
        savejson(kept_all_best_params, div_path + 'DATA/', filename_starter + '_KEPTbestparams.txt')
        savejson(kept_all_best_full_params, div_path + 'DATA/', filename_starter + '_KEPTbestfullparams.txt')
        savepickle(kept_all_best_estimators, div_path + 'DATA/', filename_starter +'_KEPTbestestimator.pickle')
        savepickle(kept_all_top_scores,  div_path + 'DATA/', filename_starter +'_KEPTALLtopscores.pickle')
        
        
        #evaluate all saved estimators up to that point
        #pass estimator for evaluation        
        EVALUATE_TEST(X, Y, kept_all_best_estimators, div_path +'EVALUATION/', filename_starter)
        EVALUATE_TOPSCORES(kept_all_top_scores, div_path + 'EVALUATION/', filename_starter)
        
        #EVALUATE AGAIN USING A BROADER X,Y)
        
        savejson('FINISHED RUNNING PIPE' + pipe_desc, div_path, pipe_desc + '_STATUS.txt')
    
    return kept_all_best_estimators, kept_all_top_scores, report
            
            
            
            

def sequencesPerUser(Y, proID, path, token, div):
    #to TEST
    cnt = np.zeros(len(np.unique(Y)))
    for i, c in enumerate(np.unique(Y)):
        cnt[i] = (len(Y[Y==c]))
    
    import matplotlib.pyplot as plt
    fig = plt.figure()

    x = np.arange(1,len(cnt)+1)
    plt.bar(x, cnt, align='center')
    plt.ylabel('Number of Sequences')
    plt.xlabel('User')
    plt.title('Sequences per User')
    plt.xticks(x)
    plt.xlim = (0,x.max())
    
    filename = str(proID) + '_' + str(token) + '_' + str(div) + '_sequenciesPerUser'
    plt.savefig(path + filename, format = 'svg', dpi=600)
            
            
if __name__ == "__main__":
    main()           
            
            
        

    
    
    
    