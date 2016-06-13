'''
Created on 2016-06-10
@author: Niranjan
'''
import json
import csv
import os
import numpy as np
import pre_process
from recommender2 import recommender

#Function to evaluate the recommender
def eval(x,file_path,k=50,sim_mesrs=['jaccard']):
    precisions = dict()
    recalls = dict()
    #Load test and train data
    testd = json.loads(open(file_path+'/test_'+str(x)+'.json').read())
    traind = json.loads(open(file_path+'/train_'+str(x)+'.json').read())

    resultFile=open(file_path+'/results.csv','w')
    wr = csv.writer(resultFile)
    for n in [2,5,10]:
        print 'Top {} '.format(n),
        #Start the folds
        for x in xrange(0,1):                       
            for metric in sim_mesrs:
                #Generate Recommendations
                recc = recommender(traind,k=k,metric=metric,n=n)
                #Evaluate results for every user in test data
                for user in testd:
                    found=0
                    scores=recc.recommend(user)
                    try:
                        for tup in scores:  #See if the project is present in test data
                            if tup[0] in testd[user]:
                                found+=1
                    except:
                        continue
                    prec = found/float(len(scores))
                    recall = found/float(len(testd[user]))
                    #Store the precision and recall
                    if metric not in precisions:
                        precisions[metric]=[prec]
                        recalls[metric]=[recall]
                    else:
                        precisions[metric].append(prec)
                        recalls[metric].append(recall)                             
        #Calculate average precision and recall
        for metric in sim_mesrs:
            avg_precision = np.average(precisions[metric])
            avg_recall = np.average(recalls[metric])
            list1=[n,k,metric,avg_precision,avg_recall]
            #To calculate final average among all the folds
            avg_p.append(avg_precision)
            avg_rec.append(avg_recall)
            print 'Precision:%2f, Recall:%2f'%(avg_precision,avg_recall)
            wr.writerow(list1)

if __name__=='__main__':
    op=os.getcwd()
    op=op+'/movdata'
    pre_process.pre_proc_mov(rate=5, split_perc=30)
    pre_process.split_data(rate=5, split_perc=30)
    
    avg_p=[]
    avg_rec=[]
    sims=['jaccard']    #Similarity measures to be used
    for x in range(0,5):    #Start 5 folds and evaluate iteratively
        print 'Starting fold {}'.format(x)
        eval(x,file_path=op,k=100,sim_mesrs=sims)
        
