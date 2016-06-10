import os
import copy
import random
import json
from math import sqrt
from scipy.spatial import distance
import numpy as np
import sys
import math
import csv
import pickle
users = {"Angelica": {"Blues Traveler": 3.5, "Broken Bells": 2.0,
                      "Norah Jones": 4.5, "Phoenix": 5.0,
                      "Slightly Stoopid": 1.5,
                      "The Strokes": 2.5, "Vampire Weekend": 2.0},
         
         "Bill":{"Blues Traveler": 2.0, "Broken Bells": 3.5,
                 "Deadmau5": 4.0, "Phoenix": 2.0,
                 "Slightly Stoopid": 3.5, "Vampire Weekend": 3.0},
         
         "Chan": {"Blues Traveler": 5.0, "Broken Bells": 1.0,
                  "Deadmau5": 1.0, "Norah Jones": 3.0, "Phoenix": 5,
                  "Slightly Stoopid": 1.0},
         
         "Dan": {"Blues Traveler": 3.0, "Broken Bells": 4.0,
                 "Deadmau5": 4.5, "Phoenix": 3.0,
                 "Slightly Stoopid": 4.5, "The Strokes": 4.0,
                 "Vampire Weekend": 2.0},
         
         "Hailey": {"Broken Bells": 4.0, "Deadmau5": 1.0,
                    "Norah Jones": 4.0, "The Strokes": 4.0,
                    "Vampire Weekend": 1.0},
         
         "Jordyn":  {"Broken Bells": 4.5, "Deadmau5": 4.0,
                     "Norah Jones": 5.0, "Phoenix": 5.0,
                     "Slightly Stoopid": 4.5, "The Strokes": 4.0,
                     "Vampire Weekend": 4.0},
         
         "Sam": {"Blues Traveler": 5.0, "Broken Bells": 2.0,
                 "Norah Jones": 3.0, "Phoenix": 5.0,
                 "Slightly Stoopid": 4.0, "The Strokes": 5.0},
         
         "Veronica": {"Blues Traveler": 3.0, "Norah Jones": 5.0,
                      "Phoenix": 4.0, "Slightly Stoopid": 2.5,
                      "The Strokes": 3.0}
        }



class recommender:

    def __init__(self, data, k=50, metric='pearson', n=5):
        """ initialize recommender
        currently, if data is dictionary the recommender is initialized
        to it.
        For all other data types of data, no initialization occurs
        k is the k value for k nearest neighbor
        metric is which distance formula to use
        n is the maximum number of recommendations to make"""
        self.n = n
        self.username2id = {}
        self.userid2name = {}
        self.productid2name = {}
        # for some reason I want to save the name of the metric
        self.metric = metric
        if self.metric == 'pearson':
            self.fn = self.pearson
        elif self.metric == 'cosine':
            self.fn = self.cosinesim
        elif self.metric == 'jaccard':
            self.fn = self.jaccardsim
        #
        # if data is dictionary set recommender data to it
        #
        if type(data).__name__ == 'dict':
            self.data = data
        if (n == 'all'):
            self.k = len(self.data)
        else:
            self.k = k
    def manhattan(self,rating1, rating2):
        """Computes the Manhattan distance. Both rating1 and rating2 are dictionaries
           of the form {'The Strokes': 3.0, 'Slightly Stoopid': 2.5}"""
        distance = 0
        commonRatings = False 
        for key in rating1:
            if key in rating2:
                distance += abs(rating1[key] - rating2[key])
                commonRatings = True
        if commonRatings:
            return distance
        else:
            return -1 #Indicates no ratings in common
        
    # method to calculate the distance using jaccard similarity
    def jaccardsim(self,rating1, rating2):
                
        user1 = set(rating1)
        user2 = set(rating2)
        # get commonly rated items
        common = user1.intersection(user2)
        uniq_itms = user1.union(user2)
        if(len(common) == 0):
            # no items were similar between users
            return 0
        else:
            jaccard_sim = len(common)/float(len(uniq_itms))
        return jaccard_sim    
    # method that calculates the ditance using cosine  similarity
    def cosinesim(self,rating1, rating2):
        
        user1 = []
        user2 = []
        for key in rating1:
            if key in rating2:
                # there is a common project that both users have backed
                x = rating1[key]
                y = rating2[key]
                user1.append(x)
                user2.append(y)
        if (len(user1) != 0):
            user1 = np.array(user1)
            user2 = np.array(user2)
            # added one since scipy subtracts 1 from cosine formula
            cosin_dist = 1 + distance.cosine(user1, user2)
        else:
            # no items were common between the users
            return 0
        
        return cosin_dist
    
    def convertProductID2name(self, id):
        """Given product id number return product name"""
        if id in self.productid2name:
            return self.productid2name[id]
        else:
            return id


    def userRatings(self, id, n):
        """Return n top ratings for user with id"""
        print ("Ratings for " + self.userid2name[id])
        ratings = self.data[id]
        print(len(ratings))
        ratings = list(ratings.items())
        ratings = [(self.convertProductID2name(k), v)
                   for (k, v) in ratings]
        # finally sort and return
        ratings.sort(key=lambda artistTuple: artistTuple[1],
                     reverse = True)
        ratings = ratings[:n]
        for rating in ratings:
            print("%s\t%i" % (rating[0], rating[1]))

                
        
    def pearson(self, rating1, rating2):
        sum_xy = 0
        sum_x = 0
        sum_y = 0
        sum_x2 = 0
        sum_y2 = 0
        n = 0
        for key in rating1:
            if key in rating2:
                n += 1
                x = rating1[key]
                y = rating2[key]
                sum_xy += x * y
                sum_x += x
                sum_y += y
                sum_x2 += pow(x, 2)
                sum_y2 += pow(y, 2)
        if n == 0:
            # no items were common between the users
            return None
        # now compute denominator
        # denominator becomes zero if both have backed the same project
        denominator = (sqrt(sum_x2 - pow(sum_x, 2) / n)
                       * sqrt(sum_y2 - pow(sum_y, 2) / n))
        if denominator == 0:
            return 0
        else:
            return (sum_xy - (sum_x * sum_y) / n) / denominator


    def computeNearestNeighbor(self, username):
        """creates a sorted list of users based on their distance to
        username"""
        distances = []
        for instance in self.data:
            if instance != username:
                distance = self.fn(self.data[username],
                                   self.data[instance])
                distances.append((instance, distance))
        # sort based on distance -- closest first
        distances.sort(key=lambda artistTuple: artistTuple[1],
                       reverse=True)
        return distances

    def recommend(self, user):
        """Give list of recommendations"""
        recommendations = {}
        # first get list of users  ordered by nearness
        nearest = self.computeNearestNeighbor(user)
        #
        # now get the ratings for the user
        #
        userRatings = self.data[user]
        #
        # determine the total distance
        totalDistance = 0.0
        for i in range(self.k):
            totalDistance += nearest[i][1]
        if (totalDistance == 0):
            # there are no nearest neighbors for this user
#             print 'no nearest neighbors found.'
            return 0
        # now iterate through the k nearest neighbors
        # accumulating their ratings
        for i in range(self.k):
            # compute slice of pie 
            weight = nearest[i][1] / totalDistance
            # get the name of the person
            name = nearest[i][0]
            # get the ratings for this person
            neighborRatings = self.data[name]
            # get the name of the person
            # now find bands neighbor rated that user didn't
            for artist in neighborRatings:
                if not artist in userRatings:
                    if artist not in recommendations:
#                         recommendations[artist] = (neighborRatings[artist]* weight)
                        recommendations[artist] = (1* weight)
                    else:
                        '''increment the weight of the item if it is already present. 
                        items having many nearest neighbors get higher preference'''
#                         recommendations[artist] = (recommendations[artist]+ neighborRatings[artist]* weight)
                        recommendations[artist] = (recommendations[artist]+ 1* weight)
        # now make list from dictionary
        recommendations = list(recommendations.items())
        recommendations = [(self.convertProductID2name(k), v)
                           for (k, v) in recommendations]
        # finally sort and return
        recommendations.sort(key=lambda artistTuple: artistTuple[1],
                             reverse = True)
        # Return the first n items
        return recommendations[:self.n]
#         return recommendations

# method to test group recommendation
def test_grprecc():
    
    # training dataset
    folds=1
    usrs_itm = dict()
    t_grpitm = set()
    grp_itm = dict()
    t_itms_rmvd = 0
    # test dataset
    test_dat = {}
    with open('grp_list_12', 'rb') as f:
        # group item list
        data = pickle.load(f)
    
        for indx,grp in enumerate(data):
            for u_itm in grp:
                usr = u_itm[0]
                itm = u_itm[1]
                if (indx not in grp_itm):
                    grp_itm[indx] = {itm:set()}
                    grp_itm[indx][itm].add(usr)
                else:
                    if(itm not in grp_itm[indx]):
                        grp_itm[indx][itm] = set()
                        grp_itm[indx][itm].add(usr)
                    else:
                        grp_itm[indx][itm].add(usr)
                t_grpitm.add((indx,itm))
                # create the user-item dictionary
                if (usr not in usrs_itm):
                    usrs_itm[usr] = {itm:1.0}
                else:
                    if (itm not in usrs_itm[usr]):
                        usrs_itm[usr][itm] = 1.0
    count = 0
    for k in usrs_itm:
        count += len(usrs_itm[k])
    print count
    # the number of ground truth to remove
    tru_remov = int(np.round(len(t_grpitm)/float(10),0))
    print 'the program will remove %d ground truths to create a test and training set'%(tru_remov)
    resultFile=open('sresults_1.csv','a')
    wr = csv.writer(resultFile)
    
    for top_k in [2,5,10]:
        for x in xrange(0,folds):
            print 'Top%d Fold:%d'%(top_k,x)
            for i in range(tru_remov):
                # get a random group
                grp = np.random.choice(grp_itm.keys())
                # generate a random item to be removed for the list of items
                itm_to_rem = np.random.choice(grp_itm[grp].keys())
                # get the list of users and item that is to be removed
                usr1 = grp_itm[grp][itm_to_rem]
                # create an entry for this grp, user and item in test dataset
                if (grp not in test_dat):
                    test_dat[grp] = {itm_to_rem:usr1}
                else:
                    # if the item was already added to the test dataset continue
                    if(itm_to_rem in test_dat[grp]):
                        continue
                    else:
                        test_dat[grp] = {itm_to_rem:usr1}
            # iterate through test dataset and remove this ground truth from training set
            for grp in test_dat:
                for itms in test_dat[grp]:
                    users = test_dat[grp][itms]
                    for u in users:
                        # check if this item is present in users backing
                        if (itms in usrs_itm[u]):
                            # delete the item for this user
                            del(usrs_itm[u][itms])
                            t_itms_rmvd += 1
            
            recc = recommender(usrs_itm,metric='jaccard',n=100)
            itm_cnt = 0
            
            pr_scores = []
            rec_scores = []
            mrr_scores=[]
            succ_scores=[]
            dcg_scores=[]
            for grp in test_dat:
                found_cnt = 0
                succ=0
                mrr=0
                dcg=0
                tot_gtruth = len(test_dat[grp])
                for itm in test_dat[grp]:
                    itm_cnt += 1
                    usrs = test_dat[grp][itm]
                    recc_itms = dict()
                    # add the items scores for all the users in this group
                    for u in usrs:
                        # query for these users
                        recc_list = recc.recommend(u)
                        # if no recommendation is found continue to the next user
                        if (recc_list == 0):
                            continue
                        for i_scor in recc_list:
                            itm1 = i_scor[0]
                            score = i_scor[1]
                            if (itm1 not in recc_itms):
                                recc_itms[itm1] = score
                            else:
#                                 if score < recc_itms[itm1]:
                                recc_itms[itm1] += score
                                
                    # sort the top-k items
                    ranked_itms = sorted(recc_itms, key=recc_itms.get, reverse=True)[:top_k]
                    
                    if itm in ranked_itms:
                        succ=1
                        mrr+=1/(float(ranked_itms.index(itm)+1))
                        if ranked_itms.index(itm)==0:
                            dcg+=10
                        else:
                            dcg+=10/math.log((ranked_itms.index(itm)+1),2)
                        found_cnt += 1
                precision = found_cnt / float(top_k)
                recall = found_cnt / float(tot_gtruth)
                pr_scores.append(precision)
                rec_scores.append(recall)
                mrr_scores.append(mrr)
                succ_scores.append(succ)
                dcg_scores.append(dcg)
        pre=np.average(pr_scores)
        rec=np.average(rec_scores)
        mr=np.average(mrr_scores)
        su=np.average(succ_scores)
        dc=np.average(dcg_scores)
        print 'Precision:%2f, Recall:%2f, MRR:%2f, Success:%2f DCG:%2f'%(pre,rec,mr,su,dc)
        list1=[top_k,pre,rec]
#         wr.writerow(list1)
        pre=[]
        rec=[]
    
    
#Test user project matrix    
def Test():
  
    with open('data/user_projectDict2.json','r') as fp:
        dataset = json.load(fp)
    
    print 'for the user 346 if jaccard distance is used the recommendations are'
    recc = recommender(dataset,metric='jaccard')
    print recc.recommend('346')
    print 'for the user 346 if cosine distance is used the recommendations are'
    recc = recommender(dataset,metric='cosine')
    print recc.recommend('346')  
    
def Test2(x):
    precisions = dict()
    recalls = dict()
    k=50
    current_dir = os.getcwd()
    file_path=current_dir+'/foursq_data'
    with open(file_path+'/ftest_'+str(x)+'.json','r') as fp:
        testd = json.load(fp)
    with open(file_path+'/ftrain_'+str(x)+'.json','r') as fp:
        traind = json.load(fp)
    
    c=0
    for usr in testd:
        lent=len(testd[usr])
        c+=lent
    print c/len(testd)
    
    c1=0
    for usr in traind:
        lent1=len(traind[usr])
        c1+=lent1
    print c1/len(traind)
    print "Test users:%d, Train users:%d"%(len(testd),len(traind))
#     for proj in dataset:
#         list_bkr=dataset[proj]
#         for bkr in list_bkr:
#             if bkr not in user_prj_data:
#                 user_prj_data[int(bkr)]=[int(proj)]
#             else:
#                 user_prj_data[int(bkr)].append(int(proj))
                
#     #Convert the data in {user:[prj1,prj2]}
#     for user in dataset:
#         prjs=dataset.get(user)
#         list1=prjs.keys()
#         user_prj_data[user]=list1

    resultFile=open(file_path+'/results.csv','w')
    wr = csv.writer(resultFile)
    for n in [2,5,10,20]:
        print 'Top {} '.format(n),
        #Start the folds
        for x in xrange(0,1):                       
            for metric in ['jaccard']:
                #Generate Recommendations
                recc = recommender(traind,k=k,metric=metric,n=n)
                #Evaluate results for every user in test data
                for user in testd:
                    found=0
                    scores=recc.recommend(user)
                    try:
                        for tup in scores:
                            if tup[0] in testd[user]:
                                found+=1
                    except:
                        continue
                    prec = found/float(len(scores))
                    recall = found/float(len(testd[user]))
                    if metric not in precisions:
                        precisions[metric]=[prec]
                        recalls[metric]=[recall]
                    else:
                        precisions[metric].append(prec)
                        recalls[metric].append(recall)                             
        for metric in ['jaccard']:
            avg_precision = np.average(precisions[metric])
            avg_recall = np.average(recalls[metric])
            list1=[n,k,metric]
            list1.append(avg_precision)
            list1.append(avg_recall)
            avg_p.append(avg_precision)
            avg_rec.append(avg_recall)
            print 'Precision:%2f, Recall:%2f'%(avg_precision,avg_recall)
            wr.writerow(list1)
def calc_fin(l1):
    dicta=dict()
    print ''
    for item in l1:
        ind=l1.index(item)
        tp=ind%4
        if tp not in dicta:
            dicta[tp]=item
        else:
            dicta[tp]+=item
    print dicta
    for val in dicta:
        print dicta[val]/float(5),

def pre_proc_mov():
    dict_mov=dict()
    op_path=os.getcwd()
    op_path+='/movdata'
    rate=5
    folds=30
    cnt=0
    set_mov=set()
    fp=open('100k_movi.dat','r')
    for line in fp:
        line=line.strip()
        line=line.split('\t')
        user=line[0]
        project=line[1]
        rating=line[2]
        if rate ==5:
            cnt+=1
            dict_1=dict()
            set_mov.add(project)
            dict_1[int(project)]=int(rating)
            if int(user) not in dict_mov:            
                dict_mov[int(user)]={int(project):int(rating)}
            else:
                dict_mov[int(user)].update(dict_1)
            dict_1=dict()   
             
        elif rate==1:
            rating=1
            dict_1=dict()
            dict_1[int(project)]=int(rating)
            if int(user) not in dict_mov:            
                dict_mov[int(user)]={int(project):int(rating)}
            else:
                dict_mov[int(user)].update(dict_1)
            dict_1=dict()   
        
    with open(op_path+"/mov_lens100_rate"+str(rate)+".json", "w") as outfile:
        json.dump(dict_mov, outfile) 

    #This is the data for Jaegul, from FilterData.py
    with open(op_path+'/mov_lens100_rate'+str(rate)+'.json','r') as fp:
        user_prj_data = json.load(fp)
    
    count=0
    for user in user_prj_data:
        lent=len(user_prj_data[user])
        count+=lent
#     print 'Density {}'.format(count/float(len(set_mov)*len(user_prj_data)))
#     sys.exit()
    length=count
    print length
    #Determine test size
    test_sz=length*folds/100
    print test_sz
    
#     f=open('test.dat','w')
#     f1=open('train.dat','w')
    count=0
    for x in range(0,5):
        dict_removed=dict()
        copy_dict=copy.deepcopy(user_prj_data)
        while count!=test_sz:
            uid=np.random.choice(copy_dict.keys())
            a=copy_dict[uid].keys()
            if len(a)!=0 :
                pindx=np.random.choice(copy_dict[uid].keys())
                if uid not in dict_removed:
                    dict_removed[uid]=[pindx]
                else:
                    dict_removed[uid].append(pindx)
#                 f.write(str(uid)+'::'+str(pindx)+'::'+str(copy_dict[uid][pindx])+'\n')
#                 f.flush()
                count+=1
                del copy_dict[uid][pindx]
                
#         for usr in copy_dict:
#             for mov in copy_dict[usr]:
#                 f1.write(str(usr)+'::'+str(mov)+'::'+str(copy_dict[usr][mov])+'\n')
#                 f1.flush()
#         print len(dict_removed) 
                          
        with open(op_path+"/test_"+str(x)+".json", "w") as outfile:
            json.dump(dict_removed, outfile)
        with open(op_path+"/train_"+str(x)+".json", "w") as outfile:
            json.dump(copy_dict, outfile)
  

def kck_preproc(fname,test_perc,folds,out_format,op_path):
    '''
    test_perc: The percentage split of the original data for test data
    folds: Number of folds
    out_format: Format of the output. Either 'json' or 'mat'
    '''
    dict_bkr=dict()
    folds=folds
    rating=1
#     fname='prj_n_bkr_500_30bkings.json'
    if(op_path == 'current'):
        op_path = os.getcwd()
    with open(fname,'r') as fp:
        dict_prj_bkr = json.load(fp)
    #Create user-project:rating dict
    for prj in dict_prj_bkr:
        list_bkr=dict_prj_bkr[prj]
        rating=1
        for bkr in list_bkr:
            dict_1=dict()
            dict_1[int(prj)]=int(rating)
            if int(bkr) not in dict_bkr:            
                dict_bkr[int(bkr)]={int(prj):int(rating)}
            else:
                dict_bkr[int(bkr)].update(dict_1)
            dict_1=dict()   
    #Calculate total number of 1s
    count=0
    for user in dict_bkr:
        lent=len(dict_bkr[user])
        count+=lent
    length=count
    print 'Total number of 1s:%d'%length
    
    #Determine test size
    test_sz=length*test_perc/100
    print 'Test size:%d'%test_sz
    
    #Start folds
    for x in range(0,folds):
        count=0
        dict_removed=dict()
        dict_train=copy.deepcopy(dict_bkr)
        #Create test and train
        while count!=test_sz:
            uid=np.random.choice(dict_train.keys())
            a=dict_train[uid].keys()
            if len(a)!=0 :
                pindx=np.random.choice(dict_train[uid].keys())
                if uid not in dict_removed:
                    dict_removed[uid]=[pindx]
                else:
                    dict_removed[uid].append(pindx)
                count+=1
                del dict_train[uid][pindx]
        
        if out_format=='json':  
            with open(op_path+"/test_"+str(x)+".json", "w") as outfile:
                json.dump(dict_removed, outfile)
            with open(op_path+"/train_"+str(x)+".json", "w") as outfile:
                json.dump(dict_train, outfile)
        elif out_format=='mat':
            #Call the function to write the data in matrix format
            json_to_mat(dict_removed, dict_train, x, len(dict_bkr), len(dict_prj_bkr),op_path)
        
#Function to convert .json file (test/train) to matrix
def json_to_mat(test,train,x,bkrs,prjs):
    ''' 
    test: a dictionary with keys as test backers and value as list of projects
    train: same as test but only with train backers and projects
    x: current fold number, used to write the filename
    bkrs: length of backers
    prjs: length of projects 
    '''
        
    count=0
    #Do for test and train both
    for typ in [test,train]:
        count+=1
        if count==1:
            name='test'
        else:
            name='train'
        #Create a zero matrix
        matrix_file=open(name+str(x)+".mat","w")
        main_data = np.zeros(shape=(bkrs,prjs),dtype=np.uint8)
        
        my_dict=typ
        #Traverse through dictionary and update matrix  
        for backer in my_dict:
            list_projects=my_dict.get(backer)
            for prj_indx in list_projects:
                main_data[backer][prj_indx]=1
        np.save(matrix_file, main_data)
        
def Test1():
    user_prj_data=dict()
    precisions = dict()
    recalls = dict()
    folds=20
    with open('mov_lens100_rate5.json','r') as fp:
        user_prj_data = json.load(fp)
    #Convert the data in {user:[prj1,prj2]}
#     for user in dataset:
#         prjs=dataset.get(user)
#         list1=prjs.keys()
#         user_prj_data[user]=list1
        
    length=len(user_prj_data)
    #Determine test size
    test_sz=length/folds
    
    resultFile=open('Rec_results_rate1.csv','a')
    wr = csv.writer(resultFile)
    for n in [2]:
        for k in xrange(50,51,25):
            print (n,k)
            #Start the folds
            for x in xrange(0,1):
                copy_dict=copy.deepcopy(user_prj_data)
                #Select list of random user ids from all users of length test_sz
                uid_list=random.sample(copy_dict.keys(),test_sz)
                #Dictionary to store the test users and their projects
                dict_removed=dict()
                
                #Remove the user projects from the train data and store in test data
                for user in uid_list:
                    limit=len(copy_dict[user])
                    pindx=random.randint(0,limit-1)
                    if user not in dict_removed:
                        dict_removed[user]=[copy_dict[user][pindx]]
                    else:
                        dict_removed[user].append(copy_dict[user][pindx])
                    del copy_dict[user][pindx]
                    
                for metric in ['jaccard']:
                    #Generate Recommendations
                    recc = recommender(copy_dict,k=k,metric=metric,n=n)
                
                    #Evaluate results for every user in test data
                    for user in dict_removed:
                        found=0
                        scores=recc.recommend(user)
                        for tup in scores:
                            if tup[0] in dict_removed[user]:
                                found+=1
                        prec = found/float(len(scores))
                        recall = found/float(len(dict_removed[user]))
                        if metric not in precisions:
                            precisions[metric]=[prec]
                            recalls[metric]=[recall]
                        else:
                            precisions[metric].append(prec)
                            recalls[metric].append(recall)
                            
            for metric in ['jaccard']:
                avg_precision = np.average(precisions[metric])
                avg_recall = np.average(recalls[metric])
                list1=[n,k,metric]
                list1.append(avg_precision)
                list1.append(avg_recall)
                print 'Metric-->%s\nPrecision:%2f, Recall:%2f'%(metric,avg_precision,avg_recall)
                wr.writerow(list1) 
 
def Vin_perproc(name,cc):
    
    dict_train=dict()
    train=np.load(op+'/'+name+'.mat')
    sh= np.shape(train)
    print sh
    for counter,row in enumerate(train):
        dict_temp=dict()
        for val in range(0,sh[1]):
            if row[val]!=0:
                dict_temp[val]=1
        dict_train[counter]=dict_temp    
        
        
    print len(dict_train)
    with open(op+'/'+name+'.json','w') as fp:
        json.dump(dict_train,fp)
    return cc

def Vin_perproc1(name,cc):
    
    dict_train=dict()
    train=np.load(op+'/'+name+'.mat')
    sh= np.shape(train)
    print sh
    for counter,row in enumerate(train):
        for val in range(0,sh[1]):
            if row[val]!=0:
                if counter not in dict_train:
                    dict_train[counter]=[val]
                    cc+=1
                else:
                    dict_train[counter].append(val)
                    cc+=1
    print len(dict_train)
    with open(op+'/'+name+'.json','w') as fp:
        json.dump(dict_train,fp)
    return cc
  
if __name__ == '__main__':

    avg_p=[]
    avg_rec=[]
    cc=0
    ip_path='/home/anand/Desktop/workspace/create_18k_data/src/new_filter/dict_prj1212_usr5389.json'
    op=os.getcwd()
    op=op+'/movdata'
#     pre_proc_mov()
#     cc=Vin_perproc('train1',cc)
#     print cc
#     cc=Vin_perproc1('test1',cc)
#     print cc
#     kck_preproc(fname=ip_path,test_perc=30,folds=5,out_format='json',op_path=op)
    for x in range(0,5):
        Test2(x)
#     print avg_p
#     print avg_rec
#     for met in [avg_p,avg_rec]:
#         calc_fin(met)

        
#     for x in range(0,5):
#         (prec,rec)=Test2(x)
#         for y in range(0,4):
#             avg_p[y]+=prec[y]
#             avg_rec[y]+=rec[y]
#     print avg_p
#     print avg_rec
#     for x in range(0,4):
#         print 'Precision:%2f, Recall:%2f'%(avg_p[y]/float(5),avg_rec[y]/float(5))
    
#     Test1()
#     test_grprecc()

