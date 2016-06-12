import os
import copy
import json
from math import sqrt
from scipy.spatial import distance
import numpy as np
import csv

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
        ratings = [(self.convertProductID2name(k), v) for (k, v) in ratings]
        # finally sort and return
        ratings.sort(key=lambda artistTuple: artistTuple[1], reverse = True)
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
                distance = self.fn(self.data[username], self.data[instance])
                distances.append((instance, distance))
        # sort based on distance -- closest first
        distances.sort(key=lambda artistTuple: artistTuple[1], reverse=True)
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
        recommendations = [(self.convertProductID2name(k), v) for (k, v) in recommendations]
        # finally sort and return
        recommendations.sort(key=lambda artistTuple: artistTuple[1], reverse = True)
        # Return the first n items
        return recommendations[:self.n]
    
def Test2(x,file_path,k=50):
    precisions = dict()
    recalls = dict()

    with open(file_path+'/test_'+str(x)+'.json','r') as fp:
        testd = json.load(fp)
    with open(file_path+'/train_'+str(x)+'.json','r') as fp:
        traind = json.load(fp)

    print "Test users:%d, Train users:%d"%(len(testd),len(traind))

    resultFile=open(file_path+'/results.csv','w')
    wr = csv.writer(resultFile)
    for n in [2]:
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
            list1=[n,k,metric,avg_precision,avg_recall]
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
        
if __name__ == '__main__':
    avg_p=[]
    avg_rec=[]
    op=os.getcwd()
    op=op+'/movdata'
    for x in range(0,5):
        Test2(x,file_path=op)


