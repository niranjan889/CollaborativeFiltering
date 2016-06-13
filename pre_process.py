'''
Created on 2016-06-10
@author: Niranjan
'''
import os
import json
import copy
import numpy as np

#Function to create ratings json file either binary or other
def pre_proc_mov(rate=5,split_perc=30):
    dict_mov=dict()
    set_mov=set()
    op_path=os.getcwd()
    op_path+='/movdata'
    
    #Open the ratings file to read
    fp=open('100k_movi.dat','r')
    for line in fp:
        line=line.strip()
        line=line.split('\t')
        user=line[0]
        project=line[1]
        rating=line[2]
        #Store the user, movie and corresponding rating
        if rate == 5:
            dict_1=dict()
            set_mov.add(project)
            dict_1[int(project)]=int(rating)
            if int(user) not in dict_mov:            
                dict_mov[int(user)]={int(project):int(rating)}
            else:
                dict_mov[int(user)].update(dict_1)
            dict_1=dict()   
        #Convert all the original ratings to binary   
        elif rate == 1:
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

# Function to split the data in test and train using cross validation
def split_data(rate=5,split_perc=30):
    folds=split_perc
    op_path=os.getcwd()
    op_path+='/movdata'

    #Load the data from json file
    with open(op_path+'/mov_lens100_rate'+str(rate)+'.json','r') as fp:
        user_prj_data = json.load(fp)
    
    count=0
    for user in user_prj_data:
        lent=len(user_prj_data[user])
        count+=lent
    length=count
    print 'Total ratings: {}'.format(length)
    
    #Determine test size
    test_sz=length*folds/100
    print 'No. of ratings to be removed for test: {}'.format(test_sz)
    
    count=0
    #Start 5 fold cross validation
    for x in range(0,5):
        dict_removed=dict()
        copy_dict=copy.deepcopy(user_prj_data)  #Create a copy of current dictionary to remove the data from
        while count!=test_sz:                   #Repeat until specified no. of ratings are removed
            uid=np.random.choice(copy_dict.keys())
            a=copy_dict[uid].keys()             #Randomly choose a user and a movie to be removed
            if len(a)!=0 :
                pindx=np.random.choice(copy_dict[uid].keys())
                #Keep the removed data in another dictionary
                if uid not in dict_removed:
                    dict_removed[uid]=[pindx]
                else:
                    dict_removed[uid].append(pindx)
                count+=1
                del copy_dict[uid][pindx]

        #Store the data in json files                          
        with open(op_path+"/test_"+str(x)+".json", "w") as outfile:
            json.dump(dict_removed, outfile)
        with open(op_path+"/train_"+str(x)+".json", "w") as outfile:
            json.dump(copy_dict, outfile)
            
