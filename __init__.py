'''
Created on 2016-06-10
@author: Niranjan
'''
import pre_process

if __name__=='__main__':
    pre_process.pre_proc_mov(rate=5, split_perc=30)
    pre_process.split_data(rate=5, split_perc=30)