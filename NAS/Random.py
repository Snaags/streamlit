#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import logging 
logging.basicConfig(level=logging.WARNING)
from nasbench import api
import nasbench
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import argparse
from worker_hp_TEPS import MyWorker
from algorithms import GA
from multiprocessing import Pool
from configEncoding import encode
def dict_loop(dict_, list_):
    for ID, config in enumerate(list_):
        dict_[ID] = config
    return dict_
import csv
import os

def save2csv(filename : str, dictionary : dict, first = False,iter_num : int = 0):

    #If this is a new run or file doesnt exist, overwrite/create the file
    if os.path.exists(filename) and first == False:
        write_type = 'a'
    else:
        write_type = 'w' 

    
    with open(filename, write_type) as csvfile:
        writer = csv.writer(csvfile) 
        for i in dictionary:
            writer.writerow([iter_num,i,dictionary[i]])

w = MyWorker(sleep_interval = 0, nameserver='127.0.0.1',run_id='example1')
configspace = w.get_configspace()
###Configuration Settings###
pop_size = 36
elite_size = 0.2
num_iter = 5
batch_size = 256

nasbench = api.NASBench('nasbench_data/nasbench_full.tfrecord')



pop_dict = dict()
population = configspace.sample_configuration(pop_size)
score_dict = dict()
pop_dict = dict_loop(pop_dict, population) 

config_file = "configs.csv"
score_file = "scores.csv"
first = True
for count,i in enumerate(range(num_iter)):
    pop = []
    for i in population:
        pop.append(encode(i.get_dictionary()))
    
    with Pool(processes = 1) as pool:

        results = pool.starmap(train_function, pop)
        pool.close()
        torch.cuda.empty_cache()
        pool.join()
    scores = []
    for i in results:
        scores.append(i)
    score_dict = dict_loop(score_dict, scores) 
    save2csv(config_file, pop_dict,first,count)  
    save2csv(score_file, score_dict,first,count)  
    population = configspace.sample_configuration(pop_size)
    pop_dict = dict_loop(pop_dict, population) 
    first = False 
                 
