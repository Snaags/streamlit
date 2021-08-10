import torch
import csv
from datasets import Train_TEPS, Test_TEPS
import copy
from multiprocessing import Pool
import time
import collections
import random
from OOP_config import init_config
import matplotlib.pyplot as plt 
import numpy as np
###Tounament Selection/Aging Selection
  #
  #parameters:
  # S : (int) number of solutions to sample at each iteration
  # P : population size
  # C : Total cycles of evolution 
  # parent : highest-accuracy model in sample
  # history : Record of all models
  # population : Currently active models


# IDEAS
#
# - Run aging evolution for "species" with 1 species per core allowing cross-over occationally
#g
#
#
#
#
#


##Define Train Function
##############################################
from TEPS_worker import main as train_function
###############################################

def train_function_test(args, t,ts):
  score = 0 
  for i in args:
    print(args[i])
    if "0" in str(args[i]):
      score +=1 
      print("Score is now: ", score)
  return score
class Model:
  def __init__(self):
    self._arch = None
    self.accuracy = 0

  def set_arch(self, arch):
    self._arch = arch
  
  def set_value(self, name, value):
    self._arch[name] = value

  def arch(self):
    return self._arch.get_dictionary()

def train_and_eval_population(population, sample_batch, train = None, test= None):
  configs = []
  for model in population:
    configs.append([model.arch(),train,test])

  with Pool(processes = sample_batch) as pool:
      results = pool.starmap(train_function, configs)
      pool.close()
      torch.cuda.empty_cache()
      pool.join()

  for mod,result in zip(population,results):
    mod.accuracy = result

def Mutate(cs, parent_model : Model) -> Model:
  model = copy.deepcopy(parent_model) 
  def select_operation(model):
    operation_list = []
    pre_string = "normal_cell_1_ops_"
    post_string = "_type"
    for parameter in model.arch():
      if (pre_string in parameter) and (post_string in parameter):
        operation_list.append(parameter[-6])
    
    return operation_list[random.randint(0,len(operation_list)-1)]
        
    
  def op_mutation(model: Model, operation_number : int) -> Model:
    parameter_name = "normal_cell_1_ops_"+operation_number+"_type"
    
    operations = cs.get_hyperparameter(parameter_name).choices
    old_op = model.arch()[parameter_name] 
   
    while model.arch()[parameter_name] == old_op:
      model.set_value(parameter_name,  operations[random.randint(0, len(operations)-1)]  )
    print("Old operation: ", old_op)
    print("New operation: ", model.arch()[parameter_name])
    return model
 
  def hidden_state_mutation(model : Model, operation_number : int) -> Model:
   
    #Select 1 of the inputs at random
    parameter_name = "normal_cell_1_ops_"+operation_number+"_input_"+str(random.randint(1,2))
    old_op = model.arch()[parameter_name] 
    count = 0 
    while model.arch()[parameter_name] == old_op:
      model.set_value(parameter_name,  random.randint(0,int(operation_number)-1)  )
      count+=1 
      if count > 20:
        operation_number = select_operation(model)
        if random.randint(0,5) == 5:
          return op_mutation(model, operation_number)
        parameter_name = "normal_cell_1_ops_"+operation_number+"_input_"+str(random.randint(1,2))
        old_op = model.arch()[parameter_name] 
        count = 0 
    print("Old operation: ", old_op)
    print("New operation: ", model.arch()[parameter_name])
    return model

  #Select Operation to mutate
  operation_number = select_operation(model) 
  print(operation_number) 
  mutation = [op_mutation, hidden_state_mutation]
 
  #Select an operation at random 
  return mutation[random.randint(0,1)](model, operation_number)



def regularized_evolution(cycles, population_size, sample_size, sample_batch_size):
  """Algorithm for regularized evolution (i.e. aging evolution).

  Follows "Algorithm 1" in Real et al. "Regularized Evolution for Image
  Classifier Architecture Search".

  Args:
  cycles: the number of cycles the algorithm should run for.
  population_size: the number of individuals to keep in the population.
  sample_size: the number of individuals that should participate in each
      tournament.
  sample_batch_size: Number of children to generate before evaluating and adding to population
  Returns:
  history: a list of `Model` instances, representing all the models computed
      during the evolution experiment.
  """
  population = []
  history = []  # Not used by the algorithm, only used to report results.
  train = Train_TEPS() 
  test = Test_TEPS()

  CS = init_config()
  # Initialize the population with random models.
  while len(population) < population_size:
    model = Model()
    model.set_arch( CS.sample_configuration() )
    population.append(model)
    history.append(model)

  train_and_eval_population(population, sample_batch_size, train, test)

  # Carry out evolution in cycles. Each cycle produces a model and removes
  # another.
  children = []
  print_counter = 0
  while len(history) < cycles:

    print("Starting cycle", len(history) - population_size)
    # Sample randomly chosen models from the current population.

    if len(children) < sample_batch_size:
      sample = []
      while len(sample) < sample_size:
        # Inefficient, but written this way for clarity. In the case of neural
        # nets, the efficiency of this line is irrelevant because training neural
        # nets is the rate-determining step.
        candidate = random.choice(list(population))
        sample.append(candidate)

      # The parent is the best model in the sample.
      parent = max(sample, key=lambda i: i.accuracy)

      # Create the child model and store it.
      child = Model()
      child = Mutate(CS,parent)
      children.append(child)

    else:
      train_and_eval_population(children, sample_batch_size, train, test)
      for i in children:
        population.append(i)
        history.append(i)
        population.pop(0)
      children = []
      # Remove the oldest model.
      best = max(population, key=lambda i: i.accuracy)
      print("--Current best model--")
      print("Accuracy: ",best.accuracy)
      print("Architecture: ", best.arch())
      print("Population Size: ", len(population))
      print("Total Evaluations: ", len(history))
      print_counter +=1
      print_counter = 0
      accuracy_scores = []
      for i in history:
        accuracy_scores.append(i.accuracy)

      with open("RegEvo.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        for i in history:
          writer.writerow([i.accuracy,i.arch()]) 
      plt.plot(accuracy_scores)
      plt.title("Accuracy Scores of Configurations")
      plt.xlabel("Generation")
      plt.ylabel("Accuracy")
      plt.grid()
      plt.savefig("regevolive.png",dpi = 1200)
      plt.clf()

  return history


def main():
  pop_size = 32
  evaluations = 200
  history = regularized_evolution(cycles = evaluations, population_size =  pop_size, sample_size = 8, sample_batch_size = 1)
  Architectures = []
  accuracy_scores = []
  generations = list(range(evaluations))
  for i in history:
    accuracy_scores.append(i.accuracy)
    Architectures.append(i.arch)
  
  plt.scatter(generations[:pop_size],accuracy_scores[:pop_size], c = "red")
  plt.scatter(generations[pop_size:],accuracy_scores[pop_size:])
  plt.title("Accuracy Scores of Configurations")
  plt.xlabel("Generation")
  plt.ylabel("Accuracy")
  plt.grid()
  plt.savefig("regevo.png",dpi = 1200)

  indexs = accuracy_scores.index(max(accuracy_scores))
  print("Best accuracy: ", accuracy_scores[indexs])
  print("Best Hyperparameters: ", Architectures[indexs])




if __name__ == '__main__':
  main()
