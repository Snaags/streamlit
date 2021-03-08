import train_function
import torch
import random
import math
import time
from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
def Dispatcher(Configurations: list, num_workers: int, train_func = train_function) -> dict:
	"""
		returns a dictionary "ID":[hyperparameters: dict, Model_weights: dict, Training Loss: list[float], Training Time: int]	
	"""
 
	with Pool(processes=num_workers) as pool:
		results = pool.map_async(train_func.main,Configurations)
		pool.close()
		pool.join()


	return results.get(timeout=2)

class Tree: 
	structure = {}

	def __init__(self,var_name,types = "node",min_val= None,max_val=None):
		self.type = types
		if self.type == "catagorical":
			self.catagories = min_val
		else:
			self.min = min_val
			self.max = max_val

		self.children = []
		self.var_name = var_name



	def fill_data(self,types,min_val,max_val):
		self.type = types
		self.min = min_val
		self.max = max_val


	def generate_random_value(self):																															
		return random.randint(self.min, self.max)

	def add_children(self,child_args):
		for args in child_args:

			self.children.append(Tree(*args))


	def traverse(self,tab_level=0):
		print("\t"*tab_level,self.var_name)
		if self.type == "catagorical":
			for i in self.catagories:
				print("\t"*(tab_level+1),i)

		for i in self.children:
			i.traverse(tab_level= tab_level+1)


def orderOfMagnitude(number):
    return math.floor(math.log(number, 10))

def orderOfMagnitudePo2(number):
    return math.floor(math.log(number, 2))


class Parameter:
	"""
	types:
		discrete
		catagorical
		continuous
	
	scale:
		linear
		log
		po2
	"""

	def __init__(self,set_type, set_scale_or_catagories, set_min = None, set_max = None, requirements = None):
		self.type = set_type
		if self.type == "continuous" or  self.type == "discrete":			 
			self.scale = set_scale_or_catagories
			self.max = set_max
			self.min = set_min
		else:
			self.catagories = set_scale_or_catagories

		self.requires = requirements
		self.last_generated_result = None
		self.output_memory =[]


	def mutate_val(self, current_val,threshold = 0.8):
		if self.type == "continuous":
			if random.random() > threshold:
				while True:
					output = current_val+(current_val*random.uniform(-0.2,0.2))
					if output > self.min and output < self.max:
						return output

		if self.type == "discrete":
			if random.random() > threshold:
				while True:
					output = current_val+random.choice([-1,1])
					if output > self.min and output < self.max:
						return output

		if self.type == "catagorical":
			if random.random() > threshold:
				while True:
					output = self.catagories.index(current_val) + random.choice([-1,1])
					if output > 0 and output < len(self.catagories)-1:
						return self.catagories[output]
		return current_val

	def generate_rand_val(self):
		if self.type == "continuous" or self.type == "discrete":
			if self.type == "discrete":
				is_discrete = True
			else:
				is_discrete = False

			if self.scale == "linear":
				output = self.__generate_linear_rand_val(is_discrete)
			elif self.scale == "log":
				output = self.__generate_log_rand_val(is_discrete)
			elif self.scale == "po2":
				output = self.__generate_po2_rand_val()

		elif self.type == "catagorical":
			output = self.__generate_catagorical_rand_val()

		self.last_generated_result = output

		self.output_memory.append(self.last_generated_result)
		return output

	def get_requirements(self, parameter = None):
		if parameter != None:
			return self.requires[parameter]
		else:
			return self.requires[self.last_generated_result]


	def __generate_linear_rand_val(self,dis_val):
		if dis_val == True:
			return random.randint(self.min, self.max)
		else:
			return random.uniform(self.min, self.max)

	def __generate_log_rand_val(self,dis_val):
		oom_max = orderOfMagnitude(self.max)
		oom_min = orderOfMagnitude(self.min)
		if dis_val == "False":
			return random.random()*10**random.randint(oom_min,oom_max)
		else:
			return 1*10**random.randint(oom_min,oom_max)

	def __generate_po2_rand_val(self):
		oom_max = orderOfMagnitudePo2(self.max)
		oom_min = orderOfMagnitudePo2(self.min)
		return 2**random.randint(oom_min,oom_max)

	def __generate_catagorical_rand_val(self):
		return self.catagories[random.randrange(len(self.catagories) -1)]

optimizer_requirements ={

	"Adadelta":	["lr", "weight_decay"],
	"Adagrad":	["lr", "lr_decay", "weight_decay"],
	"Adam":	["lr", "weight_decay"],
	"AdamW":	["lr","weight_decay"],
	"SparseAdam":	["lr"],
	"Adamax":	["lr", "weight_decay"],
	"ASGD":	["lr", "lambd", "weight_decay"],
	"RMSprop":	["lr", "weight_decay", "momentum"],
	"Rprop":	["lr"],
	"SGD":	["lr", "momentum", "weight_decay"]
}

activation_parameters= [	
	"ELU",
	"Hardshrink",
	"Hardsigmoid",
	"Hardtanh",
	"Hardswish",
	"LeakyReLU",
	"LogSigmoid",
	"PReLU",
	"ReLU",
	"ReLU6",
	"RReLU",
	"SELU",
	"CELU",
	"GELU",
	"GELU",
	"SiLU",
	"Softplus",
	"Softshrink",
	"Softsign",
	"Tanh",
	"Tanhshrink",
	"Threshold"]

optimiser_parameters =[
	"Adadelta",
	"Adagrad",
	"Adam",
	"AdamW",
	"Adamax",
	"ASGD",
	"RMSprop",
	"Rprop",
	"SGD"]


num_layers = Parameter("discrete","linear",2,5)
hidden_layer_size = Parameter("discrete","linear",2,40)
activation_function = Parameter("catagorical",activation_parameters)
dropout = Parameter("continuous","linear",0,0.5)
optimizer = Parameter("catagorical",optimiser_parameters, requirements = optimizer_requirements)
lr = Parameter("continuous", "log",1e-6,5e-1)
lr_decay = Parameter("continuous", "log",1e-10,5e-2)
weight_decay = Parameter("continuous", "log",1e-10,5e-1)
lambd = Parameter("continuous", "log",1e-8,5e-2)
momentum = Parameter("continuous", "log",1e-8,5e-2)
batch_size = Parameter("discrete","po2",1,8)
epochs = Parameter("discrete","linear",5,20)



def build_random_config():

	configuration = dict()

	#####Build model layers#####
	n_layers = num_layers.generate_rand_val()
	layers = []
	while len(layers)< n_layers:

		layers.append(
			[hidden_layer_size.generate_rand_val(),
			activation_function.generate_rand_val(),
			dropout.generate_rand_val()])

	configuration["layers"] = layers


	#####Build optimizer config#####
	configuration["optimizer"] = [optimizer.generate_rand_val()]

	optimiser_requirements = optimizer.get_requirements()
	optimizer_parameters = []
	if "lr" in optimiser_requirements:
		optimizer_parameters.append(lr.generate_rand_val())
	if "weight_decay" in optimiser_requirements:
		optimizer_parameters.append(weight_decay.generate_rand_val())
	if "lr_decay" in optimiser_requirements:
		optimizer_parameters.append(lr_decay.generate_rand_val())
	if "lambd" in optimiser_requirements:
		optimizer_parameters.append(lambd.generate_rand_val())
	if "momentum" in optimiser_requirements:
		optimizer_parameters.append(momentum.generate_rand_val())

	configuration["optimizer"].append(optimizer_parameters)

	configuration["batch_size"] = batch_size.generate_rand_val()
	configuration["epochs"] = epochs.generate_rand_val()

	return configuration




"""
optimiser_hyperparams = {
"lr" : ["lr","log",-6,-1],
"weight_decay" : ["weight_decay","log",1e-10,0.1],
"lr_decay" : ["lr_decay","linear",1e-8,1e-3],
"lambd" : ["lambd","linear",1e-8,1e-3],
"momentum" : ["momentum","linear",1e-8,1e-3]
}




layer_parameters =	[["hidden_layer_size","linear",2,30],
					["activation_function","catagorical",activationparams],
					["dropout","linear",0,1]]


layers = Tree("layers",types = "linear",min_val=1,max_val = 10)
layers.add_children(layer_parameters)

optimizer = Tree("optimiser",types = "node")
optimizer.add_children(optimiser_parameters)


for child in optimizer.children:
	args = []
	hyperparameter_names = optimizer_hyperparams_index[child.var_name]

	for name in hyperparameter_names:
		args.append(optimiser_hyperparams[name])
	child.add_children(args)


optimizer.traverse()
layers.traverse()
activationparams= [	
	"ELU",
	"Hardshrink",
	"Hardsigmoid",
	"Hardtanh",
	"Hardswish",
	"LeakyReLU",
	"LogSigmoid",
	"PReLU",
	"ReLU",
	"ReLU6",
	"RReLU",
	"SELU",
	"CELU",
	"GELU",
	"GELU",
	"SiLU",
	"Softplus",
	"Softshrink",
	"Softsign",
	"Tanh",
	"Tanhshrink",
	"Threshold"]


layerparams = {
	#name	: min, max, scale
	"hidden": [2,30,"linear"],
	"activationparams":activationparams,
	"dropout": [0,1,"linear"]
}







configparams = {
	
	"layerparams": layerparams,
	"optimizerparams":optimiser_hyperparams_index



}



config_space = Tree("config_space", "root")
config_space.add_children()



def select_linear_value(args: list):
	min_val = args[0]
	max_val = args[1]
	return radnom.randint(min_val, max_val)

def select_catagorical_value(values: list):

	return values[random.randint(len(values))]

def select_log_value(args: list):
	min_val = args[0]
	max_val = args[1]																																
	return radnom.randint(min_val, max_val)

#Select optiser
#get parameters
#select values for parameters 


def select_value(config_space: dict):

	for param in config_space:
		config_space[param]
"""
#97.5?

def mutate_current_config(configuration):



	#####Build model layers#####
	for layer in configuration["layers"]:
		layer[0] = hidden_layer_size.mutate_val(layer[0])
		layer[1] = activation_function.mutate_val(layer[1])
		layer[2] = dropout.mutate_val(layer[2])


	#####Build Optimizer Config#####
	optimiser_requirements = optimizer.get_requirements(configuration["optimizer"][0])
	requrements= 0

	if "lr" in optimiser_requirements:
		configuration["optimizer"][1][requrements] = lr.mutate_val(configuration["optimizer"][1][requrements])
		requrements +=1

	if "weight_decay" in optimiser_requirements:
		configuration["optimizer"][1][requrements] = weight_decay.mutate_val(configuration["optimizer"][1][requrements])
		requrements +=1

	if "lr_decay" in optimiser_requirements:
		configuration["optimizer"][1][requrements] = lr_decay.mutate_val(configuration["optimizer"][1][requrements])
		requrements +=1

	if "lambd" in optimiser_requirements:
		configuration["optimizer"][1][requrements] = lambd.mutate_val(configuration["optimizer"][1][requrements])
		requrements +=1

	if "momentum" in optimiser_requirements:
		configuration["optimizer"][1][requrements] = momentum.mutate_val(configuration["optimizer"][1][requrements])
		requrements +=1

	configuration["batch_size"] = batch_size.generate_rand_val()
	configuration["epochs"] = epochs.mutate_val()

	return configuration

"""
def find_num_k(search_size):
	import train_function_find_k_best
	##Generate Random Configs
	configs = []

	while len(configs) < search_size:
		configs.append(build_random_config()) 
	##Get Evaluations
	ave = [[] for x in range(search_size)]
	min_val = []
	max_val = []
	print(len(ave))
	runs = 0
	while runs < 3:

		results = Dispatcher(configs, 4,train_function_find_k_best)
		counter = 0
		for result in results:
			x = result[1]
			mean = []
			for i in result[0]:
				mean.append(np.mean(np.array(i)))
			ave[counter].append(mean)
			counter +=1
		runs += 1


	print(ave)
	for itera in ave:
		mean = []
		min_val = []
		max_val = []
		for i in itera: 
			mean.append(np.mean(i))
			min_val.append(min(i))
			max_val.append(max(i))
		print(mean)
		plt.plot(x,mean)

		plt.fill_between(x,min_val, max_val, alpha=0.4)

	plt.savefig("num_k.png",dpi = 120)
	return results
"""
def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def find_num_k(search_size: int):
	import train_function_find_k_best
	##Generate Random Configs
	configs = []
	while len(configs) < search_size:
		configs.append(build_random_config())

	#Add duplicates of the same configurations
	for i in configs:
		configs.append(i)
		if len(configs) > 16*search_size -1:
			break

	##Get Evaluations
	ave = [[] for f in range(search_size)]
	min_val = []
	max_val = []
	runs = 0


	#Run training loop
	results = Dispatcher(configs, 4,train_function_find_k_best)
	print(len(results))
	#Ordering results back 
	model_assigned_results = dict()
	padding = 0 #Makes sure that results from the same model are in the same list
	for key,value in enumerate(results):

		print(str(key),padding)
		if padding == 0:
			model_assigned_results[str(key-padding)] = [value[0]]
			x = value[1]
			print("Creating list for: ",str(key-padding))
		else:
			model_assigned_results[str(key-padding)].append(value[0])

		print(key % (search_size-1))
		if (key+1) % (search_size) == 0 and key != 0:
			padding +=search_size

	cmap = get_cmap(len(model_assigned_results))
	std_list_list = []
	save_list = []
	for c, i in enumerate(model_assigned_results):
		model_assigned_results[i] = np.array(model_assigned_results[i]).T
		print(model_assigned_results[i])
		std_list = []
		save_list.append(model_assigned_results[i])
		for elem in model_assigned_results[i]:
			std_list.append(np.std(elem))
		std_list_list.append(std_list)
		
		plt.plot(x,std_list,alpha = 0.2)
	print(std_list_list)
	
	std_list_mean = np.mean(std_list_list, axis = 0)
	std_list_std = np.std(std_list_list, axis = 0)
	plt.plot(x,std_list_mean, ls = ':',c = "y")
	plt.fill_between(x,std_list_mean-std_list_std,std_list_mean+std_list_std,color= "y", alpha=0.2)
	plt.savefig("num_k.png",dpi = 120)
	np.save("k_fold_data.csv", np.array(save_list))
	exit()
	counter = 0
	for result in results:
		ave[counter].append(result[0])
		counter +=1
		if counter == search_size:
			counter = 0
		x = result[2]
	runs += 1

	print(ave)
	cmap = get_cmap(len(ave)+10)
	output = {}
	all_ranges = []
	for i, data in enumerate(ave):
		print(i,": \n")

		means = []
		min_val = []
		max_val = []
		ranges = []
		for index in range(0,len(data[0])):
			hold = []
			print(index)
			for mean in data[index]:
				hold.append(mean)
			means.append(np.mean(np.array(hold)))
			min_val.append(min(hold))
			max_val.append(max(hold))
			ranges.append(max(hold) - min(hold))
			percentage_range = np.divide(ranges,means)
		all_ranges.append(ranges)
		
		#plt.fill_between(x,min_val,max_val,color = cmap(i), alpha=0.2)
	print(all_ranges)
	range_mean = np.mean(np.array(all_ranges),axis =0)
	print(range_mean)
 
	plt.plot(x,range_mean,c = cmap(i))
	plt.savefig("num_k.png",dpi = 1200)
	return results

search_results = []

c={'layers': [[30, 'Hardswish', 0.5256768563219513], [32, 'Hardtanh', 0.35519114598750734]], 'optimizer': ['Adagrad', [0.001, 1e-06, 1e-05]], 'batch_size': 8, 'epochs': 10}

#92.5 average
config = {'layers': [[23, 'CELU', 0.1691787927580818], [12, 'ReLU', 0.0731419482764655]], 'optimizer': ['Adam', [0.04561540502013087, 1.374454935409825e-07]], 'batch_size': 8, 'epochs': 10}


config1 = {'layers': [[17, 'GELU', 0.21779013328498015], [14, 'ReLU', 0.4892961772364511], [25, 'SELU', 0.07225233809260545]], 'optimizer': ['RMSprop', [0.1, 1e-10, 1e-05]], 'batch_size': 2, 'epochs': 10}
config2 = {'layers': [[27, 'Softsign', 0.3265761074171735], [21, 'Hardswish', 0.23652035448519482]], 'optimizer': ['Adagrad', [0.07792755208530526, 1.1842682768259104e-06, 0.012936283637877558]], 'batch_size': 4, 'epochs': 10}

#97.4
config = {'layers': [[13, 'GELU', 0.2573009857036789], [35, 'RReLU', 0.099609868730452]], 'optimizer': ['RMSprop', [0.01, 0.001, 1.1824505395337446e-06]], 'batch_size': 4, 'epochs': 5}

#config = {'layers': [[13, 'GELU', 0.31270262078514766], [35, 'ReLU6', 0.099609868730452]], 'optimizer': ['RMSprop', [0.009103534983639966, 0.0009596927819293694, 1.1824505395337446e-06]], 'batch_size': 8, 'epochs': 10}
search_size = 1
configs = []
max_runs = 20
elite_percentage = 0.3

while len(configs) < search_size:
	configs.append(build_random_config()) 
start_time = time.time()

current_run = 0
best = [0]*int(search_size*elite_percentage)
best_hp = [0]*int(search_size*elite_percentage)
best_recall = [0]*int(search_size*elite_percentage)

while current_run < max_runs:
	for i in configs:
		print(i)
	results = Dispatcher(configs, 4)
	for result in results:
		#print("Hyperparameters: \n",result[0])
		print("\nAverage Acc: ",result[1])
		print("Std Acc: ",result[2])
		print("Recall: ", result[-1])
		if result[1] > min(best):
			best[best.index(min(best))] = result[1]
			best_recall[best.index(min(best))] = result[-1]
			best_hp[best.index(min(best))] = result[0]
	print(len(best_hp))
	search_results.append([best,best_hp])
	configs = best_hp

	while len(configs) < search_size:
		configs.append(mutate_current_config(configs[random.randint(0,len(best_hp)-1)]))
	current_run+=1
for score, hp,recall in zip(best,best_hp,best_recall):
	print("accuracy: ", score)
	print("recall: ", recall)
	print("hyperparameters: ", hp)
print('Execution time: ', (time.time() - start_time), ' seconds')


