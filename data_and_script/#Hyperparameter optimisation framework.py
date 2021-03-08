#Hyperparameter optimisation framework
import np
import 
#Requirements

	#Dispatch n workers to train a model with a set of hyperparameters for a number of iterations t.

	#Information to worker can process:
							Hyperparameters:
								model:
									topology connections
								Optimiser:
									lr
									reg
									args
							Weights*
							Epochs

							Data
							ID


	#Return information from trained model:
					#		Hyperparameters
					#		Model Weights
					#		Training Loss
					#		Training time
					#		ID


#Hyperparameters
#dict = 
{
	"layers": 
	[
	[hidden_dim,act_func,dropout],
	[hidden_dim,act_func,dropout],
	...
	],
	"optimizer":["Adam",args]
	"training":[epochs,batch_size]
}



def random_search(n_configs: int):
	configs = []

	while len(configs < n_configs):









def Pool_Arg_Packer(Configurations, data):
	packed_args = []
    
    
	for config in Configurations:
        packed_args.append(GenerateID(),Configurations[config], data)


def Pool_Arg_unpacker(packed_args):
	ID = packed_args[0]
	config = packed_args[1]
	data = packed_args[2]

	return ID, config, data

def Dispatcher(Configurations: dict, num_workers: int, data) -> dict:
	"""
		returns a dictionary "ID":[hyperparameters: dict, Model_weights: dict, Training Loss: list[float], Training Time: int]	
	"""
		Pool_Arg_Packer(Configurations, data, weights)
		with Pool(processes=num_workers) as pool:
			results = pool.starmap(Model_training,Configurations)
			pool.close()
			pool.join()


	return results


#Pack and unpack hyperparameters

def unpack_configuration(config):



def Receiver(packed_args)
	ID, configuration, data = Pool_Arg_unpacker(packed_args)



def Trainer()
	for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        train_epoch( X, y, model, optimizer, loss_criterion)


def train_epoch( X, y, model, optimizer, loss_criterion):
	for samples,labels in zip(X, y):

		# zero the parameter gradients
		optimizer.zero_grad()
		# forward + backward + optimize
		outputs = model(samples)
		#Managing data format
		outputs = outputs.float()
		labels = labels.float()
		labels = labels.unsqueeze(-1)

		loss = loss_criterion(outputs, labels)
		loss.backward()
		optimizer.step()
		running_loss += loss.item()