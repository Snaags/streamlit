import torch.nn as nn
import torch.optim as opt

#activations = 
#{
#	"ELU": nn.ELU(),
#	"Hardshrink":nn.Hardshrink(),
#	"Hardsigmoid":nn.Hardsigmoid(),
#	"Hardtanh":nn.Hardtanh(),
#	"Hardswish":nn.Hardswish(),
#	"LeakyReLU":nn.LeakyReLU(),
#	"LogSigmoid":nn.LogSigmoid(),
#	"PReLU	":nn.PReLU	(),
#	"ReLU":nn.ReLU(),
#	"ReLU6":nn.ReLU6(),
#	"RReLU":nn.RReLU(),
#	"SELU":nn.SELU(),
#	"CELU":nn.CELU(),
#	"GELU":nn.GELU(),
#	"GELU":nn.GELU(),
#	"SiLU":nn.SiLU(),
#	"Softplus":nn.Softplus(),
#	"Softshrink":nn.Softshrink(),
#	"Softsign":nn.Softsign(),
#	"Tanh":nn.Tanh(),
#	"Tanhshrink":nn.Tanhshrink(),
#	"Threshold":nn.Threshold()
#
#
#	}
#




class Layer():
	def __init__(self, prev_dim, hidden_dim,act_layer,dropout):

		self.fc = nn.Linear(prev_dim, hidden_dim)


		if act_layer != None:
			if act_layer == "ELU" :
				self.act = nn.ELU()
			elif act_layer == "Hardshrink" :
				self.act = nn.Hardshrink()
			elif act_layer == "Hardsigmoid" :
				self.act = nn.Hardsigmoid()
			elif act_layer == "Hardtanh" :
				self.act = nn.Hardtanh()
			elif act_layer == "Hardswish" :
				self.act = nn.Hardswish()
			elif act_layer == "LeakyReLU" :
				self.act = nn.LeakyReLU()
			elif act_layer == "LogSigmoid" :
				self.act = nn.LogSigmoid()
			elif act_layer == "PReLU" :
				self.act = nn.PReLU	()
			elif act_layer == "ReLU" :
				self.act = nn.ReLU()
			elif act_layer == "ReLU6" :
				self.act = nn.ReLU6()
			elif act_layer == "RReLU" :
				self.act = nn.RReLU()
			elif act_layer == "SELU" :
				self.act = nn.SELU()
			elif act_layer == "CELU" :
				self.act = nn.CELU()
			elif act_layer == "GELU" :
				self.act = nn.GELU()
			elif act_layer == "GELU" :
				self.act = nn.GELU()
			elif act_layer == "SiLU" :
				self.act = nn.SiLU()
			elif act_layer == "Softplus" :
				self.act = nn.Softplus()
			elif act_layer == "Softshrink" :
				self.act = nn.Softshrink()
			elif act_layer == "Softsign" :
				self.act = nn.Softsign()
			elif act_layer == "Tanh" :
				self.act = nn.Tanh()
			elif act_layer == "Tanhshrink" :
				self.act = nn.Tanhshrink()
			elif act_layer == "Threshold" :
				self.act = nn.Threshold()
			else:
				self.act = nn.Identity
				print("Activation Layer Key Not Found: Set to Identity!!")
		else:
			self.act = nn.Identity


		if dropout != None:
			self.drop = nn.Dropout(dropout)
		else:
			self.drop = nn.Identity

	def process(self, x):
		x = self.fc(x)

		x = self.act(x)
		return self.drop(x)



class FeedForwardModel(nn.Module):
	def __init__(self, input_dim, output_dim, layers,batch_size = 4):
		super().__init__()

		"""
		layers = 
		[
		[hidden, act, dropout],
		[hidden, act, dropout],
		[hidden, act, dropout]
		]

		"""

		#Define in terms of nodes
			#Each node has a type, a size and a set of input and output connections
			#Need a way of combining outputs to fit next layer


		#List with each element representing one layer. These will be sequencially processed
		#with the output being the input for the next layer

		self.layers = []
		self.prev_dim = input_dim
		for args in layers:
			self.layers.append(Layer(self.prev_dim,args[0],args[1],args[2]))
			self.prev_dim = args[0]

		self.batch_size = batch_size

		# Readout layer
		if output_dim <= 2:
			self.fc = (nn.Linear(self.prev_dim,1))
			self.fcact = nn.Sigmoid()
		else:
			self.fc = (nn.Linear(self.prev_dim,output_dim))
			self.fcact = nn.Softmax()

	def forward(self, x):

		#Preset Input Layer
		out = x

		for i in self.layers:
			out = i.process(out)

		#Preset OutputLayer
		out = self.fcact(self.fc(out))

		return out


#optimisers = 
#	{
#	"Adadelta":opt.Adadelta(params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0),
#	"Adagrad":opt.Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10),
#	"Adam":opt.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False),
#	"AdamW":opt.AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False),
#	"SparseAdam":opt.SparseAdam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08),
#	"Adamax":opt.Adamax(params, lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0),
#	"ASGD":opt.ASGD(params, lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0),
#	"RMSprop":opt.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False),
#	"Rprop":opt.Rprop(params, lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50)),
#	"SGD":opt.SGD(params, lr=<required parameter>, momentum=0, dampening=0, weight_decay=0, nesterov=False)
#	}
#

def build_optimiser(model_parameters,optimiser, args):


	if optimiser == "Adadelta":
		output_optimiser = opt.Adadelta(model_parameters,lr = args[0], weight_decay = args[1])
	if optimiser == "Adagrad":
		output_optimiser = opt.Adagrad(model_parameters,lr = args[0], weight_decay = args[1], lr_decay= args[2])
	if optimiser == "Adam":
		output_optimiser = opt.Adam(model_parameters,lr = args[0], weight_decay= args[1])
	if optimiser == "AdamW":
		output_optimiser = opt.AdamW(model_parameters,lr = args[0],weight_decay= args[1])
	if optimiser == "SparseAdam":
		output_optimiser = opt.SparseAdam(model_parameters,lr = args[0])
	if optimiser == "Adamax":
		output_optimiser = opt.Adamax(model_parameters,lr = args[0], weight_decay= args[1])
	if optimiser == "ASGD":
		output_optimiser = opt.ASGD(model_parameters,lr = args[0], weight_decay= args[1], lambd = args[2])
	if optimiser == "RMSprop":
		output_optimiser = opt.RMSprop(model_parameters,lr = args[0], weight_decay= args[1], momentum = args[2])
	if optimiser == "Rprop":
		output_optimiser = opt.Rprop(model_parameters,lr = args[0])
	if optimiser == "SGD":
		output_optimiser = opt.SGD(model_parameters,lr = args[0], weight_decay = args[1], momentum = args[2])

	return output_optimiser

def build_model(hyperparameters, n_features, n_classes):

	model = FeedForwardModel(n_features, n_classes, hyperparameters)

	return model


def init_system(hyperparameter_package, n_features, n_classes):

	model = build_model(hyperparameter_package["layers"],n_features,n_classes)
	optim = build_optimiser(model.parameters(), *hyperparameter_package["optimizer"])

	return model, optim