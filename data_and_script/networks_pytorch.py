import torch.nn as nn
import torch

class FFModel(nn.Module):
	def __init__(self, input_dim, hidden_dim_l1,hidden_dim_l2, layer_dim, 
		output_dim,seq,l1_dropout,l2_dropout,batch_size = 100):
		super().__init__()

		self.hidden_layers = []
		self.batch_size = batch_size
		self.seq_len = seq
		self.input_layer = nn.Linear(input_dim, hidden_dim_l1)
		self.layer2 = nn.Linear(hidden_dim_l1,hidden_dim_l2)
		self.act = nn.ReLU()
		self.dropout_1 = nn.Dropout(l1_dropout)
		self.dropout_2 = nn.Dropout(l2_dropout)

		for i in range(layer_dim-1):
			self.hidden_layers.append(nn.Linear(hidden_dim_l2,hidden_dim_l2))



		# Readout layer
		if output_dim <= 2:
			self.fc = (nn.Linear(hidden_dim_l2,1))
			self.fcact = nn.Sigmoid()
		else:
			self.fc = (nn.Linear(hidden_dim_l2,output_dim))
			self.fcact = nn.Softmax()

	def init_hidden(self):
		# Initialize hidden state with zeros
		self.h0 = torch.zeros(self.layer_dim,self.batch_size, self.hidden_dim).requires_grad_().cuda()

		# Initialize cell state
		self.c0 = torch.zeros(self.layer_dim, self.batch_size, self.hidden_dim).requires_grad_().cuda()

	def forward(self, x):

		out = self.input_layer(x)
		out = self.act(out)
		out = self.dropout_1(out)
		out = self.layer2(out)
		out = self.act(out)
		out = self.dropout_1(out)
		for layer in self.hidden_layers:
			out = layer(out)
			out = self.act(out)
			out = self.dropout_2(out)
		out = self.fcact(self.fc(out))

		return out



def build_sequential(n_inputs, n_classes, n_neurons_l1, n_neurons_l2, layers, learning_rate,
                     input_dropout, l1_dropout, l2_dropout, l1_reg, l2_reg,batch_size):
    model = FFModel(input_dim = n_inputs, hidden_dim_l1 = n_neurons_l1,
    	hidden_dim_l2 = n_neurons_l2, layer_dim = layers, output_dim = n_classes,
    	seq = 1,l1_dropout = l1_dropout, l2_dropout = l2_dropout,batch_size = batch_size)
    
    # Add the input layer and the first hidden layer
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

    if n_classes <= 2:
    	loss = nn.BCELoss()
    else:
    	loss = nn.CrossEntropyLoss()



    return model, loss, optimizer


def build_sequential_from_dict(n_inputs, n_classes, params):
    n_neurons_l1 = params['n_neurons_l1']
    n_neurons_l2 = params['n_neurons_l2']
    layers = params['layers']
    learning_rate = params['learning_rate']
    l1_dropout = params['l1_dropout']
    l2_dropout = params['l2_dropout']
    l1_reg = params['l1_reg']
    l2_reg = params['l2_reg']
    return build_sequential(n_inputs, n_classes, n_neurons_l1, n_neurons_l2, layers, learning_rate, 
                            0.0, l1_dropout, l2_dropout,batch_size, l1_reg, l2_reg)
    



