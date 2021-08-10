import numpy as np
def generate_cell(op_dict, input_dict, output_op_idx):
  cell_array = np.zeros((len(op_dict.keys())+2,len(op_dict.keys())+2))
  ops = [0]*len(op_dict.keys()) 
  ops.insert(0, "INPUT")
  ops.append("OUTPUT")
  # [op_idx, input_idx]
  cell_array[output_op_idx,output_op_idx +1] = 1
  for op_idx in op_dict:
      op = op_dict[op_idx]
      ops[op_idx] = op 
      for input_idx in input_dict[op_idx]:
        cell_array[op_idx,input_idx] = 1
  return cell_array, ops



def encode(hyperparameters):
  op_dict = {} 
  input_dict = {1:[],2:[],3:[],4:[],5:[],6:[],7:[]}
  for i in hyperparameters: 
    if "normal_cell" in i:
      if "type" in i:
        op_dict[int(i[-6])] = hyperparameters[i]
      if "input" in i:
        input_dict[int(i[-9])].append(hyperparameters[i])

  for i in input_dict:
    input_dict[i] = set(input_dict[i])
        
  output_op = max(op_dict.keys())
  
  cell_array, ops = generate_cell()
  # Query an Inception-like cell from the dataset.
  # cell = api.ModelSpec(
  #   matrix=[[0, 1, 1, 1, 0, 1, 0],    # input layer
  #           [0, 0, 0, 0, 0, 0, 1],    # 1x1 conv
  #           [0, 0, 0, 0, 0, 0, 1],    # 3x3 conv
  #           [0, 0, 0, 0, 1, 0, 0],    # 5x5 conv (replaced by two 3x3's)
  #           [0, 0, 0, 0, 0, 0, 1],    # 5x5 conv (replaced by two 3x3's)
  #           [0, 0, 0, 0, 0, 0, 1],    # 3x3 max-pool
  #           [0, 0, 0, 0, 0, 0, 0]],   # output layer
  #   # Operations at the vertices of the module, matches order of matrix.
  # 
  # Querying multiple times may yield different results. Each cell is evaluated 3
  # times at each epoch budget and querying will sample one randomly.
  cell = api.ModelSpec(cell_array, ops)
  return cell
