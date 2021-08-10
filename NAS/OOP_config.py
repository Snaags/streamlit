import ConfigSpace as CS 
import ConfigSpace.hyperparameters as CSH
import os
import csv
import time 
from ConfigStruct import Parameter, Cumulative_Integer_Struct, LTP_Parameter 

"""	TODO
Seperate Pooling and Convolution Layers
Add more convolution operations (kernalSize and maybe stride)
"""

def init_config():

  cs = CS.ConfigurationSpace()

     
  conv_ops=["CONV1X1", "CONV3X3", "MAXPOOL3X3"])
  ops_parameters = [
        Parameter("type",               "Categorical", lower_or_constant_value = conv_ops ), 
        LTP_Parameter("input_1",               "Integer", 0,10),
        LTP_Parameter("input_2",               "Integer", 0,10)]
  ops = Cumulative_Integer_Struct(cs,ops_parameters,"ops","num_ops","Integer",1,7)


  conv_parameters = [
        ops]
 
  Cumulative_Integer_Struct(cs,conv_parameters,"normal_cell", "num_conv","Integer", 1, 1).init() 

  conv_ops = ["FactorizedReduce"]
  
  ops_parameters = [
        Parameter("type",               "Categorical", lower_or_constant_value = conv_ops ), 
        LTP_Parameter("input_1",               "Integer", 0,10),
        LTP_Parameter("input_2",               "Integer", 0,10)]
  ops = Cumulative_Integer_Struct(cs,ops_parameters,"ops","num_ops","Integer",1,1)


  conv_parameters = [
        ops]
 
  Cumulative_Integer_Struct(cs,conv_parameters,"reduction_cell", "num_re","Integer", 1, 1).init() 
  """
  ops_type_list = ["StdConv"]
  
  ops_parameters = [
        Parameter("type",               "Categorical", lower_or_constant_value = ops_type_list ), 
        LTP_Parameter("input",               "Integer", 0,10)]
  ops = Cumulative_Integer_Struct(cs,ops_parameters,"ops","num_ops","Integer",1,5)


  conv_parameters = [
        Parameter("type",               "Constant", lower_or_constant_value = "Conv1D"),
        Parameter("padding",            "Constant" ,lower_or_constant_value = "same"),
        Parameter("filters",            "Constant", lower_or_constant_value =  1),
        Parameter("BatchNormalization", "Integer", 0,1),
        Parameter("kernel_size",        "Integer", 1,16),
        Parameter("kernel_regularizer", "Float", 1e-8,5e-1, log = True),
        Parameter("bias_regularizer",   "Float", 1e-8,5e-1, log = True),
        Parameter("activity_regularizer", "Float", 1e-8,5e-1, log = True),
        ops]
 
  Cumulative_Integer_Struct(cs,conv_parameters,"cell", "num_cells","Integer", 1, 5).init() 


    
  dense_parameters = [
        Parameter("type",               "Constant", "Dense"),
        Parameter("units",              "Integer", 1,128),
        Parameter("kernel_regularizer", "Float", 1e-8,5e-1, log = True),
        Parameter("bias_regularizer",   "Float", 1e-8,5e-1, log = True),
        Parameter("activity_regularizer", "Float", 1e-8,5e-1, log = True)]
     
  Cumulative_Integer_Struct(cs,dense_parameters,"dense","num_dense_layers","Integer", 1, 3).init() 
  """
    ###Training Configuration###
    ###Optimiser###
  lr =CSH.Constant(name = "lr",			value = 0.001)
  p =CSH.Constant(name = "p",			value = 0.05 )
  window_size = CSH.Constant(name = "window_size", value = 200)
  channels = CSH.UniformIntegerHyperparameter(name = "channels", lower = 1 ,upper = 52)
  layers = CSH.Constant(name = "layers", value = 4)
    ###Topology Definition]###

  hp_list = [
        window_size,
        channels,
        lr,
        p,
        layers]
  cs.add_hyperparameters(hp_list)
  return cs

if __name__ == "__main__":
	configS = init_config()	
	print(configS.get_hyperparameters())
