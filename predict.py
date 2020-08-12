import sys
import os
import numpy as np
import torch
from model import Net
import coremltools as ct
# print infinitely many elements
np.set_printoptions(threshold=np.inf,suppress=True, precision=3, floatmode="fixed")

py_model_existing = sys.argv[1]
#ml_model_non_quantized_existing = sys.argv[2]
ml_model_quantized_existing = sys.argv[2]
numpy_file = sys.argv[3]

py_model = Net(output_label_count=int(7))  # 1
load = torch.load(py_model_existing)
py_model.load_state_dict(load)  # 2

ml_model = ct.models.MLModel(ml_model_quantized_existing)

py_model.eval()  # 3


# torch conversions
numpy_arr = np.load(numpy_file)
random_input = torch.rand(1, 1, 98, 40)
random_input[0] = torch.from_numpy(numpy_arr)

# Make predictions
res1_prev = py_model.forward(random_input)  # 4

# numpy conversions
x = np.zeros([1, 1, 98, 40]) #coreml4
#x = np.zeros([1, 98, 40]) #coreml3
x[0] = numpy_arr
output = ml_model.predict({'input.1': x})
#output_1 = ml_model.predict({'input_1': x})

# Convert prediction result to numpy
res1 = res1_prev.detach().numpy()
#res2 = output_1['output_1']
res2 = output['158']

# Calculate difference
diff = res1-res2

print(
    f"\n Successfully predicted on model {py_model_existing} RESULT: \n  {res1} \n ")
print(
    f"\n Successfully predicted on model {ml_model_quantized_existing} RESULT: \n  {res2} \n ")
print(f"\n  DIFF: {diff}")
