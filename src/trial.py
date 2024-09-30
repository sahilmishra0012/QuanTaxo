import torch

# # Load the state_dict from the saved checkpoint
# state_dict = torch.load("/home/avi/BoxTaxo_QLM/result/environment/model/exp_model_bert_environment_0_100_128_False_False_None.checkpoint")

# # Print all keys (parameter names) and their shapes
# for param_tensor in state_dict:
#     print(f"Parameter name: {param_tensor} \tShape: {state_dict[param_tensor].size()}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()
print(torch.cuda.device_count())
print(torch.cuda.current_device())

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    # print('Cached:   ', round(torch.cuda.memory
