import time
import torch
import argparse
from pre_process import *
from utils import *
from exp import Experiments
from utils import print_local_time, set_seed


parser = argparse.ArgumentParser()  

parser.add_argument('--dataset', type=str, default='environment', help='dataset') 
parser.add_argument('--pre_train', type=str, default="bert", help='Pre_trained model')
parser.add_argument('--hidden', type=int, default=64, help='dimension of hidden layers in MLP')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--norm',type=bool, default=False, help="Normalise [CLS] vectors")
parser.add_argument('--wandb',type=int, default=0, help="Enable wandb logging")
parser.add_argument('--word',type=bool, default=False, help="Word [SEP] Definition instead of Word: Definition")
parser.add_argument('--pooled',type=bool, default=False, help="Use pooled output instead of [CLS] token")
parser.add_argument('--mixture',type=str, default=None, help="Type of weighting in mixture model")
parser.add_argument('--padmaxlen', type=int, default=128, help='max length of padding')
parser.add_argument('--unitary', type=int, default=0, help='unitary trace')
parser.add_argument('--score', type=str, default="trace", help="Which scoring function to use: trace or obs")

## Training hyper-parameters
parser.add_argument('--expID', type=int, default=0, help='-th of experiments')
parser.add_argument('--epochs', type=int, default=100, help='training epochs')
parser.add_argument('--batch_size', type=int, default=128, help='training batch size')
parser.add_argument('--lr', type=float, default=2e-5, help='learning rate for pre-trained model')
# parser.add_argument('--lr_projection', type=float, default=1e-3, help='learning rate for projection layers')
parser.add_argument('--eps', type=float, default=1e-8, help='adamw_epsilon')
parser.add_argument('--optim', type=str, default="adamw", help='Optimizer')

## Others
parser.add_argument('--cuda', type=bool, default=True, help='use cuda for training')
parser.add_argument('--gpu_id', type=int, default=0, help='which gpu')

start_time = time.time()
print ("Start time at : ")
print_local_time()

args = parser.parse_args()
args.cuda = True if torch.cuda.is_available() and args.cuda else False
if args.cuda:
    torch.cuda.set_device(args.gpu_id)

print(args)

set_seed(42)

create_data(args)

exp = Experiments(args)

"""Train the model"""
exp.train()
exp.predict(tag="test")
# exp.save_prediction()

print ("Time used :{:.01f}s".format(time.time()-start_time))
print ("End time at : ")
print_local_time()
print ("************END***************")