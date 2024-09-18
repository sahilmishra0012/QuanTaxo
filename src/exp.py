import os
import time
import numpy as np
import pickle as pkl
import torch
import sys
from tqdm import tqdm 
from torch import optim
from transformers import BertTokenizer
from utils import *
from data import *
from model import BoxEmbed

import wandb

class Experiments(object):

    def __init__(self,args):
        super(Experiments,self).__init__()
        
        self.args = args
        self.tokenizer = self.__load_tokenizer__()
        self.train_loader,self.train_set = load_data(self.args, self.tokenizer,"train")
        self.test_loader,self.test_set = load_data(self.args, self.tokenizer,"test")
        
        self.model = BoxEmbed(args,self.tokenizer)
        self.optimizer = self._select_optimizer()
        self._set_device()
        # self.exp_setting= str(self.args.pre_train)+"_"+str(self.args.dataset)+"_"+str(self.args.expID)+"_"+str(self.args.epochs)+"_"+str(self.args.batch_size)
        self.exp_setting= "_".join([str(elem) for elem in [self.args.pre_train,self.args.dataset,self.args.expID,self.args.epochs,self.args.batch_size,self.args.mixture if self.args.mixture else "superposn", self.args.lr]])
        
        setting={
            "pre_train":self.args.pre_train,
            "dataset":self.args.dataset,
            "expID":self.args.expID,
            "epochs":self.args.epochs,
            "batch_size":self.args.batch_size,
            "norm_CLS":self.args.norm,
            "lr":self.args.lr,
            "hidden":self.args.hidden,
            "[SEP]":self.args.word,
            "pooled":self.args.pooled,
            "mixture":self.args.mixture if self.args.mixture else "superposn",
            "unit_trace":self.args.unitary,
        }
        print(setting)

        if self.args.wandb:
            wandb.init(project='quantum',config = setting,entity='taxo_iitd')


    def __load_tokenizer__(self):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        print("Tokenizer Loaded!")
        return tokenizer


    def _select_optimizer(self):
        parameters = [{"params": [p for n, p in self.model.named_parameters()],
                "weight_decay": 0.0},]

        if self.args.optim=="adam":
            optimizer = optim.Adam(parameters, lr=self.args.lr)
        elif self.args.optim=="adamw":
            optimizer = optim.AdamW(parameters,lr=self.args.lr, eps=self.args.eps)

        return optimizer
    
    def _set_device(self):
        if self.args.cuda:
            self.model = self.model.cuda()


    def train_one_step(self,it,encode_parent, encode_child,encode_negative_parents):

        self.model.train()
        self.optimizer.zero_grad()

        loss = self.model(encode_parent, encode_child,encode_negative_parents, flag=self.args.score)
        loss.backward()
        self.optimizer.step()

        return loss


    def train(self):
        time_tracker = []
        test_acc = test_mrr = test_wu_p = 0
        old_test_acc = old_test_mrr = old_test_wu_p = 0

        limit=1
        if(self.args.dataset=='environment'):
            limit = 0.46
        elif self.args.dataset=='science':
            limit = 0.52
        
        for epoch in tqdm(range(self.args.epochs)):
            epoch_time = time.time()
            train_loss = []

            for i, (encode_parent,encode_child,encode_negative_parents) in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
                loss = self.train_one_step(it=i,encode_parent=encode_parent,encode_child=encode_child,encode_negative_parents=encode_negative_parents)
                train_loss.append(loss.item())

            train_loss = np.average(train_loss)
            test_metrics = self.predict()
            test_acc = test_metrics["Acc"]
            test_mrr = test_metrics["MRR"]
            test_wu_p = test_metrics["Wu"]
            if(test_acc>old_test_acc or (test_acc==old_test_acc and (old_test_mrr<=test_mrr or old_test_wu_p<=test_wu_p))):
                # Save the best performing model
                torch.save(self.model.state_dict(), os.path.join("../result",self.args.dataset,"model","exp_model_"+self.exp_setting+".checkpoint"))
                old_test_acc = test_acc
                old_test_mrr = test_mrr
                old_test_wu_p = test_wu_p
            time_tracker.append(time.time()-epoch_time)

            print('\nEpoch: {:04d}'.format(epoch + 1),
                'train_loss:{:.05f}'.format(train_loss),
                'acc:{:.05f}'.format(test_acc),
                'mrr:{:.05f}'.format(test_mrr),
                'wu_p:{:.05f}'.format(test_wu_p),
                'mr:{:.05f}'.format(test_metrics["MR"]),
                'prec5:{:.05f}'.format(test_metrics["Prec@5"]),
                'prec10:{:.05f}'.format(test_metrics["Prec@10"]),
                'NDCG:{:.05f}'.format(test_metrics["NDCG"]),
                'epoch_time:{:.01f}s'.format(time.time()-epoch_time),
                'remain_time:{:.01f}s'.format(np.mean(time_tracker)*(self.args.epochs-(1+epoch))),
                )
            
            if(self.args.wandb):
                wandb.log({
                    'train_loss':(train_loss),
                    'acc':(test_acc),
                    'mrr':(test_mrr),
                    'wu_p':(test_wu_p),
                    'mr':(test_metrics["MR"]),
                    'prec5':(test_metrics["Prec@5"]),
                    'prec10':(test_metrics["Prec@10"]),
                    'NDCG':(test_metrics["NDCG"]),
                })

            if test_acc>=limit:
                break
        # Use the model in final epoch
        # torch.save(self.model.state_dict(), os.path.join("../result",self.args.dataset,"model","exp_model_"+self.exp_setting+".checkpoint"))            


    def predict(self, tag=None):
        print ("Prediction starting.....")
        if(tag=="test"):
            self.model.load_state_dict(torch.load(f"/home/avi/BoxTaxo_QLM/result/{self.args.dataset}/model/exp_model_{self.exp_setting}.checkpoint"))
        self.model.eval()
        with torch.no_grad():
            score_list = []
            gt_label = self.test_set.test_gt_id
            query = self.model.get_density_matrices(self.test_set.encode_query)
            candidates = []
            for j, (encode_candidate) in enumerate(self.test_loader):
                candidate_center = self.model.get_density_matrices(encode_candidate)
                candidates.append(candidate_center)
            candidates = torch.cat(candidates,0)
            num_query=len(query)
            num_candidate = len(candidates)
            for i in tqdm(range(num_query), desc="Validation Queries", total = num_query):
                query_duplicated = [query[i].unsqueeze(dim=0) for _ in range(num_candidate)]
                query_duplicated = torch.cat(query_duplicated,0)
                score = self.model.classfn_score(query_duplicated, candidates)
                score_list.extend([score])
            score_list = torch.stack(score_list,0)
            sorted_scores, indices = score_list.squeeze().sort(dim=1,descending=True)
            test_metrics = metrics(indices, gt_label, self.train_set.train_concept_set, self.test_set.path2root)

        if(tag=="test"):
            print('acc:{:.05f}'.format(test_metrics["Acc"]),
                'mrr:{:.05f}'.format(test_metrics["MRR"]),
                'wu_p:{:.05f}'.format(test_metrics["Wu"]),
                'mr:{:.05f}'.format(test_metrics["MR"]),
                'prec5:{:.05f}'.format(test_metrics["Prec@5"]),
                'prec10:{:.05f}'.format(test_metrics["Prec@10"]),
                'NDCG:{:.05f}'.format(test_metrics["NDCG"]),
                )
            return
        else:
            return test_metrics