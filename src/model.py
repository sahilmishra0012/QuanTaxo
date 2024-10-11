import os
import pickle as pkl
import torch
import numpy as np
import sys
import torch.nn as nn
from utils import *
from layers import MLP_VEC, NormalisedWeights
from transformers import BertModel


class QuanTaxo(nn.Module):
    def __init__(self,args,tokenizer):
        super(QuanTaxo, self).__init__()

        self.args = args
        self.data = self.__load_data__(self.args.dataset)
        self.FloatTensor = torch.cuda.FloatTensor if self.args.cuda else torch.FloatTensor
        self.concept_set = self.data["concept_set"]
        self.concept_id = self.data["concept2id"]
        self.id_concept = self.data["id2concept"]
        self.id_context = self.data["id2context"]

        self.train_concept_set = list(self.data["train_concept_set"])
        self.train_taxo_dict = self.data["train_taxo_dict"]
        self.train_child_parent_negative_parent_triple = self.data["train_child_parent_negative_parent_triple"]
        self.path2root = self.data["path2root"]
        self.test_concepts_id = self.data["test_concepts_id"]
        self.test_gt_id = self.data["test_gt_id"]

        self.pre_train_model = self.__load_pre_trained__()
        self.mixwt = NormalisedWeights(input_dim=128, mixturetype=self.args.mixture)
        self.logits = MLP_VEC(input_dim=(768+1),hidden=self.args.hidden,output_dim=1)
        self.dropout = nn.Dropout(self.args.dropout)
        self.classfn_loss = nn.BCELoss()

    def __load_data__(self,dataset):
        with open(os.path.join("../data/",dataset,"processed","taxonomy_data_"+str(self.args.expID)+"_.pkl"),"rb") as f:
            data = pkl.load(f)
        
        return data

    def __load_pre_trained__(self):
        model = BertModel.from_pretrained("bert-base-uncased")
        print("Model Loaded!")
        return model

    def get_bert_logits(self, enc_q, enc_cand):
        xfeat = torch.cat((enc_q, enc_cand),dim=-1)
        logits = self.bert_logits(xfeat)
        return logits
    
    def get_logits(self, norm_child, norm_parent):
        mat = torch.matmul(norm_child, norm_parent)
        trace = mat.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        xfeat = torch.cat((trace.view(len(trace),1), mat.diagonal(dim1=-2,dim2=-1)), dim=-1)
        logits = self.logits(xfeat)
        return logits

    def get_density_matrices(self, encode_inputs, printit=None):
        cls = self.pre_train_model(**encode_inputs)
        if self.args.mixture is not None:
            output = cls[0]
            weights = self.mixwt(output)
            if printit:
                np.savetxt("weights.csv",weights.cpu(),delimiter=",",fmt="%.4e")
            weighted_sum = torch.einsum('bik,bil,i->bkl',output,output, weights)
            if self.args.unitary:
                weighted_sum = self.get_unit_trace(weighted_sum)
            return weighted_sum
        else:
            output = self.dropout(cls[0][:, 0, :]) # Extract CLS embedding for all elements in the batch
            if self.args.pooled:
                pooled = cls[1]
                output = self.dropout(pooled)

            dens_mat = torch.einsum('bi,bj->bij', output, output) # Outer product to get the matrices rho_a
            if self.args.unitary:
                dens_mat = self.get_unit_trace(dens_mat)
            return dens_mat

    def classfn_score(self, norm_query, norm_candidate):
        return self.get_logits(norm_query, norm_candidate)

    def forward(self,encode_parent=None,encode_child=None,encode_negative_parents=None,flag="trace"):
        par_cls = self.get_density_matrices(encode_parent)
        child_cls = self.get_density_matrices(encode_child)
        neg_par_cls = self.get_density_matrices(encode_negative_parents)

        if flag=="trace":
            positive_logits = self.get_logits(child_cls, par_cls)
            negative_logits = self.get_logits(child_cls,neg_par_cls)
            ones = torch.ones_like(positive_logits)
            zeros = torch.zeros_like(negative_logits)
            positive_loss = self.classfn_loss(positive_logits,ones)
            negative_loss = self.classfn_loss(negative_logits,zeros)

        loss = positive_loss + negative_loss

        return loss
    