import os
import pickle as pkl
import torch
import sys
import torch.nn as nn
from utils import *
from layers import MLP_VEC, NormalisedWeights
from transformers import BertModel


class BoxEmbed(nn.Module):
    def __init__(self,args,tokenizer):
        super(BoxEmbed, self).__init__()

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
        # self.sumparam = LINEAR_ONE(input_dim = 768) #bias=True by default
        self.mixwt = NormalisedWeights(input_dim=self.args.padmaxlen, mixturetype=self.args.mixture)
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

    # def check_pd(self, matrix):
    #     # Check if the matrix is positive definite
    #     real_part = torch.linalg.eigvals(matrix).real
    #     # Print all eigenvalues with negative real part
    #     mask = real_part<0
    #     sorted_real = torch.sort(real_part[mask],descending=True).values
    #     print("Eigenvalues < 0: ", sorted_real.size())
    #     return (torch.linalg.eigvals(matrix).real>=0).all()

    def get_unit_trace(self, matrixbatch):
        trace = torch.einsum('bii->b',matrixbatch)
        matrixbatch_normalised = matrixbatch/trace.unsqueeze(-1).unsqueeze(-1)
        checktrace = torch.einsum('bii->b',matrixbatch_normalised)
        # print("Check Trace:", checktrace)
        return matrixbatch_normalised

    def get_logits(self, norm_child, norm_parent):
        # Superposition, using simply [CLS]
        query_c = torch.einsum('bi,bj->bij', norm_child, norm_child) # Outer product to get the matrices rho_a
        # print("Query_C:", self.check_pd(query_c))
        query_c = self.get_unit_trace(query_c)
        query_p = torch.einsum('bi,bj->bij', norm_parent, norm_parent) # Outer product to get the matrices rho_p
        # print("Query_P:", self.check_pd(query_p))
        query_p = self.get_unit_trace(query_p)
        mat = torch.matmul(query_c, query_p)

        trace = mat.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        xfeat = torch.cat((trace.view(len(trace),1), mat.diagonal(dim1=-2,dim2=-1)), dim=-1)
        logits = self.logits(xfeat)
        return logits

    def get_bert_logits(self, enc_q, enc_cand):
        xfeat = torch.cat((enc_q, enc_cand),dim=-1)
        logits = self.bert_logits(xfeat)
        return logits

    def projection_cls(self,encode_inputs):
        cls = self.pre_train_model(**encode_inputs)
        if self.args.mixture is not None:
            output = cls[0]
            # print("Start of mix:", output.size())
            # if self.args.mixture=="constant":
            #     output = output.mean(dim=1)
            #     output = output.squeeze(1)
            output = self.mixwt(output)
            output = self.dropout(output)
        else:
            output = self.dropout(cls[0][:, 0, :]) # Extract CLS embedding for all elements in the batch
            if self.args.pooled:
                pooled = cls[1]
                output = self.dropout(pooled)
        # print(output.size())

        # # TO-DO, Rescale the embeddings
        # print(torch.norm(cls,dim=-1))
        # norms = torch.norm(cls, dim=1, keepdim=True)
        # epsilon = 1e-8
        # norms = norms + epsilon
        # y = cls/norms
        # print(torch.norm(y,dim=-1))
        # print(y.size())
        # sys.exit(0)
        return output

    def classfn_score(self, norm_query, norm_candidate):
        return self.get_logits(norm_query, norm_candidate)

    def forward(self,encode_parent=None,encode_child=None,encode_negative_parents=None,flag="train"):

        par_cls = self.projection_cls(encode_parent)
        child_cls = self.projection_cls(encode_child)
        positive_logits = self.get_logits(child_cls, par_cls)
        # positive_logits = self.get_bert_logits(child_cls, par_cls)

        neg_par_cls = self.projection_cls(encode_negative_parents)
        negative_logits = self.get_logits(child_cls,neg_par_cls)
        # negative_logits = self.get_bert_logits(child_cls,neg_par_cls)
        ones = torch.ones_like(positive_logits)
        zeros = torch.zeros_like(negative_logits)
        positive_loss = self.classfn_loss(positive_logits,ones)
        negative_loss = self.classfn_loss(negative_logits,zeros)

        loss = positive_loss + negative_loss

        return loss
    