import numpy as np
import random
import torch
import pytz
from datetime import datetime, timezone

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def print_local_time():

    utc_dt = datetime.now(timezone.utc)
    PST = pytz.timezone('Asia/Kolkata')
    print("Pacific time {}".format(utc_dt.astimezone(PST).isoformat()))

    return


def accuracy(pred, gt):
    # pred = np.squeeze(indices.detach().cpu().numpy()[:,0])
    # predictions = []
    # for i in pred:
    #     predictions.append(list(tr)[i])
    preds = np.array(list(pred[:,0]))
    gts = np.array(list(gt))
    acc = np.sum(preds==gts)/len(gt)

    return acc


def mrr_score(pred,gt):

    mrr = 0
    for i in range(len(pred)):
        for j in range(len(pred[i])):
            if pred[i][j]==gt[i]:
                # print(f"For {i}th element: We've found it at {j}th position")
                mrr+=1/(j+1)
    mrr = mrr/len(gt)

    return mrr



def wu_p_score(pred, gt,path2root):

    pred = np.squeeze(pred[:,0])
    wu_p = 0
    for i in range(len(pred)):
        path_pred = path2root[pred[i]]
        path_gt = path2root[gt[i]]
        shared_nodes = set(path_pred)&set(path_gt)
        lca_depth = 1
        for node in shared_nodes:
            lca_depth = max(len(path2root[node]), lca_depth)
        wu_p+=2*lca_depth/(len(path_pred)+len(path_gt))
    
    wu_p = wu_p/len(gt)

    return wu_p
        

def metrics(indices, gt, train_concept_set, path2root):
    ind = np.squeeze(indices.detach().cpu().numpy())
    x,y = ind.shape
    pred = np.array([[i for i in range(y)] for _ in range(x)])
    
    for i in range(len(pred)):
        pred[i]=np.array(list(train_concept_set))[ind[i]]
        # predictions.append(list(train_concept_set)[i])

    acc = accuracy(pred,gt)
    mrr = mrr_score(pred,gt)
    wu_p = wu_p_score(pred, gt, path2root)


    return acc, mrr, wu_p