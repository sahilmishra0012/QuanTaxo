import numpy as np
import pandas as pd
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
    print("Predictions: ",preds)
    print("GT: ",gts)
    return acc

def precision_k(pred, gt,k):
    preds = np.array(list(pred[:,:k]))
    gts = np.array(list(gt))
    val = np.sum(preds==gts[:, np.newaxis])*1.0/(len(gt)*k)
    return val

def rank_scores(pred, gt):
    mrr=0
    mr=0
    dcg= 0.0
    idcg= 0.0
    cnt=0
    for i in range(len(pred)):
        for j in range(len(pred[i])):
            if pred[i][j]==gt[i]:
                # print(f"For {i}th element: We've found it at {j}th position")
                mr += (j+1)
                mrr += (1/(j+1))
                cnt += 1
                dcg += (1/np.log2((j+1)+1))
                idcg += (1/(np.log2(cnt+1)))
                break
    ndcg = dcg/idcg if idcg!=0 else 0
    ndcg = ndcg/len(gt)
    mrr = mrr/len(gt)
    mr = mr/len(gt)

    return mrr,mr,ndcg

def mrr_score(pred,gt):
    mrr = 0
    for i in range(len(pred)):
        for j in range(len(pred[i])):
            if pred[i][j]==gt[i]:
                # print(f"For {i}th element: We've found it at {j}th position")
                mrr+=1/(j+1)
    mrr = mrr/len(gt)
    # The larger the mrr, the better
    return mrr

def wu_p_score(pred, gt,path2root,compiled):

    pred = np.squeeze(pred[:,0])
    wu_p = 0
    for i in range(len(pred)):
        path_pred = path2root[pred[i]]
        path_gt = path2root[gt[i]]
        compiled[i].append(len(path_gt)-1)
        shared_nodes = set(path_pred)&set(path_gt)
        lca_depth = 1
        for node in shared_nodes:
            # lca_depth = max(len(path2root[node]), lca_depth)
            lca_depth = max(len(path2root[node])-1, lca_depth)
        wu_p+=2*lca_depth/(len(path_pred)+len(path_gt)-2)
    
    wu_p = wu_p/len(gt)

    return wu_p
        

def metrics(indices, gt, train_concept_set, path2root,testid2concept,trainid2concept,testconcepts,sortedscores):
    ind = np.squeeze(indices.detach().cpu().numpy())
    x,y = ind.shape
    pred = np.array([[i for i in range(y)] for _ in range(x)])
    # compiled=[[] for i in range(x)]
    compiled=[[testid2concept[testconcepts[i]],trainid2concept[gt[i]],sortedscores[i][0].item()] for i in range(x)]
    
    for i in range(len(pred)):
        pred[i]=np.array(list(train_concept_set))[ind[i]]
        compiled[i].append(trainid2concept[pred[i][0]])
        compiled[i].append(True if pred[i][0]==gt[i] else False)
        # predictions.append(list(train_concept_set)[i])

    acc = accuracy(pred,gt)
    # mrr = mrr_score(pred,gt)
    mrr,mr,ndcg = rank_scores(pred,gt)
    wu_p = wu_p_score(pred, gt, path2root,compiled)
    prec5 = precision_k(pred,gt,5)
    prec10 = precision_k(pred,gt,10)
    # print(compiled)
    depth = [elem[-2:] for elem in compiled]
    # print(depth)
    np.savetxt("depth_analysis.csv",depth,fmt="%s,%i",delimiter=",")

    file_path = 'depth_analysis.csv'  # Replace with your CSV file path
    data = pd.read_csv(file_path, header=None, names=['correct', 'depth'])

    # Group by difficulty level and calculate performance metrics
    report = data.groupby('depth').agg(
        total_attempts=('correct', 'count'),
        correct_answers=('correct', 'sum'),
        accuracy=('correct', lambda x: x.mean() * 100)
    ).reset_index()

    # Rename columns for clarity
    report.columns = ['Depth', 'Total Attempts', 'Correct Answers', 'Accuracy (%)']

    # Print or save the report
    print(report)
    return {"Acc": acc,"MRR": mrr,"MR": mr,"Wu": wu_p,"Prec@5": prec5,"Prec@10": prec10, "NDCG": ndcg}