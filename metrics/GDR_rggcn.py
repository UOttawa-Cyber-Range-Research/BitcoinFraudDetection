import torch
import numpy as np

def cluster_dist_metric(model_out, num_classes=2, gt_label=None):
    '''
    Implements the inter and intra clsuter distance for a sequence of graph
    '''
    
    # Placeholder
    X_labels = []
    
    # Loop and set the labels
    for i in range(num_classes):
        X_label = model_out[gt_label == i]
        
        # Check if torch tensor
        if type(X_label) == torch.Tensor:
            X_label = X_label.data.cpu().numpy()
        
        # Calculate the norm
        h_norm = np.sum(np.square(X_label), axis=1, keepdims=True)
        h_norm[h_norm == 0.] = 1e-3
        X_label = X_label / np.sqrt(h_norm)
        X_labels.append(X_label)

    # Intra cluster distance
    dis_intra = 0.0
    for i in range(num_classes):
        x2 = np.sum(np.square(X_labels[i]), axis=1, keepdims=True)
        dists = x2 + x2.T - 2 * np.matmul(X_labels[i], X_labels[i].T)
        dis_intra += np.mean(dists)
    dis_intra /= num_classes
    
    # Inter cluster distance
    dis_inter = 0.0
    for i in range(num_classes-1):
        for j in range(i+1, num_classes):
            x2_i = np.sum(np.square(X_labels[i]), axis=1, keepdims=True)
            x2_j = np.sum(np.square(X_labels[j]), axis=1, keepdims=True)
            dists = x2_i + x2_j.T - 2 * np.matmul(X_labels[i], X_labels[j].T)
            dis_inter += np.mean(dists)
    num_inter = float(num_classes * (num_classes-1) / 2)
    dis_inter /= num_inter

    return dis_intra, dis_inter