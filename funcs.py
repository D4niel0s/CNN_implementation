import cupy as cp
import cupyx.scipy.special

def relu(x):
    return cp.maximum(x,0)

def relu_derivative(x):
    return (x>0)*1



def cross_entropy_loss(logits, y_true, out_dim):
    m = y_true.shape[0]
    # Compute log-sum-exp across each column for normalization
    log_probs = logits - cupyx.scipy.special.logsumexp(logits, axis=0)
    y_one_hot = cp.eye(out_dim)[y_true].T
    # Compute the cross-entropy loss
    loss = -cp.sum(y_one_hot * log_probs) / m
    return loss

#Output is of dimenswion out_dim x batch_size
def cross_entropy_derivative(logits, y_true, out_dim):
    yt_onehot = cp.eye(out_dim)[y_true].T
    zL = cupyx.scipy.special.softmax(logits, axis=0)
    return zL - yt_onehot