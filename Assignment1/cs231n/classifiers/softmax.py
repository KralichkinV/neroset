import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  D = W.shape[0]
  C = W.shape[1]
  N = X.shape[0]

  for i in range(N):
    #scores (C,)
    scores = X[i].dot(W)
    max_scores = np.max(scores)
    scores -= max_scores
    sum = np.sum(np.exp(scores))
    for j in range(C):
      dW[:,j] += np.exp(scores[j]) * X[i,:] / sum
    dW[:, y[i]] += - X[i,:]
    loss += -scores[y[i]] + np.log(sum)
  
  loss /= N
  dW /= N
  
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]

  scores = np.dot(X, W)
  exp_scores = np.exp(scores)
  prob_scores = exp_scores/np.sum(exp_scores, axis=1, keepdims=True)
  correct_log_probs = -np.log(prob_scores[range(num_train), y])
  loss = np.sum(correct_log_probs)
  loss /= num_train
  loss += 0.5 * reg * np.sum(W**2)

  # grads
  dscores = prob_scores
  dscores[range(num_train), y] -= 1
  dW = np.dot(X.T, dscores)
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

