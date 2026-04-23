import numpy as np
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import minimize

class SoftLabelLogisticRegression:

  def __init__(self, C = 1.):
    self.C = C

  def fit(self, X, Y):
    """ Fits a logistic regression to a
    soft label target using L-BFGS and no
    intercept.

    Parameters
    ----------
    X: np.ndarray
      A num_data x num_features feature matrix
      as input for the regression.
    Y: np.ndarray
      A num_data x num_labels target matrix as
      target for the regression. It is assumed
      that each row forms a probability distribution,
      i.e. only non-negative entries and each row
      sums to 1.

    Returns
    -------
    self
    """
    if X.shape[0] != Y.shape[0]:
      raise ValueError('Inconsistent shape of X and Y')
    if np.any(Y < 0):
      raise ValueError('negative entries in Y')
    if np.any(np.abs(np.sum(Y, axis = 1) - 1.) > 1E-3):
      raise ValueError('at least one row of Y does not add up to 1')
    N, n = X.shape
    _, K = Y.shape
    # define the objective
    def obj(params):
      # extract the weight matrix from the parameter
      # vector
      W = np.reshape(params, (K, n))
      # compute logits
      Z = X @ W.T
      # compute probabilities
      P = np.exp(-Z)
      P = P / np.expand_dims(np.sum(P, axis = 1), axis = 1)
      # compute loss
      loss = np.sum(-Y * np.log(P)) / N + 1. / self.C * np.sum(W ** 2)
      # compute gradient
      grad = (Y - P).T @ X / N + 1. / self.C * 2 * W
      # return
      return loss, grad.flatten()

    init_params = np.zeros(K*n)
    res = minimize(obj, init_params, jac = True)
    if not res.success:
      print(res)
    self.W_ = np.reshape(res.x, (K, n))

  def predict(self, X):
    # compute logits
    Z = X @ self.W_.T
    return np.argmax(Z, axis = 1)

  def predict_proba(self, X):
    # compute logits
    Z = X @ self.W_.T
    # compute probabilities
    P = np.exp(-Z)
    P = P / np.expand_dims(np.sum(P, axis = 1), axis = 1)
    return P

  def score(self, X, Y):
    P = self.predict_proba(X)
    return 1. - np.mean(np.abs(P - Y))

def multi_label_dsl(X, Q, R, Y, pi = None, num_folds = 5, C = 1000.):
  """ Applies Design-Based Supervised Learning
  to the given dataset.

  Parameters
  ----------
  X: np.ndarray
    A num_data x num_features feature matrix
    as input for the regression.
  Q: np.ndarray
    A num_data x num_labels matrix of predictive
    distributions by an LLM for a committee of LLMs.
    It is assumed that each row forms a probability
    distribution, i.e. only non-negative entries
    and each row sums to 1.
  R: np.ndarray
    A binary vector with num_data elements
    where R[i] = True if sample i has been
    annotated by experts and R[i] = False,
    otherwise.
  Y: np.ndarray
    A num_annotated_data x num_labels target matrix
    of expert annotations, where Y[i, :] is the
    distribution of expert labels for the ith
    sample. Note that Y only has as many rows as
    R has True entries.
    It is assumed that each row forms a probability
    distribution, i.e. only non-negative entries
    and each row sums to 1.
  pi: np.ndarray (default = None)
    A vector with num_data elements,
    where pi[i] contains the probability of the ith
    data point to be sampled for annotation.
    If not given, this is set to np.sum(R) / len(R)
    but stratified sampling strategies can also be
    assigned here.
  num_folds: int (default = 5)
    The number of crossvalidation folds for DSL.
  C: float (default = 10)
    The inverse regularization strength for the
    logistic regression.

  Returns
  -------
  G: np.ndarray
    An adjusted num_data x num_labels array of
    label distributions predicted by the logistic
    regression in crossvalidation.
  Ytilde: np.ndarray
    An adjusted num_data x num_labels array of
    label distributions via DSL. Note that entries
    where R[i] = 1 will have substantially larger
    entries due to the DSL weighting.

  """
  M, n = X.shape
  if Q.shape[0] != M:
    raise ValueError('Inconsistent number of rows in X and Q')
  if np.any(Q < 0):
    raise ValueError('negative entries in Q')
  if np.any(np.abs(np.sum(Q, axis = 1) - 1.) > 1E-3):
    raise ValueError('at least one row of Q does not add up to 1')
  _, K = Q.shape
  if len(R) != M:
    raise ValueError('Inconsistent number of rows in X and R')
  N = int(np.sum(R))
  if N != Y.shape[0]:
    raise ValueError('Expected as many true entries in R as there are rows in Y')
  if K != Y.shape[1]:
    raise ValueError('Inconsistent number of columns in Q and Y')
  if np.any(Y < 0):
    raise ValueError('negative entries in Y')
  if np.any(np.abs(np.sum(Y, axis = 1) - 1.) > 1E-3):
    raise ValueError('at least one row of Y does not add up to 1')

  # set pi if not given
  if pi is None:
    pi = np.ones(M) * N / M

  if len(pi) != M:
    raise ValueError('Inconsistent number of rows in X and pi')

  # construct data matrix for logistic regression,
  # where we concatenate X and Q
  X_logreg = np.concatenate((X, Q), axis = 1)

  # initialize the matrix of predictions
  G = np.zeros((M, K))

  # set up a num_folds crossvalidation, such that each fold
  # gets at least some labeled samples
  kf = StratifiedKFold(n_splits=num_folds)
  for (train_index, test_index) in kf.split(X, R):
    # Retrieve the input for the logistic regression for all
    # training samples that have been labeled by human experts
    Xtrain = X_logreg[train_index, :][R[train_index]]
    # Retrieve the target for all training data that have
    # been labeled by human experts
    train_logical = np.zeros(M)
    train_logical[train_index] = 1.
    Ytrain = Y[train_logical[R] > 0.5, :]
    # train the logistic regression
    model = SoftLabelLogisticRegression(C = C)
    model.fit(Xtrain, Ytrain)
    # predict for all samples in the test set
    G[test_index, :] =  model.predict_proba(X_logreg[test_index, :])

  # adjust the prediction using the DSL formula (4)
  Ytilde = np.copy(G)
  Ytilde[R, :] = G[R, :] + (Y - G[R, :]) / np.expand_dims(pi[R], 1)

  # return
  return G, Ytilde

