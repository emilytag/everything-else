from feedforwardnn import train_model_1layer
from feedforwardnn import loss_fx
import random
import numpy as np
import matplotlib.pyplot as plt
import argparse

#nonlinear activation function
def sigmoid(x):
  return 1.0/(1 + np.exp(-x))

def gibbsSample(model, X, hidden_dim, sample_num):
  vis_states = X
  W = model['W']
  hb = model['hb']
  vb = model['vb']
  for i in range(0, sample_num):
    #1, 784 x 784, 100   
    activations = np.dot(vis_states, W) + hb
    probs = sigmoid(activations)
    states = (probs > np.random.rand(len(X), hidden_dim)).astype(np.int)
    vis_activations = np.dot(states, W.T) + vb
    vis_probs = sigmoid(vis_activations)
    vis_states = (vis_probs > np.random.random(vis_probs.shape)).astype(np.int)
  return vis_states

#error function for the RBM
def calcError(model, X, hidden_dim):
  W = model['W']
  hb = model['hb'] 
  vb = model['vb']
  activations = np.dot(X, W) + hb
  probs = sigmoid(activations)
  states = (probs > np.random.rand(len(X), hidden_dim)).astype(np.int)
  vis_activations = np.dot(states, W.T) + vb
  vis_probs = sigmoid(vis_activations)
  error = np.square(X - vis_probs).sum() / float(len(X))
  return error

#visualize and save model weights
def visualize(model, model_type):
  fig = plt.figure()
  try:
    W1 = model['W']
  except:
    W1 = model['W1']
  #sqr = W1[:,0].reshape(28, 28)/len(W1[0])
  img = plt.imshow(W1[:,0].reshape(28, 28), plt.get_cmap('gray'))
  plt.savefig('0layer-'+model_type+'.png')
  for i in range(1, len(W1[0])):
    img = plt.imshow(W1[:,i].reshape(28, 28), plt.get_cmap('gray'))
    plt.savefig(str(i)+'layer-'+model_type+'.png')
    #sqr += W1[:,i].reshape(28, 28)/len(W1[0])
  #img = plt.imshow(sqr, plt.get_cmap('gray'))
  #plt.savefig('sqr.png')

#trains the RBM using contrastive divergence
def cdTrain(X, valid_X, epochs, input_dim, hidden_dim, learning_rate, k):
  print "training..."
  train_examples = len(X)
  np.random.seed(1)
  b = np.sqrt(6) / np.sqrt(hidden_dim + input_dim)
  #784, 100
  W = np.random.uniform(low=-b, high=b, size=(input_dim, hidden_dim))
  #1, 100
  hb = np.zeros((1, hidden_dim))
  #1, 784
  vb = np.zeros((1, input_dim))
  model = {}

  for i in xrange(0, epochs):
    #3000, 784 x 784, 100
    activations = np.dot(X, W) + hb
    probs = sigmoid(activations)
    for j in range(0, k):
      #sample
      if j > 0:
        states = (neg_probs > np.random.rand(train_examples, hidden_dim)).astype(np.int)
        #784, 3000 x 3000, 100
        associations = np.dot(vis_probs.T, states)
      else:  
        states = (probs > np.random.rand(train_examples, hidden_dim)).astype(np.int)
        associations = np.dot(X.T, states)
      #negative CD phase, daydreaming
      #3000, 100 x 100, 784
      vis_activations = np.dot(states, W.T) + vb
      vis_probs = sigmoid(vis_activations)
      #3000, 784 x 784, 100
      neg_activations = np.dot(vis_probs, W) + hb
      neg_probs = sigmoid(neg_activations)
      #784, 3000 x 3000, 100
      neg_associations = np.dot(vis_probs.T, neg_probs)

    #update weights
    #784, 100 - 784, 100
    W += learning_rate * ((associations - neg_associations) / train_examples)
    #3000, 784 - 3000, 784
    vb += learning_rate * (X - vis_probs).mean(axis=0)
    #3000, 100 - 3000, 100
    hb += learning_rate * (probs - neg_probs).mean(axis=0)
    model['W'] = W
    model['vb'] = vb
    model['hb'] = hb
    if i % 10 == 0:
      error = np.square(X - vis_probs).sum() / float(train_examples)
      valid_error = calcError(model, valid_X, hidden_dim)
      print error, '\t', valid_error
  return model

#loss function for the autoencoder
def encLoss(X, model):
  W1 = model['W1']
  W2 = model['W2']
  b1 = model['b1']
  b2 = model['b2']
  #3000, 784 x 784, 100
  z1 = X.dot(W1) + b1
  # 3000, 100
  a1 = sigmoid(z1)
  # 3000, 100 x 100, 784
  z2 = z1.dot(W2) + b2
  # 3000, 784
  a2 = sigmoid(z2)
  eps = 1e-10
  loss = - np.sum((X * np.log(a2 + eps) + (1.-X) * np.log(1.-a2 + eps)))
  return loss / len(X)

#training for the autoencoder
def autoencTrain(X, valid_X, epochs, input_dim, hidden_dim, learning_rate, batch_size, dropout):
  print "training..."
  train_examples = len(X)
  batches = train_examples / batch_size
  #initial weights are random
  np.random.seed(1)
  b = np.sqrt(6) / np.sqrt(hidden_dim + input_dim)
  # 784, 100
  W1 = np.random.uniform(low=-b, high=b, size=(input_dim, hidden_dim))
  # 100, 784
  W2 = W1.T
  # 1, 100
  b1 = np.zeros((1, hidden_dim))
  # 1, 784
  b2 = b2 = np.zeros((1, input_dim))
  model = {}

  for i in range(0, epochs):
    data = X
    np.random.shuffle(data)
    data = data[0:batch_size * batches]
    data = data.reshape((batches, batch_size, input_dim))
    for img in data:
      if dropout > 0.0:
        dropout_W1 = np.random.choice([0.0, 1.0], size=(input_dim, hidden_dim), p=[dropout, (1.0-dropout)])
        W1 = W1 * dropout_W1
      # batch_size, 784 x 784, 100 + 1, 100 
      z1 = img.dot(W1) + b1
      # batch_size, 100
      a1 = sigmoid(z1)
      # batch_size, 100 x 100, 784
      z2 = a1.dot(W2) + b2
      # batch_size, 784
      a2 = sigmoid(z2)
      # batch_size, 784
      err = -(img - a2)
      grad_activation = a1 * (1 - a1)
      # batch_size, 784 x batch_size, 100 -> 784, 100
      dW1 = np.dot(err.T, a1) / batch_size
      db2 = np.sum(err, axis=0, keepdims=True) / batch_size
      #batch_size, 784 x 784, 100
      grad_h = np.dot(err, W2.T)
      delta2 = grad_h * grad_activation
      #add 0.25 binary mask for denoising
      db1 = np.sum(delta2, axis=0) / batch_size
      W1 = W1 - (learning_rate * dW1)
      b1 = b1 - (learning_rate * db1)
      W2 = W2 - (learning_rate * dW1.T)
      b2 = b2 - (learning_rate * db2)
      model['W1'] = W1
      model['W2'] = W2
      model['b1'] = b1
      model['b2'] = b2
    loss = encLoss(X, model)
    valid_loss = encLoss(valid_X, model)
    
    if i % 10 == 0:
      print loss, '\t', valid_loss
  return model
  
#splits MNIST data into training and testing sets
def dataprocess(filename, output_dim):
  data = open(filename, 'r')
  examples = 0
  all_train = []
  labels = []

  for line in data:
    examples += 1
    line = line.strip().split(',')
    label = int(line[-1])
    label_vec = [0.0] * output_dim
    label_vec[label] = 1.0
    floats = [float(x) for x in line[:-1]]
    for y in range(0, len(floats)):
      if floats[y] >= 0.5:
        floats[y] = 1.0
      else:
        floats[y] = 0.0
    image = np.array(floats)
    all_train.append(image)
    labels.append(np.array(label_vec))
  X = np.array(all_train)
  y = np.array(labels)
  return X, y

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--k', default=1, type=int)
  parser.add_argument('--sample_num', default=1000, type=int)
  parser.add_argument('--input_dim', default=784, type=int)
  parser.add_argument('--hidden_dim', default=100, type=int)
  parser.add_argument('--output_dim', default=10, type=int)
  parser.add_argument('--learning_rate', default=0.1, type=float)
  parser.add_argument('--epochs', default=500, type=int)
  parser.add_argument('--dropout', default=0.0, type=float)
  parser.add_argument('--momentum', default=0.0, type=float)
  parser.add_argument('--batch_size', default=300, type=int)
  parser.add_argument('--viz', default=False, type=bool)
  args = parser.parse_args()

  X, y = dataprocess('digitstrain.txt', args.output_dim)
  valid_X, valid_y = dataprocess('digitsvalid.txt', args.output_dim)

  #a) CD algorithm for training RBM
  model = cdTrain(X, valid_X, args.epochs, args.input_dim, args.hidden_dim, args.learning_rate, args.k)
  print
  if args.viz:
    visualize(model, "rbm")
 
  #c) initialize 100 gibbs chains with random configs, run for 1000 steps and display 100 sampled images
  ''' 
  for i in xrange(0, 100):
    print "iteration", i, "..."
    r = np.random.rand(1, args.input_dim)
    vis = gibbsSample(model, r, args.hidden_dim, args.sample_num)
    sqr = vis.reshape(28, 28)
    img = plt.imshow(sqr, cmap=plt.get_cmap('gray'))
    plt.savefig(str(i)+'sample.png')
  '''

  #d) use learned weights to classify using 1-layer network like in assignment 1
  train_model_1layer(args.hidden_dim, args.input_dim, args.output_dim, args.learning_rate, X, y, args.epochs, valid_X, valid_y, args.momentum, args.dropout, model)

  #e) train autoencoder w 100 sigmoid hidden units instead of RBM
  model = autoencTrain(X, valid_X, args.epochs, args.input_dim, args.hidden_dim, args.learning_rate, args.batch_size, args.dropout)
  if args.viz:
    visualize(model, "ae-den1")
  train_model_1layer(args.hidden_dim, args.input_dim, args.output_dim, args.learning_rate, X, y, args.epochs, valid_X, valid_y, args.momentum, args.dropout, model)

main()
