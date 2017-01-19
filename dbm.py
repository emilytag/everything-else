from hw1nn import train_model_1layer
from hw1nn import loss_fx
import random
import numpy as np
import matplotlib.pyplot as plt
import cPickle
import argparse

#nonlinear activation function
def sigmoid(x):
  return 1.0/(1 + np.exp(-x))

def gibbsSample(model, X, hidden_dim1, hidden_dim2, sample_num):
  vis_states = X
  W1 = model['W1']
  W2 = model['W2']
  train_examples = len(X)
  input_dim = len(W1[0])
  b = np.sqrt(6) / np.sqrt(hidden_dim1 + input_dim)
  mu_1 = np.random.uniform(low=-b, high=b, size=(train_examples, hidden_dim1))
  mu_2 = np.random.uniform(low=-b, high=b, size=(train_examples, hidden_dim2))
  for n in range(0, len(X)):
    for x in range(0, 1000):
      #hidden_dim1, 784  x 784, 1 + 1, hidden_dim2 x hidden_dim2, hidden_dim1
      mu_1_inside = np.dot(W1.T, X[n]) + np.dot(mu_2[n], W2.T)
      mu_1[n] = sigmoid(mu_1_inside)
      #1, hidden_dim1 x hidden_dim1, hidden_dim2
      mu_2_inside = np.dot(mu_1[n], W2)
      mu_2[n] = sigmoid(mu_2_inside)
  v_pred_inside = np.dot(W1, mu_1.T)
  v_pred = sigmoid(v_pred_inside)
  vis_states = (v_pred > np.random.random(v_pred.shape)).astype(np.int)
  return vis_states

#error function for the DBM
def calcError(model, X, hidden_dim1, hidden_dim2, b):
  W1 = model['W1']
  W2 = model['W2']
  train_examples = len(X)
  mu_1 = np.random.uniform(low=-b, high=b, size=(train_examples, hidden_dim1))
  mu_2 = np.random.uniform(low=-b, high=b, size=(train_examples, hidden_dim2))
  for n in range(0, len(X)):
    for x in range(0, 10):
      #hidden_dim1, 784  x 784, 1 + 1, hidden_dim2 x hidden_dim2, hidden_dim1
      mu_1_inside = np.dot(W1.T, X[n]) + np.dot(mu_2[n], W2.T)
      mu_1[n] = sigmoid(mu_1_inside)
      #1, hidden_dim1 x hidden_dim1, hidden_dim2
      mu_2_inside = np.dot(mu_1[n], W2)
      mu_2[n] = sigmoid(mu_2_inside)
  #784, hidden_dim1 x hidden_dim1, batch_size
  v_pred_inside = np.dot(W1, mu_1.T)
  v_pred = sigmoid(v_pred_inside)
  #batch_size, 784 x 784, k
  xe1 = np.log(np.multiply(X, v_pred.T))
  xe2 = np.log(np.multiply(1.0 - X, 1.0 - v_pred.T))
  error = (xe1[np.isfinite(xe1)].sum() + xe2[np.isfinite(xe2)].sum()) / train_examples
  return error

#visualize and save model weights
def visualize(model):
  fig = plt.figure()
  W1 = model['W1']
  sqr = W1[:,0].reshape(28, 28)/len(W1[0])
  img = plt.imshow(W1[:,0].reshape(28, 28), plt.get_cmap('gray'))
  plt.savefig('0layer.png')
  for i in range(1, len(W1[0])):
    img = plt.imshow(W1[:,i].reshape(28, 28), plt.get_cmap('gray'))
    plt.savefig(str(i)+'layer.png')
    sqr += W1[:,i].reshape(28, 28)/len(W1[0])
  img = plt.imshow(sqr, plt.get_cmap('gray'))
  plt.savefig('sqr.png')

#trains the DBM using contrastive divergence
def cdTrain(X, valid_X, epochs, input_dim, hidden_dim1, hidden_dim2, learning_rate, k, batch_size):
  print "training..."
  train_examples = len(X)
  batches = train_examples / batch_size
  np.random.seed(1)
  b = np.sqrt(6) / np.sqrt(hidden_dim1 + input_dim)
  #784, 100
  W1 = np.random.uniform(low=-b, high=b, size=(input_dim, hidden_dim1))
  #100, 100
  W2 = np.random.uniform(low=-b, high=b, size=(hidden_dim1, hidden_dim2))
  #1, 100
  hb1 = np.zeros((1, hidden_dim1))
  hb2 = np.zeros((1, hidden_dim2))
  #1, 784
  vb = np.zeros((1, input_dim))
  model = {} 
  model['W1'] = W1
  visualize(model)

  h_2 = np.random.choice([0.0, 1.0], size=(k, hidden_dim2), p=[0.5, 0.5])
  h_1 = np.zeros((k, hidden_dim1))
  v_tilde = np.random.choice([0.0, 1.0], size=(k, input_dim), p=[0.5, 0.5])
  h_tilde = np.random.choice([0.0, 1.0], size=(k, hidden_dim1), p=[0.5, 0.5])
  
  for i in xrange(0, epochs):
    data = X
    np.random.shuffle(data)
    data = data[0:batch_size * batches]
    data = data.reshape((batches, batch_size, input_dim))
    for img in data:
      mu_1 = np.random.uniform(low=-b, high=b, size=(batch_size, hidden_dim1))
      mu_2 = np.random.uniform(low=-b, high=b, size=(batch_size, hidden_dim2))
      for n in range(0, len(img)):
        for x in range(0, 10):
          #hidden_dim1, 784  x 784, 1 + 1, hidden_dim2 x hidden_dim2, hidden_dim1
          mu_1_inside = np.dot(W1.T, img[n]) + np.dot(mu_2[n], W2.T)
          mu_1[n] = sigmoid(mu_1_inside)
          #1, hidden_dim1 x hidden_dim1, hidden_dim2
          mu_2_inside = np.dot(mu_1[n], W2)
          mu_2[n] = sigmoid(mu_2_inside)
      for m in range(0, k):
        #1, 784 x 784, hidden_dim1 + hidden_dim1, hidden_dim2 x hidden_dim2, 1
        h_1_inside = np.dot(v_tilde[m], W1) + np.dot(W2, h_2[m].T)
        h_1_sig = sigmoid(h_1_inside + hb1)
        h_1[m] = (h_1_sig > np.random.rand(1, hidden_dim1)).astype(np.int)

        #1, hidden_dim1 x hidden_dim1, hidden_dim2 
        h_2_inside = np.dot(h_1[m], W2)
        h_2_sig = sigmoid(h_2_inside + hb2)
        h_2[m] = (h_2_sig > np.random.rand(1, hidden_dim2)).astype(np.int)
        
        #784, hidden_dim1 x hidden_dim1, 1 
        v_tilde_inside = np.dot(W1, h_1[m].T)
        v_tilde_sig = sigmoid(v_tilde_inside + vb)
        v_tilde[m] = (v_tilde_sig > np.random.rand(1, input_dim)).astype(np.int)

      #update weights
      #hidden_dim1, batch_size x batch_size, 784 - hidden_dim1, k x k, 784
      W1  += learning_rate * (np.dot(mu_1.T, img)/batch_size - np.dot(h_1.T, v_tilde)/k).T
      W2 += learning_rate * (np.dot(mu_2.T, mu_1)/batch_size - np.dot(h_2.T, h_1)/k) 
      model['W1'] = W1
      model['W2'] = W2

    if i % 10 == 0:
      error = calcError(model, X, hidden_dim1, hidden_dim2, b)
      valid_error = calcError(model, valid_X, hidden_dim1, hidden_dim2, b)
      print error, '\t', valid_error
      outfile5 = open('model'+str(hidden_dim1)+'.pickle', 'wb')
      fastPickler = cPickle.Pickler(outfile5, cPickle.HIGHEST_PROTOCOL)
      fastPickler.fast = 1
      fastPickler.dump(model)
      outfile5.close()
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
  parser.add_argument('--k', default=100, type=int)
  parser.add_argument('--sample_num', default=1000, type=int)
  parser.add_argument('--input_dim', default=784, type=int)
  parser.add_argument('--hidden_dim1', default=200, type=int)
  parser.add_argument('--hidden_dim2', default=200, type=int)
  parser.add_argument('--output_dim', default=10, type=int)
  parser.add_argument('--learning_rate', default=0.01, type=float)
  parser.add_argument('--epochs', default=100, type=int)
  parser.add_argument('--dropout', default=0.0, type=float)
  parser.add_argument('--momentum', default=0.0, type=float)
  parser.add_argument('--batch_size', default=50, type=int)
  parser.add_argument('--viz', default=False, type=bool)
  parser.add_argument('--input_model', default='')
  parser.add_argument('--output_model', default='')
  args = parser.parse_args()

  X, y = dataprocess('digitstrain.txt', args.output_dim)
  valid_X, valid_y = dataprocess('digitsvalid.txt', args.output_dim)
  
  if args.input_model != '': 
    FILE4 = open(args.input_model, 'r')
    f4 = FILE4.read()
    model = cPickle.loads(f4)
    FILE4.close()
  
  model = cdTrain(X, valid_X, args.epochs, args.input_dim, args.hidden_dim1, args.hidden_dim2, args.learning_rate, args.k, args.batch_size)
  print
  if args.viz:
    visualize(model)
 
  outfile5 = open(args.output_model, 'wb')
  fastPickler = cPickle.Pickler(outfile5, cPickle.HIGHEST_PROTOCOL)
  fastPickler.fast = 1
  fastPickler.dump(model)
  outfile5.close()
  
main()
