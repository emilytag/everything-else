from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse

#creates a probability distribution across the vector
def softmax(x):
  exp_scores = np.exp(x)
  return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

#nonlinear activation function
def sigmoid(x):
  return 1/(1 + np.exp(-x))

#visualize and save model weights
def visualize(model):
  W1 = model['W1']
  sqr = W1[:,0].reshape(28, 28)/200
  for i in range(1, len(W1[0])):
    img = plt.imshow(W1[:,i].reshape(28, 28))
    plt.savefig(str(i)+'layer.png')
    sqr += W1[:,i].reshape(28, 28)/200  
  img = plt.imshow(sqr)
  plt.savefig('sqr.png')

#cross-entropy loss
def loss_fx(model, X, y, num_examples):
  W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
  z1 = X.dot(W1) + b1
  a1 = sigmoid(z1)
  z2 = a1.dot(W2) + b2
  probs = softmax(z2)

  #look at mistakes and update loss fx
  #-1/n \sum \sum y_ni log yhat_ni
  #for each training example n , for each class i multiply y * log yhat
  logprobs = y * np.log(probs)
  data_loss = -np.sum(logprobs)
  #data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
  x_entropy_loss = data_loss / num_examples
 
  #determine accuracy on the gold data
  correct = 0.0
  for i in range(0, len(probs)):
    if np.argmax(probs[i]) == np.argmax(y[i]):
      correct += 1.0
  accuracy = correct / float(num_examples)
  return x_entropy_loss, accuracy

#cross-entropy loss for 2-layer model
def loss_fx_2layer(model, X, y, num_examples):
  W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']
  z1 = X.dot(W1) + b1
  a1 = sigmoid(z1)
  z2 = a1.dot(W2) + b2
  a2 = sigmoid(z2)
  z3 = a2.dot(W3) + b3
  probs = softmax(z3)

  #look at mistakes and update loss fx
  #-1/n \sum \sum y_ni log yhat_ni
  #for each training example n , for each class i multiply y * log yhat
  logprobs = y * np.log(probs)
  data_loss = -np.sum(logprobs)
  #data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
  x_entropy_loss = data_loss / num_examples
  
  #determine accuracy on the gold data
  correct = 0.0
  for i in range(0, len(probs)):
    if np.argmax(probs[i]) == np.argmax(y[i]):
      correct += 1.0
  accuracy = correct / float(num_examples)
  return x_entropy_loss, accuracy

#given an image, feed through the trained model and classify
def predict(model, x):
  W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
  z1 = x.dot(W1) + b1
  a1 = sigmoid(z1)
  z2 = a1.dot(W2) + b2
  probs = softmax(z2)
  return np.argmax(probs, axis=1)  

#trains the 2-layer feed forward network
def train_model_2layer(hidden_dim, input_dim, output_dim, learning_rate, X, y, epochs, valid_X, valid_y, momentum, dropout):
  train_examples = len(X)
  np.random.seed(1)
  b = np.sqrt(6) / np.sqrt(hidden_dim + input_dim)
  W1 = np.random.uniform(low=-b, high=b, size=(input_dim, hidden_dim))
  b1 = np.zeros((1, hidden_dim))
  W2 = np.random.uniform(low=-b, high=b, size=(hidden_dim, hidden_dim))
  b2 = np.zeros((1, hidden_dim))
  W3 = np.random.uniform(low=-b, high=b, size=(hidden_dim, output_dim))
  b3 = np.zeros((1, output_dim))
  vW1 = 0
  vW2 = 0
  vW3 = 0
  vb1 = 0
  vb2 = 0
  vb3 = 0
  model = {}
  
  for i in range(0, epochs):
    if dropout > 0.0:
      dropout_W1 = np.random.choice([0.0, 1.0], size=(input_dim, hidden_dim), p=[dropout, (1.0-dropout)])
      W1 = W1 * dropout_W1
    #forward pass
    z1 = X.dot(W1) + b1
    a1 = sigmoid(z1)
    z2 = a1.dot(W2) + b2
    a2 = sigmoid(z2)
    z3 = a2.dot(W3) + b3
    probs = softmax(z3)
    #backprop
    softmax_out = probs
    err = -(y - softmax_out)
    grad_activation = a1 * (1-a1)
    dW3 = (a2.T).dot(err) / train_examples
    db3 = np.sum(err, axis=0, keepdims=True) / train_examples
    grad_hk = err.dot(W3.T)
    grad_ak = grad_hk * grad_activation
    dW2 = (grad_hk.T).dot(grad_ak) / train_examples
    db2 = np.sum(grad_ak, axis=0, keepdims=True) / train_examples
    grad_h = grad_ak.dot(W2.T)
    grad_a = grad_h * grad_activation
    # dW1 size = 784 x 100
    dW1 = np.dot(X.T, grad_a) / train_examples
    db1 = np.sum(grad_a, axis=0) / train_examples
    #update parameters
    
    if momentum < 0.0:
      W1 = W1 - (learning_rate * dW1)
      b1 = b1 - (learning_rate * db1)
      W2 = W2 - (learning_rate * dW2)
      b2 = b2 - (learning_rate * db2)
      W3 = W3 - (learning_rate * dW3)
      b3 = b3 - (learning_rate * db3)
    else:
      vW1 = momentum * vW1 - learning_rate * dW1
      W1 += vW1
      vb1 = momentum * vb1 - learning_rate * db1
      b1 += vb1
      vW2 = momentum * vW2 - learning_rate * dW2
      W2 += vW2
      vb2 = momentum * vb2 - learning_rate * db2
      b2 += vb2
      vW3 = momentum * vW3 - learning_rate * dW3
      W3 += vW3
      vb3 = momentum * vb3 - learning_rate * db3
      b3 += vb3

    model['W1'] = W1
    model['W2'] = W2
    model['W3'] = W3
    model['b1'] = b1
    model['b2'] = b2
    model['b3'] = b3
    loss, accuracy = loss_fx_2layer(model, X, y, train_examples)
    valid_loss, valid_accuracy = loss_fx_2layer(model, valid_X, valid_y, len(valid_X))
    #print loss

    if i % 10 == 0:
      print loss,"\t", 1-accuracy,"\t", valid_loss,"\t", 1-valid_accuracy
  return model

#train the single layer feed forward network
def train_model_1layer(hidden_dim, input_dim, output_dim, learning_rate, X, y, epochs, valid_X, valid_y, momentum, dropout):
  print "training..."
  train_examples = len(X)
  #initial weights are random
  np.random.seed(1)
  b = np.sqrt(6) / np.sqrt(hidden_dim + input_dim)
  # 784 x 100
  W1 = np.random.uniform(low=-b, high=b, size=(input_dim, hidden_dim))
  # 1 x 100
  b1 = np.zeros((1, hidden_dim))
  # 100 x 10
  W2 = np.random.uniform(low=-b, high=b, size=(hidden_dim, output_dim))
  # 1 x 10 
  b2 = np.zeros((1, output_dim))
  vW1 = 0
  vW2 = 0
  vb1 = 0
  vb2 = 0
  model = {}
  
  #currently gradient descent
  #TO-DO: stochastic gradient descent updates every training example
  for i in range(0, epochs):
    if dropout > 0.0:
      dropout_W1 = np.random.choice([0.0, 1.0], size=(input_dim, hidden_dim), p=[dropout, (1.0-dropout)])
      W1 = W1 * dropout_W1
    #forward pass
    z1 = X.dot(W1) + b1
    a1 = sigmoid(z1)
    z2 = a1.dot(W2) + b2
    probs = softmax(z2)
    #backprop
    softmax_out = probs
    err = -(y - softmax_out)
    dW2 = (a1.T).dot(err) / train_examples
    db2 = np.sum(err, axis=0, keepdims=True) / train_examples
    # grad_h size: 3000 x 100
    grad_h = err.dot(W2.T)
    # grad_activation size: 100 x 100
    grad_activation = a1 * (1-a1)
    # delta2 size = 3000 x 100
    delta2 = grad_h * grad_activation
    # dW1 size = 784 x 100
    dW1 = np.dot(X.T, delta2) / train_examples
    db1 = np.sum(delta2, axis=0) / train_examples
    #regularization
    #dW2 += reg_lambda * W2
    #dW1 += reg_lambda * W1
    #update parameters
    
    if momentum < 0.0:
      W1 = W1 - (learning_rate * dW1)
      b1 = b1 - (learning_rate * db1)
      W2 = W2 - (learning_rate * dW2)
      b2 = b2 - (learning_rate * db2)
    else:
      vW1 = momentum * vW1 - learning_rate * dW1
      W1 += vW1
      vb1 = momentum * vb1 - learning_rate * db1
      b1 += vb1
      vW2 = momentum * vW2 - learning_rate * dW2
      W2 += vW2
      vb2 = momentum * vb2 - learning_rate * db2
      b2 += vb2
    
    model['W1'] = W1
    model['W2'] = W2
    model['b1'] = b1
    model['b2'] = b2
    loss, accuracy = loss_fx(model, X, y, train_examples)
    valid_loss, valid_accuracy = loss_fx(model, valid_X, valid_y, len(valid_X))
    #print loss
    
    if i % 10 == 0:
      print loss,"\t", 1-accuracy,"\t", valid_loss,"\t", 1-valid_accuracy
  return model

#splits MNIST data into training and testing sets  
def dataprocess(filename, output_dim):
  #TO-DO: shuf the training examples if SGD
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
    image = np.array(floats)
    all_train.append(image)
    labels.append(np.array(label_vec))
  X = np.array(all_train)
  y = np.array(labels)
  return X, y

#data normalization
def normalize(X):
  x_mean = X.mean()
  x_std = X.std()
  norm = (X - x_mean) / x_std
  return norm

#get accuracy on the test data
def test(model, output_dim):
  X, y = dataprocess('digitstest.txt', output_dim)
  examples = len(X)
  norm = normalize(X)
  #loss, accuracy = loss_fx(model, X, y, train_examples, reg_lambda)
  loss, accuracy = loss_fx_2layer(model, norm, y, examples)
  print "TEST RESULTS: ", loss, accuracy

def main():
  #784 -> 100 -> 10
  #full batch training
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_dim', default=784, type=int)
  parser.add_argument('--hidden_dim', default=200, type=int)
  parser.add_argument('--output_dim', default=10, type=int)
  parser.add_argument('--learning_rate', default=0.1, type=float)
  parser.add_argument('--epochs', default=1000, type=int)
  parser.add_argument('--layers', default=1, type=int)
  parser.add_argument('--viz', default=False, type=bool)
  parser.add_argument('--momentum', default=0.9, type=float)
  parser.add_argument('--dropout', default=0.0, type=float)
  args = parser.parse_args()

  #open training data with format vector,of,values,label
  X, y = dataprocess('digitstrain.txt', args.output_dim)
  valid_X, valid_y = dataprocess('digitsvalid.txt', args.output_dim)
  norm = normalize(X)
  valid_norm = normalize(valid_X)
  #take the average of the entire matrix
  #take std for entire matrix
  #x - mean / std to normalize the data
  if args.layers == 1:
    model = train_model_1layer(args.hidden_dim, args.input_dim, args.output_dim, args.learning_rate, norm, y, args.epochs, valid_norm, valid_y, args.momentum, args.dropout)
  elif args.layers == 2:
    model = train_model_2layer(args.hidden_dim, args.input_dim, args.output_dim, args.learning_rate, norm, y, args.epochs, valid_norm, valid_y, args.momentum, args.dropout)
  else:
    print "Sorry, only 1 and 2 layer models currenly supported"
  if args.viz:
    visualize(model)
  test(model, args.output_dim)
main()
