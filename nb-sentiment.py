import argparse
import cPickle
import os
import math
import re

#Reads in the positive & negative lexicon items to a uniqued list and returns that list
def readLexicon(posLexFile, negLexFile, negation, adverbs):
  posLex = open(posLexFile, "r")
  negLex = open(negLexFile, "r")
  entireLex = []
  appendAdverbs = ["extremely", "quite", "just", "almost", "very", "too", "enough"]
  for line in posLex:
    if not line.startswith(";"): #gets rid of the file headers
      entireLex.append(line.strip())
      if negation:
        entireLex.append("not_"+line.strip()) #appends not_s to lexicon if specified
      if adverbs:
        for adverb in appendAdverbs:
          entireLex.append(adverb+"_"+line.strip()) #appends adverbs to lexicon if specified
  for line in negLex:
    if not line.startswith(";"):
      entireLex.append(line.strip())
      if negation:
        entireLex.append("not_"+line.strip())
      if adverbs:
        for adverb in appendAdverbs:
          entireLex.append(adverb+"_"+line.strip()) #appends adverbs to lexicon if specified
  return list(set(entireLex)) #converting to a set removes duplicates


#trains on given training data
def train(lexicon, trainFiles, negation, adverbs):
  nbData = {"positive": [], "negative": [], "totalNeg": 0, "totalPos": 0}
  vocabSize = len(lexicon)
  for trainfile in os.listdir(trainFiles+"neg/"):
    if "cv8" not in trainfile and "cv9" not in trainfile: #skips testing files
      sentiment = "negative" #To be used as a key in nbData
      nbData["totalNeg"] += 1 #Updates count to be used for p(neg)
      train = open(trainFiles+"neg/"+trainfile,"r")
      if negation:
        data = train.read()
        newdata = re.sub('n\'t\s|\snot\s',' not_', data,flags=re.IGNORECASE) #replaces nots and n'ts
        words = newdata.split()
        tokens = list(set([word for word in words if word in lexicon]))
      if adverbs:
        data = train.read()
        newdata = re.sub('\sextremely\s',' extremely_', data,flags=re.IGNORECASE) #replaces adverbs
        newdata = re.sub('\squite\s',' quite_', newdata,flags=re.IGNORECASE)
        newdata = re.sub('\sjust\s',' just_', newdata,flags=re.IGNORECASE)
        newdata = re.sub('\salmost\s',' almost_', newdata,flags=re.IGNORECASE)
        newdata = re.sub('\svery\s',' very_', newdata,flags=re.IGNORECASE)
        newdata = re.sub('\stoo\s',' too_', newdata,flags=re.IGNORECASE)
        newdata = re.sub('\senough\s',' enough_', newdata,flags=re.IGNORECASE)
        words = newdata.split()
        tokens = list(set([word for word in words if word in lexicon]))
      else:
        words = train.read().split()
        tokens = list(set([word for word in words if word in lexicon])) #filters out words not in the lexicon and uniques them
      startVector = [0] * vocabSize
      for word in tokens:
        index = lexicon.index(word)
        startVector[index] += 1 #Updates the index for words that occur
      nbData[sentiment].append(startVector)

  for trainfile in os.listdir(trainFiles+"pos/"): #Same deal for positive files
    if "cv8" not in trainfile and "cv9" not in trainfile:
      sentiment = "positive"
      nbData["totalPos"] += 1
      train = open(trainFiles+"pos/"+trainfile,"r")
      if negation:
        data = train.read()
        newdata = re.sub('n\'t\s|\snot\s',' not_', data,flags=re.IGNORECASE) #replaces nots and n'ts
        words = newdata.split()
        tokens = list(set([word for word in words if word in lexicon]))
      if adverbs:
        data = train.read()
        newdata = re.sub('\sextremely\s',' extremely_', data,flags=re.IGNORECASE) #replaces adverbs
        newdata = re.sub('\squite\s',' quite_', newdata,flags=re.IGNORECASE)
        newdata = re.sub('\sjust\s',' just_', newdata,flags=re.IGNORECASE)
        newdata = re.sub('\salmost\s',' almost_', newdata,flags=re.IGNORECASE)
        newdata = re.sub('\svery\s',' very_', newdata,flags=re.IGNORECASE)
        newdata = re.sub('\stoo\s',' too_', newdata,flags=re.IGNORECASE)
        newdata = re.sub('\senough\s',' enough_', newdata,flags=re.IGNORECASE)
        words = newdata.split()
        tokens = list(set([word for word in words if word in lexicon]))
      else:
        words = train.read().split()
        tokens = list(set([word for word in words if word in lexicon])) #filters out words not in the lexicon and uniques them
      startVector = [0] * vocabSize
      for word in tokens:
        index = lexicon.index(word)
        startVector[index] += 1
      nbData[sentiment].append(startVector)  
  return nbData, vocabSize

#Estimates conditional probability of a word
def estimateParameter(lexicon, nbData, word, sentiment):
  index = lexicon.index(word) #index in the training vector for that word
  count = 0
  total = len(nbData[sentiment]) #total amount of training vectors 
  for vector in nbData[sentiment]:
    count += vector[index]
  parameter = (float(count)+0.1)/(float(total)+0.1)
  return parameter

#outputs top ten highest probabilities for P(word|sentiment)
def topWords(lexicon, model):
  posWords = {}
  negWords = {}
  for word in lexicon: #estimates probabilities for every word in lexicon
    posWords[word] = estimateParameter(lexicon, model, word, "positive")
    negWords[word] = estimateParameter(lexicon, model, word, "negative")
  #prints highest values in the dict  
  print "most informative positive words"
  for w in sorted(posWords.items(), key=lambda x: x[1], reverse=True)[:10]:
    print w[0], w[1]
  print "most informative negative words"
  for w in sorted(negWords.items(), key=lambda x: x[1], reverse=True)[:10]:
    print w[0], w[1]

#Tests on the training files for 5.3a
def trainTest(lexicon, model, vocabSize, testFiles, negation, adverbs):
  negTrials = 0 #total negative training examples
  negAccuracy = 0 #negative examples classified correctly

  negPrior = float(model["totalNeg"]) / (float(model["totalNeg"])+float(model["totalPos"]))
  posPrior = float(model["totalPos"]) / (float(model["totalNeg"])+float(model["totalPos"]))

  for testfile in os.listdir(testFiles+"neg/"):
    if "cv8" not in testfile and "cv9" not in testfile:
      test = open(testFiles+"neg/"+testfile,"r")
      if negation:
        data = test.read()
        newdata = re.sub('n\'t\s|\snot\s',' not_', data,flags=re.IGNORECASE) #replaces nots and n'ts
        words = newdata.split()
        tokens = list(set([word for word in words if word in lexicon]))
      if adverbs:
        data = test.read()
        newdata = re.sub('\sextremely\s',' extremely_', data,flags=re.IGNORECASE) #replaces adverbs
        newdata = re.sub('\squite\s',' quite_', newdata,flags=re.IGNORECASE)
        newdata = re.sub('\sjust\s',' just_', newdata,flags=re.IGNORECASE)
        newdata = re.sub('\salmost\s',' almost_', newdata,flags=re.IGNORECASE)
        newdata = re.sub('\svery\s',' very_', newdata,flags=re.IGNORECASE)
        newdata = re.sub('\stoo\s',' too_', newdata,flags=re.IGNORECASE)
        newdata = re.sub('\senough\s',' enough_', newdata,flags=re.IGNORECASE)
        words = newdata.split()
        tokens = list(set([word for word in words if word in lexicon]))
      else:
        words = test.read().split()
        tokens = list(set([word for word in words if word in lexicon])) #filters out words not in the lexicon
      negProb = 0
      posProb = 0
      for word in tokens:
        conditionalProbNeg = estimateParameter(lexicon, model, word, "negative")
        conditionalProbPos = estimateParameter(lexicon, model, word, "positive")
        
        negProb += math.log(conditionalProbNeg) #update the probability of (x|negative)
        posProb += math.log(conditionalProbPos) #update the probability of (x|positive)

      negProb = negProb + math.log(negPrior) #add the prior for the full probability
      posProb = posProb + math.log(posPrior)
      if negProb > posProb: #counts the correct classifications
        negTrials += 1
        negAccuracy += 1
      else:
        negTrials += 1
  print "negative accuracy on training data=", float(negAccuracy) / float(negTrials)

  posTrials = 0
  posAccuracy = 0
  for testfile in os.listdir(testFiles+"pos/"): #same thing for positive
    if "cv8" not in testfile and "cv9" not in testfile:
      test = open(testFiles+"pos/"+testfile,"r")
      if negation:
        data = test.read()
        newdata = re.sub('n\'t\s|\snot\s',' not_', data,flags=re.IGNORECASE) #replaces nots and n'ts
        words = newdata.split()
        tokens = list(set([word for word in words if word in lexicon]))
      if adverbs:
        data = test.read()
        newdata = re.sub('\sextremely\s',' extremely_', data,flags=re.IGNORECASE) #replaces adverbs
        newdata = re.sub('\squite\s',' quite_', newdata,flags=re.IGNORECASE)
        newdata = re.sub('\sjust\s',' just_', newdata,flags=re.IGNORECASE)
        newdata = re.sub('\salmost\s',' almost_', newdata,flags=re.IGNORECASE)
        newdata = re.sub('\svery\s',' very_', newdata,flags=re.IGNORECASE)
        newdata = re.sub('\stoo\s',' too_', newdata,flags=re.IGNORECASE)
        newdata = re.sub('\senough\s',' enough_', newdata,flags=re.IGNORECASE)
        words = newdata.split()
        tokens = list(set([word for word in words if word in lexicon]))
      else:
        words = test.read().split()
        tokens = list(set([word for word in words if word in lexicon])) #filters out words not in the lexicon
      negProb = 0
      posProb = 0
      for word in tokens:
        conditionalProbNeg = estimateParameter(lexicon, model, word, "negative")
        conditionalProbPos = estimateParameter(lexicon, model, word, "positive")
        
        negProb += math.log(conditionalProbNeg)
        posProb += math.log(conditionalProbPos)
      negProb = negProb + math.log(negPrior)
      posProb = posProb + math.log(posPrior)
      if posProb > negProb:
        posTrials += 1
        posAccuracy += 1
      else:
        posTrials += 1
  print "positive accuracy on training data=", float(posAccuracy) / float(posTrials)
  print "total accuracy on training data=", (float(posAccuracy) + float(negAccuracy))/(float(posTrials)+float(negTrials)), "\n"

#Exactly the same but it tests on only the testing files for 5.3b
def test(lexicon, model, vocabSize, testFiles, negation, adverbs):
  negTrials = 0
  negAccuracy = 0

  negPrior = float(model["totalNeg"]) / (float(model["totalNeg"])+float(model["totalPos"]))
  posPrior = float(model["totalPos"]) / (float(model["totalNeg"])+float(model["totalPos"]))
  
  for testfile in os.listdir(testFiles+"neg/"):
    if "cv8" in testfile or "cv9" in testfile:
      test = open(testFiles+"neg/"+testfile,"r")
      if negation:
        data = test.read()
        newdata = re.sub('n\'t\s|\snot\s',' not_', data,flags=re.IGNORECASE) #replaces nots and n'ts
        words = newdata.split()
        tokens = list(set([word for word in words if word in lexicon]))
      if adverbs:
        data = test.read()
        newdata = re.sub('\sextremely\s',' extremely_', data,flags=re.IGNORECASE) #replaces adverbs
        newdata = re.sub('\squite\s',' quite_', newdata,flags=re.IGNORECASE)
        newdata = re.sub('\sjust\s',' just_', newdata,flags=re.IGNORECASE)
        newdata = re.sub('\salmost\s',' almost_', newdata,flags=re.IGNORECASE)
        newdata = re.sub('\svery\s',' very_', newdata,flags=re.IGNORECASE)
        newdata = re.sub('\stoo\s',' too_', newdata,flags=re.IGNORECASE)
        newdata = re.sub('\senough\s',' enough_', newdata,flags=re.IGNORECASE)
        words = newdata.split()
        tokens = list(set([word for word in words if word in lexicon]))
      else:
        words = test.read().split()
        tokens = list(set([word for word in words if word in lexicon])) #filters out words not in the lexicon
      negProb = 0
      posProb = 0
      for word in tokens:
        conditionalProbNeg = estimateParameter(lexicon, model, word, "negative")
        conditionalProbPos = estimateParameter(lexicon, model, word, "positive")
        
        negProb += math.log(conditionalProbNeg) #update the probability of (x|negative)
        posProb += math.log(conditionalProbPos) #update the probability of (x|positive)

      negProb = negProb + math.log(negPrior) #add the prior for the full probability
      posProb = posProb + math.log(posPrior)
      if negProb > posProb: #counts the correct classifications
        negTrials += 1
        negAccuracy += 1
      else:
        negTrials += 1
  print "negative accuracy on testing data=", float(negAccuracy) / float(negTrials)

  posTrials = 0
  posAccuracy = 0
  for testfile in os.listdir(testFiles+"pos/"): #same thing for positive
    if "cv8" in testfile or "cv9" in testfile:
      test = open(testFiles+"pos/"+testfile,"r")
      if negation:
        data = test.read()
        newdata = re.sub('n\'t\s|\snot\s',' not_', data,flags=re.IGNORECASE) #replaces nots and n'ts
        words = newdata.split()
        tokens = list(set([word for word in words if word in lexicon]))
      if adverbs:
        data = test.read()
        newdata = re.sub('\sextremely\s',' extremely_', data,flags=re.IGNORECASE) #replaces adverbs
        newdata = re.sub('\squite\s',' quite_', newdata,flags=re.IGNORECASE)
        newdata = re.sub('\sjust\s',' just_', newdata,flags=re.IGNORECASE)
        newdata = re.sub('\salmost\s',' almost_', newdata,flags=re.IGNORECASE)
        newdata = re.sub('\svery\s',' very_', newdata,flags=re.IGNORECASE)
        newdata = re.sub('\stoo\s',' too_', newdata,flags=re.IGNORECASE)
        newdata = re.sub('\senough\s',' enough_', newdata,flags=re.IGNORECASE)
        words = newdata.split()
        tokens = list(set([word for word in words if word in lexicon]))
      else:
        words = test.read().split()
        tokens = list(set([word for word in words if word in lexicon])) #filters out words not in the lexicon
      negProb = 0
      posProb = 0
      for word in tokens:
        conditionalProbNeg = estimateParameter(lexicon, model, word, "negative")
        conditionalProbPos = estimateParameter(lexicon, model, word, "positive")
        
        negProb += math.log(conditionalProbNeg)
        posProb += math.log(conditionalProbPos)
      negProb = negProb + math.log(negPrior)
      posProb = posProb + math.log(posPrior)
      if posProb > negProb:
        posTrials += 1
        posAccuracy += 1
      else:
        posTrials += 1
  print "positive accuracy on testing data=", float(posAccuracy) / float(posTrials)
  print "total accuracy on testing data=", (float(posAccuracy) + float(negAccuracy))/(float(posTrials)+float(negTrials))


def main():
  parser = argparse.ArgumentParser()
  parser.usage = "python naiveBayes.py -lp opinion-lexicon-English/positive-words.txt -ln opinion-lexicon-English/negative-words.txt --train review_polarity/"
  parser.add_argument("--train", dest="trainFiles", action="store", help="Directory should contain two other directories with positive / negative training data separated")
  parser.add_argument("-lp", "--lexiconPos", dest="posLex", action="store", help="A lexicon of positive words")
  parser.add_argument("-ln", "--lexiconNeg", dest="negLex", action="store", help="A lexicon of negative words")
  parser.add_argument("--test", dest="test", action="store_true", default=False, help="Boolean for testing")
  parser.add_argument("-tw", "--topwords", dest="topWords", action="store_true", default=False, help="Boolean for top sentiment carrying words")
  parser.add_argument("-nh", "--negation", dest="negation", action="store_true", default=False, help="Boolean for negation handling")
  parser.add_argument("-ah", "--adverbs", dest="adverbs", action="store_true", default=False, help="Boolean for adverb handling")
  opts = parser.parse_args()
  if opts.posLex and opts.negLex: #prepares the lexicon if lexicon files are given
    lexicon = readLexicon(opts.posLex, opts.negLex, opts.negation, opts.adverbs)
  else:
    print "Please specify lexicon directory"
  if opts.trainFiles: #trains model if training data is given
  	model,vocabSize = train(lexicon, opts.trainFiles, opts.negation, opts.adverbs)
  else:
    print "Please specify training file directory"
  if opts.test: #tests model if testing data is given
    trainTest(lexicon, model, vocabSize, opts.trainFiles, opts.negation, opts.adverbs)
    test(lexicon, model, vocabSize, opts.trainFiles, opts.negation, opts.adverbs)
  if opts.topWords:
    topWords(lexicon, model)
 
main()