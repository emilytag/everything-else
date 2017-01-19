import re
import numpy
import math
import random
import sys
import cPickle

#a class for computing probability distribution across a map of counts
class ProbDistribution(object):
  def __init__(self, countMap):
    self.countMap = countMap
    self.total = float(sum(self.countMap.values()))
    self.countMap =  { k: math.log((v / self.total)) for k, v in self.countMap.items() }

  def prob(self, target):
    return self.countMap.get(target, float("-inf"))

  def addzero(self, key):
  	self.countMap[key] = float("-inf")

#a class for computing a conditional probability distribution
class CondProbDistribution(object):
  def __init__(self, pd):
    self.pd = pd

  def prob(self, target, given):
    total = self.pd[given]
    return total.prob(target)

def clean(data):
	data = data.upper()
	data = re.sub('[^A-Z]', " ", data)
	data = re.sub('\s+', " ", data)
	return data

def probs(data):
	unigramProbs = {"V": {}, "C": {}}
	transitionProbs = {"V": {}, "C": {}}
	vowels = ["A", "E", "I", "O", "U", "Y", " "]
	consonants = ["B", "C", "D", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "X", "Z"]
	prev = ""
	for letter in data:
		if letter in vowels:
			letterType = "V"
			try:
				unigramProbs["V"][letter] += 1
			except:
				unigramProbs["V"][letter] = 1
		if letter in consonants:
			letterType = "C"
			try:
				unigramProbs["C"][letter] += 1
			except:
				unigramProbs["C"][letter] = 1
		if prev != "":
			try:
				transitionProbs[prev][letterType] += 1
			except:
				transitionProbs[prev][letterType] = 1
		prev = letterType
	for state in transitionProbs:
		transitionProbs[state] = ProbDistribution(transitionProbs[state])
	for state in unigramProbs:
		unigramProbs[state] = ProbDistribution(unigramProbs[state])
	transitions = CondProbDistribution(transitionProbs)
	emissions = CondProbDistribution(unigramProbs)

	return emissions, transitions

def forwardbackward(data, emissions, transitions):
	iterations = 50
	pi = [math.log(1.0), float("-inf")]
	V = len(data)
	letterTypes = ["C", "V"]
	last = 0.0
	for i in range(0, iterations):
		#print "****NEW ITERATION****"
		#print pi
		all_alphas = []
		alphas_initial = []
		for i in range(0,len(pi)):
			letter = data[0]
			try:
				alphas_initial.append(pi[i]+emissions.prob(letter, letterTypes[i]))
			except ValueError:
				alphas_initial.append(float("-inf"))
		#print alphas_initial
		all_alphas.append(alphas_initial)
		for t in range(1, V):
			new_alpha = ["x", "x"]
			for j in range(0,len(pi)):
				f = 0.0
				for i in range(0,len(pi)):
					#print all_alphas[t-1][i], transitions.prob(letterTypes[i], letterTypes[j])
					x = all_alphas[t-1][i] + transitions.prob(letterTypes[i], letterTypes[j])
					if f != 0.0:
						f = numpy.logaddexp(f, x)
					else:
						f = x
				try:
					new_alpha[j] = f + emissions.prob(data[t], letterTypes[j])
				except ValueError:
					new_alpha[j] = float("-inf")
			#print "new alpha", new_alpha
			all_alphas.append(new_alpha)

		all_betas = [0] * V
		beta_initial = [math.log(1.0), math.log(1.0)]
		all_betas[V-1] = beta_initial
		for t in range(V-2, -1, -1):
			new_beta = ["x", "x"]
			for i in range(0,len(pi)):
				f = 0.0
				for j in range(0,len(pi)):
					x = transitions.prob(letterTypes[i], letterTypes[j]) + emissions.prob(data[t+1], letterTypes[j]) + all_betas[t+1][j]
					#print all_betas[t+1][j], emissions.prob(data[t+1], letterTypes[j]), transitions.prob(letterTypes[i], letterTypes[j]), x
					if f != 0.0:
						f = numpy.logaddexp(f, x)
					else:
						f = x
				try:
					new_beta[i] = f
				except ValueError:
					new_beta[i] = float("-inf")
			#print new_beta
			all_betas[t] = new_beta
		#pcll = max(all_betas[0])
		#print "pcll-beta", pcll / V

		all_ksi = []
		denom = 0.0
		#print all_alphas[-1]
		for x in all_alphas[-1]:
			if denom != 0.0:
				#print denom, x
				denom = numpy.logaddexp(denom, x)
			else:
				denom = x
		#print denom
		for t in range(1, V-1):
			new_ksi = ["x", "x"]
			for i in range(0,len(pi)):
				new_ksi_i = ["x", "x"]
				for j in range(0,len(pi)):
					try:
						ksi = (all_alphas[t][i]+transitions.prob(letterTypes[i], letterTypes[j])+emissions.prob(data[t+1], letterTypes[j])+all_betas[t+1][j]) - denom
					except ValueError:
						ksi = float("-inf")
					new_ksi_i[j] = ksi
				new_ksi[i] = new_ksi_i
			all_ksi.append(new_ksi)
		#for x in all_ksi:
			#print x
		all_gamma = []
		for t in range(0, V):
			new_gamma = ["x", "x"]
			for i in range(0,len(pi)):
				try:
					gamma = (all_alphas[t][i]+all_betas[t][i]) - denom
				except ValueError:
					gamma = float("-inf")
				new_gamma[i] = gamma
			all_gamma.append(new_gamma)
			#print new_gamma
		#print all_gamma[0]
		pi, transitions, emissions = baumwelch(all_ksi, all_gamma, pi, V, letterTypes, transitions, emissions, data)
		pcll = max(all_alphas[-1])
		print pcll / V, pcll / V - last
		last = pcll / V
	
	print "c given c", transitions.prob("C", "C")
	print "c given v", transitions.prob("C", "V")
	print "v given v", transitions.prob("V", "V")
	print "v given c", transitions.prob("V", "C")
	print "\n"
	for w in sorted(emissions.pd["C"].countMap.items(), key=lambda x: x[1], reverse=True):
		print w[0], w[1]
	print "\n"
	for w in sorted(emissions.pd["V"].countMap.items(), key=lambda x: x[1], reverse=True):
		print w[0], w[1]
	
	outfile = open('emissions.pickle', 'wb')
	fastPickler = cPickle.Pickler(outfile, cPickle.HIGHEST_PROTOCOL)
	fastPickler.fast = 1
	fastPickler.dump(emissions)
	outfile.close()
	outfile1 = open('transitions.pickle', 'wb')
	fastPickler = cPickle.Pickler(outfile1, cPickle.HIGHEST_PROTOCOL)
	fastPickler.fast = 1
	fastPickler.dump(transitions)
	outfile1.close()
	outfile1 = open('pi.pickle', 'wb')
	fastPickler = cPickle.Pickler(outfile1, cPickle.HIGHEST_PROTOCOL)
	fastPickler.fast = 1
	fastPickler.dump(pi)
	outfile1.close()
	

def evaluate(data, emissions, transitions, pi):
	V = len(data)
	letterTypes = ["C", "V"]
	all_alphas = []
	alphas_initial = []
	for i in range(0,len(pi)):
		letter = data[0]
		try:
			alphas_initial.append(pi[i]+emissions.prob(letter, letterTypes[i]))
		except ValueError:
			alphas_initial.append(float("-inf"))
	#print alphas_initial
	all_alphas.append(alphas_initial)
	for t in range(1, V):
		new_alpha = ["x", "x"]
		for j in range(0,len(pi)):
			f = 0.0
			for i in range(0,len(pi)):
				#print all_alphas[t-1][i], transitions.prob(letterTypes[i], letterTypes[j])
				x = all_alphas[t-1][i] + transitions.prob(letterTypes[i], letterTypes[j])
				if f != 0.0:
					f = numpy.logaddexp(f, x)
				else:
					f = x
			try:
				new_alpha[j] = f + emissions.prob(data[t], letterTypes[j])
			except ValueError:
				new_alpha[j] = float("-inf")
		#print "new alpha", new_alpha
		all_alphas.append(new_alpha)
	pcll = max(all_alphas[-1])
	print pcll / V

def viterbi(data, emissions, transitions, pi):
	V = len(data)
	letterTypes = ["C", "V"]
	all_alphas = []
	alphas_initial = []
	for i in range(0,len(pi)):
		letter = data[0]
		try:
			alphas_initial.append(pi[i]+emissions.prob(letter, letterTypes[i]))
		except ValueError:
			alphas_initial.append(float("-inf"))
	all_alphas.append(alphas_initial) 
	for t in range(1, V):
		new_alpha = ["x", "x"]
		for i in range(0,len(pi)):
			for j in range(0,len(pi)):
				try:	
					new_alpha[j] = max(all_alphas[t-1]) + transitions.prob(letterTypes[j], letterTypes[i]) + emissions.prob(data[t], letterTypes[j])
				except ValueError:
					new_alpha[j] = float("-inf")
		print "new alpha", new_alpha
		all_alphas.append(new_alpha)
	for k in range(0, V):
		print data[k], all_alphas[k].index(max(all_alphas[k]))

def simpledecode(data, emissions):
	states = ["C", "V"]
	for letter in data:
		poss = [0,0]
		for i in range(0, len(states)):
			poss[i] = emissions.prob(letter, states[i])
		print letter, poss.index(max(poss))


def baumwelch(all_ksi, all_gamma, pi, V, letterTypes, transitions, emissions, data):
	for i in range(0,len(pi)):
		pi[i] = all_gamma[0][i]
	for i in range(0,len(pi)):
		for j in range(0,len(pi)):
			ksi_sum = 0
			gamma_sum = 0
			for t in all_ksi:
				ksi_sum = numpy.logaddexp(ksi_sum, t[i][j])
			for t in all_gamma:
				gamma_sum = numpy.logaddexp(gamma_sum, t[i])
			transitions.pd[letterTypes[j]].countMap[letterTypes[i]] = ksi_sum - gamma_sum
	for j in range(0,len(pi)):
		for k in emissions.pd[letterTypes[j]].countMap:
			num_gamma = 0
			denom_gamma = 0
			for t in range(0,len(all_gamma)):
				denom_gamma = numpy.logaddexp(denom_gamma, all_gamma[t][j])
				if data[t] == k:
					num_gamma = numpy.logaddexp(num_gamma, all_gamma[t][j])
			emissions.pd[letterTypes[j]].countMap[k] = num_gamma - denom_gamma
	return pi, transitions, emissions

#CCVVCVCVCVVVCVCVCCCV
#THE FAMILY OF DASHWO
def main():
	alphabet = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", " "]
	if sys.argv[1] != "test":
		data = open("hmm-train-japanese.txt","r").read()
		cleandata = clean(data)
		emissions, transitions = probs(cleandata)

		randEmissions = {"V": {}, "C": {}}
		randTransitions = {"V": {}, "C": {}}
		for c in emissions.pd.keys():
			for letter in alphabet:
				randEmissions[c][letter] = random.randrange(1, 100)
		for d in transitions.pd.keys():
			for e in transitions.pd.keys():
				randTransitions[d][e] = random.randrange(1, 100)
		for state in randTransitions:
			randTransitions[state] = ProbDistribution(randTransitions[state])
		for state in randEmissions:
			randEmissions[state] = ProbDistribution(randEmissions[state])
		rtransitions = CondProbDistribution(randTransitions)
		remissions = CondProbDistribution(randEmissions)
		forwardbackward(cleandata, remissions, rtransitions)
		
	else:
		FILE = open("emissions.pickle", 'r')
		f = FILE.read()
		emissions = cPickle.loads(f)
		FILE.close()
		FILE1 = open("transitions.pickle", 'r')
		f1 = FILE1.read()
		transitions = cPickle.loads(f1)
		FILE1.close()
		FILE2 = open("pi.pickle", 'r')
		f2 = FILE2.read()
		pi = cPickle.loads(f2)
		FILE2.close()
		data = open(sys.argv[2],"r").read()
		cleandata = clean(data)
		evaluate(cleandata, emissions, transitions, pi)
		#viterbi(cleandata, emissions, transitions, pi)
		#simpledecode(cleandata, emissions)
main()
