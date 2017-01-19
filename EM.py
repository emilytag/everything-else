import math
def main():
  #open 3 files with ngram probabilities
  stream_a = open('mystery_A.fprobs.txt','r')
  stream_b = open('mystery_B.fprobs.txt', 'r')
  stream_c = open('mystery_C.fprobs.txt', 'r')
  list_a = []
  list_b = []
  list_c = []
  allstreams = []
  for line in stream_a:
    list_a.append(float(line.strip()))
  for line in stream_b:
    list_b.append(float(line.strip()))
  for line in stream_c:
    list_c.append(float(line.strip()))
  iterations = 0
  #list_d = [float(1)/float(8000)] * len(list_a)
  allstreams.append(list_a)
  allstreams.append(list_b)
  allstreams.append(list_c)
  #allstreams.append(list_c)
  #allstreams.append(list_d)

  #start with uniform lambdas
  lambdas = [float(1)/float(len(allstreams))] * len(allstreams)
  print lambdas
  last = [0] * len(allstreams)

  #run EM for 100 iterations
  while iterations <= 100:
    iterations += 1
    for j in range (0,len(lambdas)):
      update = 0
      last[j] = lambdas[j]
      for i in range(0,len(list_a)):
        num = lambdas[j] * allstreams[j][i]
        denom = 0
        for k in range(0,len(lambdas)):
          denom += lambdas[k] * allstreams[k][i]
        update += num/denom
      lambdas[j] = update/len(list_a)
    outside_t = 0
    outside_t1 = 0
    for i in range(0,len(list_a)):
      inside_t = 0
      inside_t1 = 0
      for k in range(0,len(lambdas)):
         inside_t += last[k] * allstreams[k][i]
         inside_t1 += lambdas[k] * allstreams[k][i]
      outside_t += math.log(inside_t)
      outside_t1 += math.log(inside_t1)
    l_t = outside_t/len(list_a)
    l_t1 = outside_t1/len(list_a)
    change = (l_t1 - l_t)/abs(l_t1)
    print "weights:", lambdas, "avg log-likelihood:", l_t1,  "ratio:", l_t1/l_t
    if change <= .00001:
      print "converged!!"
      break
main()
