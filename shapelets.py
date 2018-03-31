import math
import numpy as np

import scipy.stats as stats
from sklearn.metrics.cluster import adjusted_rand_score
from operator import itemgetter


def compute_distance(s, D):
	dis = []
	s = stats.zcore(s)

	for i in range(1, len(D)):
		ts = D[i:]
		dis[i] = math.inf

		for j in range(1, len(ts) - len(s) + 1):
			z = stats.zscore(ts) # help idk
			d = np.linalg.norm(z, s)
			dis[i] = min(d, dis[i])

	return list(set([i/math.sqrt(len(s)) for i in dis]))


def compute_gap(s, D, k=3):
	dis = compute_distance(s, D)
	dis = sorted(dis)

	maxgap = 0
	dt = 0

	for l in range(0, abs(dis) - 1):
		d = (dis[l] + dis[l+1]) / 2
		D_A = find(dis < d)
		D_B = find(dis > d)

		r = len(D_A)/len(D_B)

		if (1/k) < (1 - (1/k)):
			m_A = np.mean(dis(D_A))
			m_B = np.mean(dis(D_B))
			s_A = np.std(dis(D_A))

			s_B = np.std(dis(D_B))

			gap = m_B - s_B - (m_A + s_A)

			if gap > maxgap:
				maxgap = gap
				dt = d
	return maxgap, dt 

def extract_shapelets(D, s_length):
	S = [] 
	ts = D[1:]

	while True:
		count = 0
		subsequences = []
		gap = [] # list of tuples (maxgap, dt)

		for sl in range(1, s_length):
			for i in range(1, len(ts) - sl + 1):
				subsequences[count + 1] = ts[:i:i + sl -1]
				gap[count + 1] = compute_gap(subsequences[count+1],D)

		index1, dt = max(gap)
		S = subsequences[index1]
		dis = compute_distance(subsequences[index1], D)

		D_lambda = find(dis < dt) #not sure what this is

		if len(D_lambda) == 1: break

		index2 = max(dis)
		ts = D[index2:]

		theta = np.mean(dis(D_lambda)) + np.std(dis(D_lambda))
		D = find(dis < theta)

	return S

# K-means helper functions 
def dist(p, q):
    (x1,y1) = p
    (x2,y2) = q
    return (x1-x2)**2 + (y1-y2)**2

def plus(args):
    p = [0,0]
    for (x,y) in args:
        p[0] += x
        p[1] += y
    return tuple(p)

def scale(p, c):
    (x,y) = p
    return (x/c, y/c)


# K-means 
def k_means(DIS, k):
	P = DIS
	M = DIS[0:k]

	OLD = []
	while OLD != M:
	    OLD = M

	    MPD = [(m, p, dist(m,p)) for (m, p) in product(M, P)]
	    PDs = [(p, dist(m,p)) for (m, p, d) in MPD]
	    PD = aggregate(PDs, min)
	    MP = [(m, p) for ((m,p,d), (p2,d2)) in product(MPD, PD) if p==p2 and d==d2]
	    MT = aggregate(MP, plus)

	    M1 = [(m, 1) for (m, _) in MP]
	    MC = aggregate(M1, sum)

	    M = [scale(t,c) for ((m,t),(m2,c)) in product(MT, MC) if m == m2]
	    return sorted(M)

def cluster_data(D, S, k, c=0):
	DIS = []
	cls[0] = c # default cluster label

	for count in range(1, len(S)):
		s = S # a u-shapelet
		dis = compute_distance(s, D)

		DIS = [dis] # pseudocode says [DIS dis]
		sum_dis = math.inf

		for i in range(1, n):
			IDX, SUMD = k_means(DIS, k)
			if sum(SUMD) < sum_dis:
				sum_dis = sum(SUMD)

				cls[count] = IDX
		CRI[count] = 1 - adjusted_rand_score(cls[count-1], cls[count])
	a = min(enumerate(cls), key=itemgetter(1))[0]
	return cls[a]

