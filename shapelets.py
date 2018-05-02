import math
import numpy as np

from scipy import stats
from sklearn.metrics.cluster import adjusted_rand_score
from operator import itemgetter

# Based on the pseudocode found on pgs 5-6 of this paper:
# http://www.cs.ucr.edu/~eamonn/ClusteringTimeSeriesUsingUnsupervised-Shapelets.pdf

# Compute distance vector 
# s = subsequence, D = dataset
def compute_distance(s, D):
	print(s)
	print(D)
	return
	dis = [None for i in range(0, len(D))]
	s = stats.zscore(s)
	print(s)
	for i in range(0, len(D)):
		ts = D[i:]
		dis[i] = 1000000 # represents inf

		for j in range(1, len(ts) - len(s) + 1):
			z = stats.zscore(ts[j:j+len(s)])
			print(z)
			print(s)
			d = np.linalg.norm(z, s)
			print(d)
			dis[i] = min(d, dis[i])
			print(dis)

	return list(set([i/math.sqrt(len(s)) for i in dis]))


# s = u-shapelet, D = dataset 
def compute_gap(s, D, k=3):
	dis = compute_distance(s, D)
	print(dis)
	dis = sorted(dis)

	maxgap = 0
	dt = 0

	for l in range(0, abs(dis) - 1):
		d = (dis[l] + dis[l+1]) / 2
		D_A = [i for i in dis if i < d]
	
		D_B = [i for i in dis if i > d]

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

# Extract unsupervised shapelets 
# D = dataset, s_length = shapelet length 
def extract_shapelets(D, s_length):
	S = [] 
	ts = D[1:]

	while True:
		count = 0
		subsequences = [None for i in range(0,len(D)**2)]
		gap = [] # list of tuples (maxgap, dt)

		for sl in range(1, s_length):
			for i in range(1, len(ts) - sl + 1):
				subsequences[count + 1] = ts[i:i + sl-1]
				gap[count + 1] = compute_gap(subsequences[count+1],D)

		index1, dt = max(gap)
		S = subsequences[index1]
		dis = compute_distance(subsequences[index1], D)

		D_lambda = [i for i in dis if i < dt]

		if len(D_lambda) == 1: break

		index2 = max(dis)
		ts = D[index2:]

		theta = np.mean(dis(D_lambda)) + np.std(dis(D_lambda))
		D = [i for i in dis if i < theta]

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

# Cluster time series
# D = dataset, S = set of unsupervised shapelets, k = number of clusters
def cluster_data(D, S, k, c=0):
	DIS = []
	cls[0] = c # default cluster label

	for count in range(1, len(S)):
		s = S # a u-shapelet
		dis = compute_distance(s, D)

		DIS += [dis] # pseudocode says [DIS dis]
		sum_dis = math.inf

		for i in range(1, n):
			IDX, SUMD = k_means(DIS, k)
			if sum(SUMD) < sum_dis:
				sum_dis = sum(SUMD)

				cls[count] = IDX
		CRI[count] = 1 - adjusted_rand_score(cls[count-1], cls[count])
	a = min(enumerate(cls), key=itemgetter(1))[0]
	return cls[a]


extract_shapelets([1,2,3,4],2)