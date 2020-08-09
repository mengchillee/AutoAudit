####################################
# Author: Jeremy (Meng-Chieh) Lee  #
# Date	: 2020/05/04               #
####################################

import numpy as np
from scipy import sparse
from math import ceil
from collections import OrderedDict
import pickle
import copy
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import argparse

def log_star(x):
	"""
	Compute universal code length used for encode real number

	INPUTS
	Real number

	OUTPUTS
	Approximate universal code length of the real number
	"""
	return 2 * np.log2(x) + 1

def edgelist_to_matrix(edgelist):
	"""
	Transform edgelist to adjacency matrix

	INPUTS
	Edge List

	OUTPUTS
	Adjacency matrix and dictionary of nodes
	"""
	node_dict = {e: idx for idx, e in enumerate(np.unique(list(edgelist)))}
	ajm = np.zeros((len(node_dict), len(node_dict)))
	for e in edgelist:
		ajm[node_dict[e[0]]][node_dict[e[1]]] = 1
	return ajm, node_dict

def compute_mdl(ajm, order, start, count):
	"""
	Encode matrix by given order and the information of smurf patterns

	INPUTS
	ajm: Binary adjacency matrix
	order: Order used to reorder the matrix
	start: Start positions of each smurf pattern
	count: number of detected patterns and intermediaries

	OUTPUTS
	[Encoding description length, Purity]
	"""
	purity, mdl, n = [], 0, len(ajm)
	order.extend([i for i in range(n) if i not in order])
	ajm = np.array(ajm)[np.ix_(order, order)]

	### Encode sub-matrix A, B and C
	for idx in range(1, len(start)):
		s, e = start[idx-1], start[idx] - 1
		k = e - s + 1
		e1 = np.sum(ajm[s+1:e, s:e-1]) * (2 * ceil(np.log2(k - 1)))
		e2 = np.sum(ajm[e+1:-1, s:e]) * (ceil(np.log2(n)) + ceil(np.log2(n - k)))
		e3 = np.sum(ajm[s:e, e+1:-1]) * (ceil(np.log2(n)) + ceil(np.log2(n - k)))
		et = e1 + e2 + e3
		mdl += et
		sum_abc = np.sum(ajm[s:e, s:e]) + np.sum(ajm[e+1:-1, s:e]) + np.sum(ajm[s:e, e+1:-1])
		purity.append((k - 2) * 2 / sum_abc)

	### Encode sub-matrix D
	ajm = 1 - ajm
	mdl += np.sum(ajm[start[-1]:-1, start[-1]:-1]) * (2 * ceil(np.log2(n)))

	### Encode the real number of found patterns and intermediaries
	mdl += ceil(log_star(count[0])) + ceil(log_star(count[1]))
	### Encode the indexes of senders, receivers and intermediaries
	mdl += np.sum(count) * ceil(np.log2(n))
	### Encode for the start point of each pattern
	mdl += ceil(log_star(len(start) - 1))

	return mdl, np.mean(purity)

def AA_Smurf(ajm, max_iter, visualize):
	"""
	Identify the best order for spotting smurf pattern

	INPUTS
	ajm: Binary adjacency matrix
	max_iter: Maximum iteration to run the algorithm
	visualize: Path of visualization result

	OUTPUTS
	[Reordered matrix, Best order for reordering]
	"""
	### In case using 'cfd_injected.pkl' file
	# ajm, node_dict = edgelist_to_matrix(edgelist)

	### Get edge-pairs which have the number of intermediaries hgiher than the threshold c
	print('Get Edge-Pairs...')
	row, col = ajm, ajm.T
	edis = OrderedDict()
	dis_mtr = (sparse.csr_matrix(ajm) * sparse.csr_matrix(ajm)).todense()
	for idx1, idx2 in zip(*dis_mtr.nonzero()):
		val = dis_mtr[idx1, idx2]
		if val >= 3:
			edis[(idx1, idx2)] = [val, np.arange(len(row))[(row[idx1] + col[idx2]) == 2]]
	edis = OrderedDict(sorted(edis.items(), key=lambda t: t[1][0])[::-1])
	print('Done!\n')

	### Heuristically identify the best order by MDL and purity
	print('Identify Best Order...')

	def func(ajm, key, value, order, count, start, prev_mdl):
		if key[0] not in order and key[1] not in order:
			order.append(key[0])
			tmp_mid = [a for a in value[1] if a not in order]
			if len(tmp_mid) == 0:
				return -1, -1, -1, -1, -1
			order.extend(tmp_mid)
			order.append(key[1])
			start.append(len(order))
			mdl, purity = compute_mdl(ajm, order, start,
								[count[0] + 1, count[1] + len(tmp_mid), count[2] + 1])
			score = ((prev_mdl - mdl) / prev_mdl) * purity

			if mdl < prev_mdl:
				count = [count[0] + 1, count[1] + len(tmp_mid), count[2] + 1]
				return mdl, score, order, start, count
		return -1, -1, -1, -1, -1

	old_mdl = np.ceil(np.sum(1 - ajm)) * (2 * ceil(np.log2(len(ajm))))
	count_arr, order_arr, start_arr, mdl_arr = [[0, 0, 0]], [[]], [[0]], [old_mdl]
	iter = 0
	while True:
		prev_mdl = mdl_arr[-1]
		tmp_mdl, tmp_score, tmp_order, tmp_start, tmp_count = [], [], [], [], []

		results = Parallel(n_jobs=4)(
			[delayed(func)(ajm, key, value, \
						   copy.copy(order_arr[-1]), \
						   copy.copy(count_arr[-1]), \
						   copy.copy(start_arr[-1]), prev_mdl)
			 for idx, (key, value) in enumerate(edis.items())])
		tmp_mdl = [r[0] for r in results]
		tmp_score = [r[1] for r in results]
		tmp_order = [r[2] for r in results]
		tmp_start = [r[3] for r in results]
		tmp_count = [r[4] for r in results]

		### No more smurf-like pattern found
		if np.max(tmp_score) == -1 or (max_iter != None and iter > max_iter):
			break
		max_idx = np.argmax(tmp_score)
		count_arr.append(tmp_count[max_idx])
		order_arr.append(tmp_order[max_idx])
		start_arr.append(tmp_start[max_idx])
		mdl_arr.append(tmp_mdl[max_idx])
		iter += 1
	print('Done!\n')

	### Get the result with MDL 10% higher than the minimum bits
	max_idx = next(idx - 1 for idx, m in enumerate(mdl_arr) if m < mdl_arr[-1] * 1.1)
	count, order, start, ori_mdl = count_arr[max_idx], order_arr[max_idx], start_arr[max_idx], mdl_arr[max_idx]
	order.extend([i for i in range(len(ajm)) if i not in order])
	ro_ajm = ajm[np.ix_(order, order)]


	if visualize != None:
		print('Start Visualize Result...')
		plt.figure(figsize=(12, 5))
		plt.subplot(1, 2, 1)
		plt.matshow(ajm, fignum=False, cmap='binary')
		plt.title('Before Reordering')
		plt.subplot(1, 2, 2)
		plt.matshow(ro_ajm, fignum=False, cmap='binary')
		plt.title('After Reordering')
		plt.savefig(visualize)
		print('Done!\n')

	return ro_ajm, order

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Parameters for AA-Smurf of AutoAudit')
	parser.add_argument('--f', default='data/sample_matrix.txt', type=str, help='Input Path')
	parser.add_argument('--o', default='results/AA-Smurf_result.png', type=str, help='Output Path')
	parser.add_argument('--i', default=None, type=int, help='Maximum Iteration')
	args = parser.parse_args()

	ajm = np.loadtxt(args.f)
	ro_ajm, order = AA_Smurf(ajm, args.i, args.o)

	### In case using 'cfd_injected.pkl' file
	# with open(args.f, 'rb') as handle:
	# 	data = pickle.load(handle)
	# for k, v in data.items():
	# 	for idx, vv in enumerate(v):
	# 		ro_ajm, order = AA_Smurf(vv['Edgelist'], args.i, args.o)
