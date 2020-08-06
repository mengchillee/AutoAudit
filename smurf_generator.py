####################################
# Author: Jeremy (Meng-Chieh) Lee  #
# Date	: 2020/05/04               #
####################################

import pandas as pd
import numpy as np
import copy
import pickle
import argparse

def generate_eval_df(el, bank_acc, client_acc, middle_num, folds=10, seed=0):
	np.random.seed(seed)
	evaluate = []
	for _ in range(folds):
		src, dst = np.random.choice(client_acc, 2)
		middle_acc = np.random.choice(bank_acc, middle_num)
		edgelist = copy.copy(el)

		for mcc in middle_acc:
			edgelist.add((src, mcc))
			edgelist.add((mcc, dst))

		noise_num = np.random.randint(1, 6)
		for _ in range(noise_num):
			r = np.random.randint(11)
			mn = middle_num + (5 - r)
			n_src, n_dst = np.random.choice([ca for ca in client_acc if ca != src and ca != dst], 2)
			n_middle = np.random.choice(bank_acc, mn)
			for mcc in n_middle:
				edgelist.add((n_src, mcc))
				edgelist.add((mcc, n_dst))

				for c in n_middle:
					### Interaction between intermediaries
					if c != mcc and np.random.randint(1, 11) > r / 2.5 + 6:
						edgelist.add((mcc, c))

		evaluate.append({'Edgelist': edgelist, 'Label': [src, dst]})
	return evaluate

def smurf_generator(file_name, interact_prob, folds=10, seed=0):
	np.random.seed(seed)

	old_df = pd.read_csv('data/raw_cfd_trans.csv', sep=';', usecols=[1, 2, 4, 5, 9]).dropna()
	old_df = old_df[old_df['operation'].isin(['PREVOD Z UCTU', 'PREVOD NA UCET'])]
	bank_acc = old_df['account_id'].unique().astype(int)
	client_acc = old_df['account'].unique().astype(int)

	edgelist = set()
	for i in old_df.values:
		if i[2] == 'PREVOD Z UCTU':
			edgelist.add((int(i[4]), int(i[0])))
		else:
			edgelist.add((int(i[0]), int(i[4])))

	for s in bank_acc:
		for d in bank_acc:
			if np.random.random() > interact_prob:
				edgelist.add((s, d))

	data = {}
	for i in range(10, 51, 10):
		data[i] = generate_eval_df(edgelist, bank_acc, client_acc, i, folds=folds, seed=seed)

	with open(file_name + '.pkl', 'wb') as handle:
		pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Parameters for "Smurfing" Generator')
	parser.add_argument('--f', default='data/cfd_injected', type=str, help='Output File Name')
	parser.add_argument('--i', default=0.9995, type=int, help='Interation Probability')
	parser.add_argument('--l', default=10, type=int, help='Fold Number')
	parser.add_argument('--r', default=0, type=int, help='Random Seed')
	args = parser.parse_args()

	smurf_generator(args.f, args.i, args.l, args.r)
