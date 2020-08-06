####################################
# Author: Jeremy (Meng-Chieh) Lee  #
# Date	: 2020/05/04               #
####################################

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from scipy.spatial.distance import cosine
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import os
import argparse
np.seterr(divide='ignore', invalid='ignore')

fn = 12
pn = int((fn + 1) * fn / 2)

feature_names = ['Unique In Degree', 'Multi In Degree',
				 'Unique Out Degree', 'Multi Out Degree',
				 'Total In Weight', 'Mean In Weight',
				 'Median In Weight', 'Variance In Weight',
				 'Total Out Weight', 'Mean Out Weight',
				 'Median Out Weight', 'Variance Out Weight']


def total_mean_median_variance(arr):
	tol = np.sum(arr)
	if len(arr) != 0:
		avg, med, var = np.mean(arr), np.median(arr), np.var(arr)
	else:
		avg, med, var = 0, 0, 0
	return tol, avg, med, var

def generate_features(df):
	account_id = np.unique(df[['Source', 'Destination']].values)
	features = [[] for _ in range(fn)]

	for aid in account_id:
		tmp_src, tmp_dst = df[df['Source'] == aid], df[df['Destination'] == aid]

		features[0].append(len(tmp_dst['Source'].unique()))
		features[1].append(tmp_dst['Source'].count())
		features[2].append(len(tmp_src['Destination'].unique()))
		features[3].append(tmp_src['Destination'].count())

		inw = tmp_dst['Weight'].values
		tol, avg, med, var = total_mean_median_variance(inw)
		features[4].append(tol)
		features[5].append(avg)
		features[6].append(med)
		features[7].append(var)

		ouw = tmp_src['Weight'].values
		tol, avg, med, var = total_mean_median_variance(ouw)
		features[8].append(tol)
		features[9].append(avg)
		features[10].append(med)
		features[11].append(var)

	return account_id, np.array(features)

def generate_isolation_forest(df, ts, window_size, aid_dict):
	tdf = df[(df['Timestamp'] >= ts) & (df['Timestamp'] < ts + window_size)]
	aid, features = generate_features(tdf)

	plot = np.zeros((len(aid_dict), pn))
	arr1, arr2, num = [], [], 0
	focus_plots, fea_arr, aid_arr = [], [], []
	for f1 in range(len(features)):
		for f2 in range(f1 + 1, len(features)):
			nzidx = np.where((features[f1] != 0) & (features[f2] != 0))[0]
			tmp_aid = aid[nzidx]
			if len(nzidx) != 0:
				f = np.array([np.log10(features[f1][nzidx] + 1), np.log10(features[f2][nzidx] + 1)]).T
				cls = IsolationForest(n_estimators=100, contamination='auto', behaviour='new')
				cls.fit(f)
				scores = np.array([-1 * s + 0.5 for s in cls.decision_function(f)])
				for idx, s in zip(tmp_aid, scores):
					plot[aid_dict[idx]][num] += s
			arr1.append(f)
			arr2.append(tmp_aid)
			num += 1

	return plot, arr1, arr2

def generate_focus_plots(df, window_size, overlap=0.5):
	account_id = np.unique(df[['Source', 'Destination']].values)
	aid_dict = {a: idx for idx, a in enumerate(account_id)}
	move = int(window_size * overlap)

	ts_range = [[ts, ts + window_size] for ts in range(df['Timestamp'].min(), df['Timestamp'].max() - window_size, move)]
	fea_dict, num = {}, 0
	for f1 in range(fn):
		for f2 in range(f1 + 1, fn):
			fea_dict[num] = (f1, f2)
			num += 1

	results = Parallel(n_jobs=4)(
		[delayed(generate_isolation_forest)(df, ts, window_size, aid_dict)
		 for ts in range(df['Timestamp'].min(), df['Timestamp'].max() - window_size, move)])
	focus_plots = [r[0] for r in results]
	fea_arr = [r[1] for r in results]
	aid_arr = [r[2] for r in results]

	return focus_plots, account_id, fea_arr, fea_dict, aid_arr, aid_dict, ts_range

def sketching(focus_plots, sketch_num, account_id, num_dst, s_rate=0.995, d_rate=0.8, seed=0):
	np.random.seed(seed)

	permutation = []
	sum_graph = np.sum(np.array(focus_plots), axis=0)
	for ski in range(sketch_num):
		src_id = [idx for idx in range(len(account_id)) if np.random.random_sample() > s_rate]
		ori_dst_id = [idx for idx in range(pn) if np.random.random_sample() > d_rate]
		dst_id = []
		for _ in range(min(len(ori_dst_id), num_dst)):
			score = np.zeros(len(ori_dst_id))
			max_s, max_idx = 0, 0
			for did in ori_dst_id:
				tmp_did, s = np.concatenate([dst_id, [did]]).astype(int), 0
				for sid in src_id:
					s += np.max(sum_graph[sid, tmp_did])
				if s > max_s:
					max_idx, max_s  = did, s
			ori_dst_id.remove(max_idx)
			dst_id.append(max_idx)
		permutation.append([src_id, dst_id])

	sketches = []
	for plot in focus_plots:
		sketch = []
		for src_id, dst_id in permutation:
			x = 0
			for sid in src_id:
				for did in dst_id:
					x += plot[sid, did]
			sketch.append(x)
		sketches.append(sketch)

	return sketches, permutation

def find_past_pos(fea_arr, aid_arr, gid, did, back, sorted_sid):
	pos = [[0, 0] for _ in range(len(sorted_sid))]
	for ggid in range(gid-back, gid):
		f = fea_arr[ggid][did]
		aaid = aid_arr[ggid][did]
		for idx1, s in enumerate(aid_arr[gid][did][sorted_sid]):
			for idx2, a in enumerate(aaid):
				if a == s:
					pos[idx1][0] += f[idx2, 0]
					pos[idx1][1] += f[idx2, 1]
					break
		return np.array(pos) / back

def twod_plot(op, x, y, xname, yname, title, past_pos=None, anomaly=None, account_id=None):
	x, y = x + 1, y + 1
	fig = plt.figure(figsize=(10, 6))
	plt.suptitle(title)
	bins = np.array([2**i for i in range(int(np.log2(np.max(x))) + 2)])
	digitized = np.digitize(x, bins, right=True)
	bin_means = [y[digitized == i].mean() if len(y[digitized == i]) != 0 else 0 for i in range(1, len(bins))]
	bins = (bins[:-1] + bins[1:]) / 2

	reg = LinearRegression(normalize=True).fit(np.log(x).reshape(-1, 1), np.log(y).reshape(-1, 1))
	a = reg.coef_[0][0]
	b = reg.intercept_[0]
	pred = reg.predict(np.log(x).reshape(-1, 1))
	r2 = r2_score(np.log(y).reshape(-1, 1), pred)

	plt.scatter(x, y, s=3, c='b')
	plt.scatter(x, y, s=125, c='b', edgecolors='none', alpha=0.25)

	for p, a in zip(past_pos, anomaly):
		plt.scatter(x[a], y[a], s=3, c='r')
		plt.scatter(x[a], y[a], s=125, c='r', edgecolors='none', alpha=0.5)
		plt.annotate(account_id[a], (x[a], y[a]), c='r', size=20)

		plt.scatter(p[0] + 1, p[1] + 1, s=3, c='purple')
		plt.scatter(p[0] + 1, p[1] + 1, s=125, c='purple', edgecolors='none', alpha=0.5)
		plt.arrow(p[0] + 1, p[1] + 1, x[a] - p[0] - 1, y[a] - p[1] - 1, length_includes_head=True, head_width=0.5, head_length=0.5, color='purple', alpha=0.6)

	plt.plot(bins, bin_means, c='limegreen', marker='o', alpha=0.25, markersize=5, label='Mean of P2 Bins')
	plt.plot(x, np.exp(pred), c='r', alpha=1, linewidth=3, label='Fit Line')
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel(xname + '\nSlope: ' + str(round(a, 3)) + '\nR2: ' + str(round(r2, 3)), size=20)
	plt.ylabel(yname, size=20)
	plt.legend()
	fig.savefig(os.path.join(op, title + '.png'))

def AA_AR(df, op, window_size, sketch_num, back_ws, num_dst, plot_acc):
	print('Generate Focus Plots...')
	focus_plots, account_id, fea_arr, fea_dict, aid_arr, aid_dict, ts_range = generate_focus_plots(df, window_size)
	print()

	print('Generate Sketches...')
	sketches, permutation = sketching(focus_plots, sketch_num, account_id, num_dst)
	print()

	z, max_arr = [], []
	for i in range(back_ws, len(sketches)):
		ev, sk = np.abs(np.linalg.svd(np.array(sketches[i-back_ws:i-1]).T)[0][:, 0].T), np.array(sketches[i])
		ev, sk = ev / np.sum(ev), sk / np.sum(sk)
		z.append(cosine(ev, sk))
		max_arr.append(np.argmax((sk - ev) / ev))

	print('Print Change Score...')
	plt.figure(figsize=(12, 4))
	plt.plot(z, c='b')
	plt.xlabel('Timestamp')
	plt.ylabel('Change Score')
	plt.savefig(os.path.join(op, 'change_score.png'))
	print()

	print('Print Attention Routing...')
	ts = np.argmax(z)
	gid = ts + back_ws
	src_id, dst_id = permutation[max_arr[ts]]
	for idx, did in enumerate(dst_id):
		f1, f2 = fea_arr[gid][did][:, 0], fea_arr[gid][did][:, 1]
		f1_idx, f2_idx = fea_dict[did]
		sorted_sid = np.argsort(np.array([focus_plots[gid][aid_dict[sid]][did] for sid in aid_arr[gid][did]]))[::-1][:plot_acc]
		past_pos = find_past_pos(fea_arr, aid_arr, gid, did, back_ws, sorted_sid)
		title = 'Timestamp ' + str(ts_range[gid][0]) + ' ~ Timestamp ' + str(ts_range[gid][1]) + ' - Figure ' + str(idx)
		twod_plot(op, f1, f2, feature_names[f1_idx], feature_names[f2_idx], title, past_pos, sorted_sid, aid_arr[gid][did])
	print()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Parameters for AA-AR of AutoAudit')
	parser.add_argument('--f', default='data/sample_edges.csv', type=str, help='Input Path')
	parser.add_argument('--o', default='results/', type=str, help='Output Path')
	parser.add_argument('--w', default=14, type=int, help='Window Size')
	parser.add_argument('--s', default=256, type=int, help='Sketch Number')
	parser.add_argument('--b', default=4, type=int, help='Backtrack Window')
	parser.add_argument('--a', default=3, type=int, help='Attention Figure Number')
	parser.add_argument('--c', default=3, type=int, help='Attention Account Number')
	args = parser.parse_args()

	df = pd.read_csv(args.f, dtype=int, skiprows=1, names=['Source', 'Destination', 'Weight', 'Timestamp'])
	AA_AR(df, args.o, args.w, args.s, args.b, args.a, args.c)
