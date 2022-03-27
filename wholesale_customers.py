import os
import itertools
import sys
import pandas as pd
import sklearn.cluster as cluster
import matplotlib.pyplot as plt
import sklearn.metrics as metrics


DATA_FILE = 'wholesale_customers.csv'
OUTPUT_FILE = 'log.txt'
# define the number of clusters
K = [3, 5, 10]


# Part 2: Cluster Analysis

# Return a pandas dataframe containing the data set that needs to be extracted from the data_file.
# data_file will be populated with the string 'wholesale_customers.csv'.
def read_csv_2(data_file):
	try:
		# open data file in csv format
		df = pd.read_csv(data_file)
	# handle exceptions:
	except IOError as ioe:
		print('there was an I/O error trying to open the data file: ' + str(ioe))
		sys.exit()
	except Exception as ex:
		print('there was an error: ' + str(ex))
		sys.exit()
	# drop attributes Channel and Region
	df = df.drop(columns=['Channel'])
	df = df.drop(columns=['Region'])
	return df


# Return a pandas dataframe with summary statistics of the data.
# Namely, 'mean', 'std' (standard deviation), 'min', and 'max' for each attribute.
# These strings index the new dataframe columns. 
# Each row should correspond to an attribute in the original data and be indexed with the attribute name.
def summary_statistics(df):
	stat_name = ['mean', 'std', 'min', 'max']
	attr_name = df.columns.values.tolist()
	mean = df.mean()
	std = df.std()
	min_ = df.min()
	max_ = df.max()
	summary = pd.DataFrame(list(zip(mean, std, min_, max_)), index=attr_name, columns=stat_name)
	summary = summary.round(0)
	return summary.astype(int)


# Given a dataframe df with numeric values, return a dataframe (new copy)
# where each attribute value is subtracted by the mean and then divided by the
# standard deviation for that attribute.
def standardize(df):
	stand_df = (df - df.mean())/df.std()
	return stand_df


# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans.
# y should contain values in the set {0,1,...,k-1}.
def kmeans(df, k):
	# use random as initializing
	km = cluster.KMeans(n_clusters=k, init='random', n_init=10)
	km.fit(df)
	y = pd.Series(km.labels_)
	return y


# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans++.
# y should contain values from the set {0,1,...,k-1}.
def kmeans_plus(df, k):
	# use kmeans++ as initializing
	km = cluster.KMeans(n_clusters=k, n_init=10)
	km.fit(df)
	y = pd.Series(km.labels_)
	return y


# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using agglomerative hierarchical clustering.
# y should contain values from the set {0,1,...,k-1}.
def agglomerative(df, k):
	ac = cluster.AgglomerativeClustering(n_clusters=k, linkage='average')
	ac.fit(df)
	y = pd.Series(ac.labels_)
	return y


# Given a data set X and an assignment to clusters y
# return the Silhouette score of the clustering.
def clustering_score(X,y):
	return metrics.silhouette_score(X, y, metric='euclidean')


# Perform the cluster evaluation described in the coursework description.
# Given the dataframe df with the data to be clustered,
# return a pandas dataframe with an entry for each clustering algorithm execution.
# Each entry should contain the: 
# 'Algorithm' name: either 'Kmeans' or 'Agglomerative', 
# 'data' type: either 'Original' or 'Standardized',
# 'k': the number of clusters produced,
# 'Silhouette Score': for evaluating the resulting set of clusters.
def cluster_evaluation(df):
	k = [3, 5, 10]
	columns = ['Algorithm', 'data', 'k', 'Silhouette Score']
	std_df = standardize(df)
	algorithm = []
	datas = []
	k_num = []
	scores = []
	for i in k:
		for j in range(4):
			if j % 2 == 0:
				algorithm.append('Kmeans')
			else:
				algorithm.append('Agglomerative')
			if j < 2:
				datas.append('Original')
			else:
				datas.append('Standardized')
			k_num.append(i)
		scores.append(clustering_score(df, kmeans(df, i)))
		scores.append(clustering_score(df, agglomerative(df, i)))
		scores.append(clustering_score(std_df, kmeans(std_df, i)))
		scores.append(clustering_score(std_df, agglomerative(std_df, i)))
	eval_dict = {columns[0]: algorithm, columns[1]: datas, columns[2]: k_num, columns[3]: scores}
	evaluation = pd.DataFrame(eval_dict, columns=columns)
	return evaluation


# Given the performance evaluation dataframe produced by the cluster_evaluation function,
# return the best computed Silhouette score.
def best_clustering_score(rdf):
	return rdf['Silhouette Score'].max()


# Run some clustering algorithm of your choice with k=3 and generate a scatter plot for each pair of attributes.
# Data points in different clusters should appear with different colors.
def scatter_plots(df):
	k = 3
	attr_names = df.columns.values.tolist()
	# get 15 pairs of attributes
	combinations = list(itertools.combinations(attr_names, 2))
	# get the best clustering algorithm
	max_id = cluster_evaluation(df)['Silhouette Score'].idxmax()
	max_row = cluster_evaluation(df).iloc[max_id, :]
	if max_row[0] == 'Agglomerative':
		if max_row[1] == 'Original':
			cluster = agglomerative(df, k)
		else:
			cluster = agglomerative(standardize(df), k)
	else:
		if max_row[1] == 'Original':
			cluster = kmeans(df, k)
		else:
			cluster = kmeans(standardize(df), k)
	# plot the scatter
	for i in range(15):
		save_path = './image'
		x_label = combinations[i][0]
		y_label = combinations[i][1]
		plt.xlabel(x_label)
		plt.ylabel(y_label)
		plt.scatter(df[x_label], df[y_label], c=cluster)
		if not os.path.exists(save_path):
			os.makedirs(save_path)
		plt.savefig(os.path.join(save_path, str(i)))
		plt.show()


# MAIN
# print log in the file
f = open(OUTPUT_FILE, 'a')
sys.stdout = f
sys.stderr = f
df = read_csv_2(DATA_FILE)
print("------------------Part 2--------------------")
print("--------original dataframe-----------")
print(df)
print("--------summary statistics-----------")
print(summary_statistics(df))
print("--------standardize-----------")
standardize_data = standardize(df)
print(standardize_data)
print("--------k means-----------")
for i in K:
	clu = kmeans(standardize_data, i)
	print(str(i) + ' means clustering:')
	print(clu)
	print('clustering score: ')
	print(clustering_score(standardize_data, clu))
print("--------k means++ -----------")
for i in K:
	clu = kmeans_plus(standardize_data, i)
	print(str(i) + ' means clustering:')
	print(clu)
	print('clustering score: ')
	print(clustering_score(standardize_data, clu))
print("--------agglomerative clustering-----------")
for i in K:
	clu = agglomerative(standardize_data, i)
	print(str(i) + ' agglomerative clustering:')
	print(clu)
	print('clustering score: ')
	print(clustering_score(standardize_data, clu))
print("---------clustering evaluation-----------")
evaluation = cluster_evaluation(df)
print(evaluation)
print("---------best score-----------")
print(best_clustering_score(evaluation))
scatter_plots(df)
