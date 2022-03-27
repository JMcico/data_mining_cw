import copy
import sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import sklearn.tree as tree

DATA_FILE = 'adult.csv'
OUTPUT_FILE = 'log.txt'

# Part 1: Decision Trees with Categorical Attributes

# Return a pandas dataframe containing the data set that needs to be extracted from the data_file.
# data_file will be populated with the string 'adult.csv'.
def read_csv_1(data_file):
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
	# drop attribute fnlwgt
	return df.drop(columns=['fnlwgt'])


# Return the number of rows in the pandas dataframe df.
def num_rows(df):
	return df.shape[0]


# Return a list with the column names in the pandas dataframe df.
def column_names(df):
	return df.columns.values.tolist()


# Return the number of missing values in the pandas dataframe df.
def missing_values(df):
	return sum(df.isna().sum())


# Return a list with the columns names containing at least one missing value in the pandas dataframe df.
def columns_with_missing_values(df):
	columns = df.isna().sum()
	return columns[columns > 0].index.tolist()


# Return the percentage of instances corresponding to persons whose education level is 
# Bachelors or Masters, by rounding to the third decimal digit,
# in the pandas dataframe df containing the data set in the adult.csv file.
# For example, if the percentage is 0.21547%, then the function should return 0.216.
def bachelors_masters_percentage(df):
	edu = df['education'].value_counts()
	percent = 100 * (edu['Bachelors'] + edu['Masters']) / num_rows(df)
	return round(percent, 3)


# Return a pandas dataframe (new copy) obtained from the pandas dataframe df 
# by removing all instances with at least one missing value.
def data_frame_without_missing_values(df):
	return df.dropna()


# Return a pandas dataframe (new copy) from the pandas dataframe df 
# by converting the df categorical attributes to numeric using one-hot encoding.
# The function should not encode the target attribute, and the function's output
# should not contain the target attribute.
def one_hot_encoding(df):
	new_df = copy.copy(df)
	attributes = column_names(new_df)
	attributes.remove('class')
	temp_df = pd.DataFrame()
	for attr in attributes:
		temp_df = pd.concat([temp_df, pd.get_dummies(new_df[attr], prefix=attr)], axis=1)
	return temp_df


# Return a pandas series (new copy), from the pandas dataframe df,
# containing only one column with the labels of the df instances
# converted to numeric using label encoding. 
def label_encoding(df):
	le = LabelEncoder()
	class_series = copy.copy(df['class'])
	le.fit(class_series)
	return pd.Series(le.transform(class_series))


# Given a training set X_train containing the input attribute values 
# and labels y_train for the training instances,
# build a decision tree and use it to predict labels for X_train. 
# Return a pandas series with the predicted values. 
def dt_predict(X_train,y_train):
	# initialize decision tree
	dt = tree.DecisionTreeClassifier(random_state=0)
	# fit the tree model to the training data
	dt.fit(X_train, y_train)
	pred = pd.Series(dt.predict(X_train))
	return pred


# Given a pandas series y_pred with the predicted labels and a pandas series y_true with the true labels,
# compute the error rate of the classifier that produced y_pred.  
def dt_error_rate(y_pred, y_true):
	score = 0
	M = len(y_true)
	for i in range(M):
		if y_pred[i] == y_true[i]:
			score += 1
	error_rate = 1 - score/M
	return error_rate


# MAIN
# print log in the file
f = open(OUTPUT_FILE, 'a')
sys.stdout = f
sys.stderr = f
df = read_csv_1(DATA_FILE)
print("------------------Part 1--------------------")
print("--------original dataframe-----------")
print(df)
print("--------number of rows-----------")
print(num_rows(df))
print("--------column names-----------")
print(column_names(df))
print("--------number of missing values-----------")
print(missing_values(df))
print("--------missing columns-----------")
print(columns_with_missing_values(df))
print("--------percentage-----------")
print(bachelors_masters_percentage(df))
print("--------clean dataframe-----------")
new_df = data_frame_without_missing_values(df)
print(new_df)
print("--------one hot encoding-----------")
attributes = one_hot_encoding(new_df)
print(attributes)
print("--------label encoding-----------")
targets = label_encoding(new_df)
print(targets)
print("--------decision tree predict-----------")
predict = dt_predict(attributes, targets)
print(predict)
print("--------error rate-----------")
print(dt_error_rate(predict, targets))
