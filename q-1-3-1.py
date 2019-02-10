import numpy as np
import pandas as pd

def calculate_beta(x, y):
	transpose_x = x.T
	inverse_x =  np.linalg.inv(np.dot(transpose_x,x))
	beta = np.dot(np.dot(inverse_x,transpose_x),y)
	return beta

def x_and_y_finder(df):
	y = df[['Chance of Admit ']]
	df[['Chance of Admit ']] = 1
	df.columns = df.columns.str.replace('Chance of Admit ','Constant')
	return df,y

def main():
	data_frame = pd.read_csv('AdmissionDataset/data.csv')
	data_frame = data_frame.drop(['Serial No.'], axis=1)
	df_test = data_frame.sample(frac = 0.2)
	df_train = data_frame.drop(df_test.index)
	print(data_frame.head(20))
	x_df_train,y_df_train = x_and_y_finder(df_train)
	beta = calculate_beta(x_df_train,y_df_train)
	x_df_test,y_df_test = x_and_y_finder(df_test)
	y_predict = np.dot(x_df_test,beta)
	print(y_predict)
	# print(beta)
	exit(0)
	print(y.head(20))
	
	data_frame = data_frame.drop(['Chance of Admit '], axis=1)
	print(data_frame.head(20))
	exit(0)
	data_frame = data_frame.drop(data_frame.index[0])
	cols = data_frame.columns.tolist()
	cols = [cols[8]] + cols[0:3] + [cols[5]] + [cols[7]] + cols[3:5] + [cols[6]] + cols[9:13]
	data_frame = data_frame[cols]
	data_frame.columns = range(data_frame.shape[1])
	# print(data_frame.head(20))
	
	

	naive_bayes_calculator(df_train, df_test)
	# print(list(mean_zero))

if __name__ == '__main__':
	main()