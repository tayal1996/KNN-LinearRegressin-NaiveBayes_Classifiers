import numpy as np
import pandas as pd

def categorical_prob(i, r, df):
	l = len(df)
	df = df[df[i] == r]
	return len(df)/l

def normpdf(x, mean, sd):
    var = float(sd)**2
    denom = (2*np.pi*var)**.5
    num = np.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom

def predict(row, df_train_zero, df_train_one, mean_zero, sd_zero, mean_one, sd_one):
	prob_yes = 1
	prob_no = 1
	for i in range(1,len(row)):
		if i <= 5:	#numerical
			prob_yes *= normpdf(row[i],mean_one[i],sd_one[i])
			prob_no *= normpdf(row[i],mean_zero[i],sd_zero[i])
		else:		#categorical
			temp_yes = categorical_prob(i,row[i],df_train_one)
			temp_no = categorical_prob(i,row[i],df_train_zero)
			if temp_no == temp_yes == 0:
				continue
			prob_yes *= temp_yes
			prob_no *= temp_no

	if prob_yes > prob_no:
		return 1
	else:
		return 0

def calculate_mean_and_sd(df_train):
	return np.mean(df_train),np.std(df_train)

def naive_bayes_calculator(df_train, df_test):
	df_train_one = df_train[df_train[0] == 1]
	df_train_zero = df_train[df_train[0] == 0]
	mean_zero,sd_zero = calculate_mean_and_sd(df_train_zero.iloc[:,1:6])
	mean_one,sd_one = calculate_mean_and_sd(df_train_one.iloc[:,1:6])

	correct, wrong = 0,0
	for index,row in df_test.iterrows():
		prediction = predict(row, df_train_zero, df_train_one, mean_zero, sd_zero, mean_one, sd_one)
		if prediction == row[0]:
			correct += 1
		else:
			wrong += 1
	print("correct = {} wrong = {}".format(correct,wrong))
	accuracy = 0
	accuracy = correct/(wrong+correct)
	print("Accuracy : %0.2f" %(accuracy*100),"%")

def main():
	data_frame = pd.read_csv('LoanDataset/data.csv', header=None)
	data_frame = data_frame.drop(data_frame.columns[[0,]], axis=1)
	data_frame = data_frame.drop(data_frame.index[0])
	cols = data_frame.columns.tolist()
	cols = [cols[8]] + cols[0:3] + [cols[5]] + [cols[7]] + cols[3:5] + [cols[6]] + cols[9:13]
	data_frame = data_frame[cols]
	data_frame.columns = range(data_frame.shape[1])
	# print(data_frame.head(20))
	
	df_test = data_frame.sample(frac = 0.2)
	df_train = data_frame.drop(df_test.index)

	naive_bayes_calculator(df_train, df_test)
	# print(list(mean_zero))

if __name__ == '__main__':
	main()