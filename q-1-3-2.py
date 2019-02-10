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
    # print(data_frame.head(20))
    x_df_train,y_df_train = x_and_y_finder(df_train)
    beta = calculate_beta(x_df_train,y_df_train)
    x_df_test,y_df_test = x_and_y_finder(df_test)
    y_predict = np.dot(x_df_test,beta)
    
    y1 = y_predict.tolist()
    y2 = y_df_test['Chance of Admit '].values.tolist()
    # print(len(y2),len(y1))
    # print(y_df_test)
    sum = 0
    for i in range(len(y2)):
        sum += (y1[i][0]-y2[i])**2
    MSE = sum/len(y2)
    print("MSE: %f" %MSE)

    sum = 0
    for i in range(len(y2)):
        sum += np.abs(y1[i][0]-y2[i])
    MAE = sum/len(y2)
    print("MAE: %f" %MAE)


    sum = 0
    for i in range(len(y2)):
        sum += (y1[i][0]-y2[i])
    MPE = np.abs(sum*100/len(y2))
    print("MPE: %f" %MPE)
    
if __name__ == '__main__':
	main()




