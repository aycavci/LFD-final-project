import json
import pandas as pd
import random
import numpy as np


def split_data(df, path_list):
	test_percentage = 0.10

	train_idx = int(np.ceil(len(path_list) - len(path_list) * test_percentage))
	#print(train_idx)
	test_list = path_list[train_idx:]
	val_idx = int(np.ceil(train_idx - len(path_list) * test_percentage))
	#print(val_idx)
	train_list = path_list[:val_idx]

	val_list = path_list[val_idx:train_idx]

	train_df = df.loc[df['path'].isin(train_list)]
	test_df= df.loc[df['path'].isin(test_list)]
	val_df = df.loc[df['path'].isin(val_list)]

	return train_df, val_df, test_df


path_list = list()
body_list = list()
head_list = list()
name_list = list()
date_list = list()

for i in range(1, 25):
	file_name = './COP_filt3_sub/COP' + str(i) + '.filt3.sub.json'
	# JSON file
	f = open (file_name, "r")
	
	# Reading from file
	data = json.loads(f.read())

	for i in data['articles']:
		path = i['path']
		name = i['newspaper']
		body = i['body']
		date = i['date']
		headline = i['headline']
		path_list.append(path)
		body_list.append(body)
		head_list.append(headline)
		name_list.append(name)
		date_list.append(date)

df = pd.DataFrame(list(zip(path_list, date_list, name_list, body_list, head_list)),
			columns =['path', 'date', 'newspaper_name', 'body', 'headline'])

#Shuffling the dataset
random.shuffle(path_list)

df.head()

train_df, val_df, test_df = split_data(df, path_list)

train_df.to_csv('./data/train.csv', index=False)

val_df.to_csv('./data/val.csv', index=False)

test_df.to_csv('./data/test.csv', index=False)


