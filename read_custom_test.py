import json
import pandas as pd
import random
import numpy as np


path_list = list()
body_list = list()
head_list = list()
name_list = list()
date_list = list()


file_name = './COP_filt3_sub/COP24.filt3.sub.json'
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

df.to_csv('./data/custom_test.csv', index=False)


