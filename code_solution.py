import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re, nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings('ignore')
import os

df1=pd.read_csv("Input.csv")
data_new=df1
data_new['para']=''

#scraping the data from websites

for i in range (0,len(df1['URL'])):
    
    base_url=data_new['URL'][i]
    url = base_url
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    div_tag = soup.find('div', {'class': 'td-post-content tagdiv-type'})
    if div_tag==None:
          div_tag = soup.find('div', {'class': 'tdb-block-inner td-fix-index'})
    whole_paragraph = ""
    for child in div_tag.children:
        if child.name is not None:
            whole_paragraph += child.get_text(separator=' ', strip=True) + ' '
    data_new['para'][i]=whole_paragraph

data_new.to_csv('data_withPara.csv', index=False)

#Text analysis and storing variables in csv

neg=open("negative-words.txt",'r')
pos=open("positive-words.txt",'r')
n_words=neg.read()
p_words=pos.read()
neg_words=nltk.word_tokenize(n_words)
pos_words=nltk.word_tokenize(p_words)
stopwords1=[]
folder_path = 'stop_word'
files = os.listdir(folder_path)
for file_name in files:
    if os.path.isfile(os.path.join(folder_path, file_name)):
        with open(os.path.join(folder_path, file_name), 'r') as file:
            file_contents = file.read()
            stop_words=nltk.word_tokenize(file_contents)
            stopwords1.append(stop_words)
stop_words_array=np.array(stopwords1, dtype="object")
stopwords_list=np.reshape(stop_words_array,-1)
df=pd.read_csv('data_withPara.csv')

# Defining Variables

def countpositive(s):
    paragraph=nltk.word_tokenize(s)
    count=0
    for word in paragraph:
        if word in pos_words:
            count+=1
    return count

def countnegative(s):
    paragraph=nltk.word_tokenize(s)
    count=0
    for word in paragraph:
        if word in neg_words:
            count+=1
    return count

def cleanwords(s):
    paragraph=nltk.word_tokenize(s)
    count=0
    for word in paragraph:
        if word not in stopwords_list:
            count+=1
    return count

def clean_tokenized_sentence(s):
    words=nltk.word_tokenize(s)
    count=0
    for word in words:
        word=re.sub(r'[^\w\s]','',word)
        if word !='' and word not in stopwords.words('english'):
          count+=1
    return count

def count_complex(s):
    words=nltk.word_tokenize(s)
    count=0
    for word in words:
      vowels = {'a', 'e', 'i', 'o', 'u'}
      count_v = 0
      for char in word:
        if char.lower() in vowels:
            count_v += 1
        if count_v>1:
           count+=1
    return count

def personal_pronouns(s):
    words1=nltk.word_tokenize(s)
    count=0
    for word in words1:
        word=re.sub(r'[^\w\s]','',word)
        if word !='' and word not in stopwords.words('english') and word in ['I', 'we', 'my', 'ours', 'us']:
          count+=1
    return count

df['Positive_score']=None
df['Negative_score']=None
df['subjectivity_score']=None
df['polarity_score']=None
df['complex_percentage']=None
df['word_count']=None
df['personal_pronouns']=None

# Adding variables to dataframe

for i in range(0,len(df['URL_ID'])):
    pos_score=countpositive(df['para'][i])
    neg_score=countnegative(df['para'][i])
    clean_s=cleanwords(df['para'][i])
    pol_score=(pos_score-neg_score)/(pos_score+neg_score+0.000001)
    sub_score=(pos_score+neg_score)/(clean_s+0.000001)
    df['Positive_score'][i]=pos_score
    df['Negative_score'][i]=neg_score
    df['subjectivity_score'][i]=sub_score
    df['polarity_score'][i]=pol_score

for i in range(0,len(df['URL_ID'])):
    complexs=count_complex(df['para'][i])
    pronouns=personal_pronouns(df['para'][i])
    cleans=clean_tokenized_sentence(df['para'][i])
    if cleans!=0:
        complex_percentage=(complexs)/(cleans)
    df['complex_percentage'][i]=complex_percentage
    df['personal_pronouns'][i]=pronouns
    df['word_count'][i]=cleans

#saving output file
df.to_csv('Output.csv', index=False)