#------- Importing the libraries/packages

import bs4 as bs
import urllib.request
import re    #regular expression
import nltk
import heapq

nltk.download('stopwords')
nltk.download('punkt')


#--------Extract the data

page = urllib.request.urlopen("").read()
soup = bs.BeautifulSoup(page,'lxml')
#print(soup)                         #print the page compressed
#print(soup.prettify)                #print the page readable form

text = ""
for paragraph in soup.find_all('p'):
    text += paragraph.text
#print(text)


#  ------Step 3: Data Cleaning

text = re.sub(r'\[[0-9]*\]',' ',text)  #remove all the references in the text which is denoted by [1],[2] etc.
text = re.sub(r'\s+',' ',text)          #removed all the extra spaces with single space
clean_text = text.lower()               # converted into lower case
clean_text = re.sub(r'\W',' ',clean_text)  #4,5,6: I removed all the extra punctuation, digits, extra spaces etc.
clean_text = re.sub(r'\d',' ',clean_text)
clean_text = re.sub(r'\s+',' ',clean_text)
sentences = nltk.sent_tokenize(text)     #break the all big text into sentences using sent_tokenize()
stop_words = nltk.corpus.stopwords.words('english')
print(sentences)

# stop_words


#-----Step 4: Build the histogram
word2count = {}  #line 1
for word in nltk.word_tokenize(clean_text):     #line 2
    if word not in stop_words:                  #line 3
        if word not in word2count.keys():
            word2count[word]=1
        else:
            word2count[word]+=1
for key in word2count.keys():                   #line 4
    word2count[key]=word2count[key]/max(word2count.values())

'''
Line 1: create an empty dictionary

Line 2: use for loop and using word_tokenize break down the clean _text into words and put into word

Line 3: check if condition if word not present in stop_words

then again check if condition and check word not in word2count.keys()

(if this condition satisfy then) put word2count[word]=1,

else word2count[word]+=1

Line 4: calculated the weighted histogram (see the output you can see the weights not the counts for example — ‘artificial’:0.3620689 etc.)

'''


#----- Calculating the Sentence score

"""
Line 1: create an empty dictionary

Line 2: use for loop and put sentences into sentence (step 3 where we created the sentences)

Line 3: converted into lower case and tokenize tinto word and put into word

Line 4: use if condition and checks whether word in word2count.keys() or not

Line 5: I’ve specified the length here less than 30, you can choose depending upon your needs

Line 6: Again use if-else condition and put sent2score[sentence]=word2count[word] if sentence not present in sentence2keys()

else, sent2score[sentence]+=word2count[word]
"""

sent2score = {}
for sentence in sentences:
    for word in nltk.word_tokenize(sentence.lower()):
        if word in word2count.keys():
            if len(sentence.split(' '))<30:
                if sentence not in sent2score.keys():
                     sent2score[sentence]=word2count[word]
                else:
                    sent2score[sentence]+=word2count[word]

#------Step 6: Find out the best sentences

# I’ve used heapq to find seven best sentences from the Wikipedia (ANN)article.

best_sentences = heapq.nlargest(7,sent2score,key=sent2score.get)
for sentences in best_sentences:
    print(sentences,'\n')