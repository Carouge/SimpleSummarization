from gensim.summarization.summarizer import summarize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer


with open('./document.txt', "r") as f:
    content = f.read()
summarized = summarize(content, word_count=5)
words = summarized.split(' ')

tokenizer = RegexpTokenizer(r'\w+')
filtered_words = [word for word in words if word not in stopwords.words('english')]
filtered_words = tokenizer.tokenize(' '.join(filtered_words))

result = ' '.join(filtered_words)
print(result)
