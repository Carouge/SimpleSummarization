from gensim.summarization.summarizer import summarize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge


with open('./document.txt', "r") as f:
    content = f.read()
summarized = summarize(content, word_count=5)
words = summarized.split(' ')

tokenizer = RegexpTokenizer(r'\w+')
filtered_words = [word for word in words if word not in stopwords.words('english')]
filtered_words = tokenizer.tokenize(' '.join(filtered_words))

result = ' '.join(filtered_words)
reference = [['abstractive','scientific', 'text', 'summarization', 'using', 'GAN']]
candidate = filtered_words
print(filtered_words)
score = sentence_bleu(reference, candidate)

#ROUGE-1 refers to the overlap of 1-gram (each word) between the system and reference summaries.
#ROUGE-2 refers to the overlap of bigrams between the system and reference summaries.

rouge = Rouge()
scores = rouge.get_scores(' '.join(candidate), ' '.join(reference[0]), avg=True)
print("BLEU score: ", score)
print( "Rouge scores: ", scores)
