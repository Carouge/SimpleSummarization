from gensim.summarization.summarizer import summarize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import pandas as pd


def summarize_doc(content, abstract_ratio):
    summarized = summarize(content, ratio=abstract_ratio)
    words = summarized.split(' ')
    tokenizer = RegexpTokenizer(r'\w+')
    filtered_words = [word for word in words if word not in stopwords.words('english')]
    filtered_words = tokenizer.tokenize(' '.join(filtered_words))
    result = ' '.join(filtered_words)
    return result, filtered_words

dataset = pd.read_csv('./dataset.csv')
testrank_res = pd.DataFrame(dataset['id'])
testrank_res['BLUE'], testrank_res['ROUGE2_f'], testrank_res['ROUGE1_f'], testrank_res['ROUGE1_p'], testrank_res['ROUGE2_p'] = None, None, None, None, None

for index, paper in dataset.iterrows():
    try:
        content = paper['text']
        ratio = round(len(paper['abstract'])/len(content), 3)
        sum_text, filtered_words = summarize_doc(content, ratio)

        abstract = paper['abstract'].split()
        blue_score = sentence_bleu(abstract, filtered_words)

        rouge = Rouge()
        rouge_score = rouge.get_scores(' '.join(filtered_words), ' '.join(abstract))

        testrank_res['BLUE'].iloc[index] = blue_score
        testrank_res['ROUGE2_f'].iloc[index] = rouge_score[0]['rouge-2']['f']
        testrank_res['ROUGE1_f'].iloc[index] = rouge_score[0]['rouge-1']['f']
        testrank_res['ROUGE2_p'].iloc[index] = rouge_score[0]['rouge-2']['p']
        testrank_res['ROUGE1_p'].iloc[index] = rouge_score[0]['rouge-1']['p']
        print("Iteration: ", index)
    except:
        pass

print(testrank_res.head(5))
testrank_res.to_csv('testrank_scores.csv', index=False)