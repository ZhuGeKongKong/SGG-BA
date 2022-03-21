import json
import numpy as np
import spacy
import logging
nlp = spacy.load('en_core_web_sm')
from gensim.corpora import WikiCorpus
import os
import sys
import re
from spacy.matcher import PhraseMatcher
import tqdm
nlp.max_length = 169647309+50
# scrapped_data = urllib.request.urlopen('https://en.wikipedia.org/wiki/Artificial_intelligence')  
# article = scrapped_data.read()

# parsed_article = bs.BeautifulSoup(article,'lxml')

# paragraphs = parsed_article.find_all('p')
if __name__ == "__main__":
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    VG_dict = json.load(open('/mnt/hdd2/guoyuyu/datasets/visual_genome/data/genome/VG-SGG-dicts.json','r'))
    VG_pred2idx = VG_dict['predicate_to_idx']
    pred_wiki_count = {}
    phrase_matcher = PhraseMatcher(nlp.vocab)

    phrases = []
    patterns = []
    for pred_i in VG_pred2idx:
        patterns.append(nlp(pred_i))
    print(patterns)
    print(len(patterns))
    phrase_matcher.add('phrase_matcher', None, *patterns)
    # inp = "/mnt/hdd3/guoyuyu/datasets/wikipedia/simplewiki-latest-pages-articles.xml.bz2"
    # wiki = WikiCorpus(inp, lemmatize=False, dictionary={})
    # article_text = " "

    # for text in wiki.get_texts():
        # article_text.join(text) + "\n"
        # if (i % 10000 == 0):
            # logger.info("Append " + str(i) + " articles")
    wiki_file = open("/mnt/hdd3/guoyuyu/datasets/wikipedia/simplewiki-latest-pages-articles.txt", 'r')
    article_text = wiki_file.readline()
    article_count = 0
    pred_wiki_count_article = {}
    while article_text:
        processed_article = article_text.lower()  
        processed_article = re.sub('[^a-zA-Z]', ' ', processed_article )  
        processed_article = re.sub(r'\s+', ' ', processed_article)
        sentence = nlp (processed_article)
        matched_phrases = phrase_matcher(sentence)
        pred_wiki_flag = {}
        for match_id, start, end in matched_phrases:
            string_id = nlp.vocab.strings[match_id]  
            span = sentence[start:end] 
            if span.text not in pred_wiki_count:
                pred_wiki_count[span.text] = 1
            else:
                pred_wiki_count[span.text] = pred_wiki_count[span.text] + 1
            pred_wiki_flag[span.text] = 1
            if (article_count % 100 == 0):
                logger.info("Matching " + str(match_id) + " matcher")
                print("article_count: ",article_count, "string_id", string_id, 
                "start: ", start, "end: ", end, "span.text: ", span.text, "sentence[start-2:end+2]: ",sentence[start-1:end+1])
                print("predicate count: ", pred_wiki_count)
        for flag_key in pred_wiki_flag.keys():
            if flag_key not in pred_wiki_count_article:
                pred_wiki_count_article[flag_key] = 1
            else:
                pred_wiki_count_article[flag_key] = pred_wiki_count_article[flag_key] + 1
        article_text = wiki_file.readline() 
        article_count = article_count + 1
        # if article_count == 200:
        #     break
        #print(match_id, string_id, start, end, span.text)
    print("----------------------------------------")
    print("article_count: ", article_count)
    print("before remove repetition: ", pred_wiki_count)
    for pred_i in pred_wiki_count:
        for pred_j in pred_wiki_count:
            if pred_i != pred_j and pred_i in pred_j:
                pred_wiki_count[pred_i] = pred_wiki_count[pred_i] - pred_wiki_count[pred_j]
    print("after remove repetition: ", pred_wiki_count)
    file = open('pred_wiki_count.json','w')
    json.dump(pred_wiki_count,file)
    print("before remove repetition pred_wiki_count_article: ", pred_wiki_count_article)
    for pred_i in pred_wiki_count_article:
        for pred_j in pred_wiki_count_article:
            if pred_i != pred_j and pred_i in pred_j:
                pred_wiki_count_article[pred_i] = pred_wiki_count_article[pred_i] - pred_wiki_count_article[pred_j]
    print("after remove repetition pred_wiki_count_article: ", pred_wiki_count_article)
    file2 = open('pred_wiki_count_article.json','w')
    json.dump(pred_wiki_count_article, file2)
    