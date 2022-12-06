import glob
from tqdm import tqdm
import numpy as np
import nltk
from collections import Counter, defaultdict
from nltk import ngrams
import en_core_web_sm
nlp = en_core_web_sm.load()
from nltk.corpus import stopwords
stopwords = set(stopwords.words())
nltk.download('stopwords')
from tqdm import tqdm
from spacy.matcher import Matcher 

NEED_TERMS = set(['need', 'needs', 'needed', 'needing', 'want', 'wanting', 'wanted', 'requires', 'required', 'require'])
files = glob.glob('sampled_tweets/*[0-9].npy')
files.sort()
noise = set(["https", "@", "…", "Putin", "“", "’", "Ukraine, Russia"])
noun = set(("NN", "NNS"))
UKRAINE = set(["ukraine", "Ukraine", "Ukrainian"])

def get_entities(sent):
  ## chunk 1
  ent1 = ""
  ent2 = ""

  prv_tok_dep = ""    # dependency tag of previous token in the sentence
  prv_tok_text = ""   # previous token in the sentence

  prefix = ""
  modifier = ""
  
  for tok in nlp(sent):
    ## chunk 2
    # if token is a punctuation mark then move on to the next token
    if tok.dep_ != "punct":
      # check: token is a compound word or not
      if tok.dep_ == "compound":
        prefix = tok.text
        # if the previous word was also a 'compound' then add the current word to it
        if prv_tok_dep == "compound":
          prefix = prv_tok_text + " "+ tok.text
      
      # check: token is a modifier or not
      if tok.dep_.endswith("mod") == True:
        modifier = tok.text
        # if the previous word was also a 'compound' then add the current word to it
        if prv_tok_dep == "compound":
          modifier = prv_tok_text + " "+ tok.text
      
      ## chunk 3
      if tok.dep_.find("subj") == True:
        ent1 = modifier +" "+ prefix + " "+ tok.text
        prefix = ""
        modifier = ""
        prv_tok_dep = ""
        prv_tok_text = ""      

      ## chunk 4
      if tok.dep_.find("obj") == True:
        ent2 = modifier +" "+ prefix +" "+ tok.text
        
      ## chunk 5  
      # update variables
      prv_tok_dep = tok.dep_
      prv_tok_text = tok.text


  return [ent1.strip(), ent2.strip()]

def predicates(sent):
    try:
        doc = nlp(sent)
        # Matcher class object 
        matcher = Matcher(nlp.vocab)

        #define the pattern 
        pattern = [{'DEP':'ROOT'}, 
                {'DEP':'prep','OP':"?"},
                {'DEP':'agent','OP':"?"},  
                {'POS':'ADJ','OP':"?"}] 

        matcher.add("matching_1", [pattern]) 

        matches = matcher(doc)
        k = len(matches) - 1

        span = doc[matches[k][1]:matches[k][2]] 

        return(span.text)
    except:
        return

if __name__=="__main__":

    unigram_counter = defaultdict(int)
    bigram_counter = defaultdict(int)
    ent_what_counter = defaultdict(int)
    ent_who_counter = defaultdict(int)
    good_pairs = []

    for file in tqdm(files[:]):
        tweet_text = []
        reaction = []
        tweets = np.load(file, allow_pickle=True).tolist()
        for tweet in tweets:
            if tweet['lang'] == 'en':
                text = tweet['full_text']
                if any(word in text for word in NEED_TERMS):
                    tweet_text.append(text)
                    reaction.append(int(0.1*tweet['retweet_count']) + int(0.05*tweet['favorite_count']))
        
        for i, sentence in tqdm(enumerate(tweet_text), total=len(tweet_text)):
            pos = nltk.pos_tag(nltk.word_tokenize(sentence))
            for word, tag in pos:
                if tag in noun and word not in noise and len(word) > 3:
                    unigram_counter[word] += reaction[i]

            n = 2
            ngram = ngrams(sentence.split(), n)  
            for grams in ngram:
                pos = nltk.pos_tag(grams)
                if pos[0][1] in noun and pos[1][1] in noun:
                    bigram_counter[" ".join([pos[0][0], pos[1][0]])] += reaction[i]

            ent1, ent2 = get_entities(sentence)
            verb = predicates(sentence)
            if ent1 not in noise and ent2 not in noise:
                ent_who_counter[ent1] += reaction[i]
                ent_what_counter[ent2] += reaction[i]
                if any(word in ent1 for word in UKRAINE) and verb in NEED_TERMS:
                  good_pairs.append([ent1, verb, ent2, reaction[i]])

    unigram_counter = dict(sorted(unigram_counter.items(), key=lambda k_v: k_v[1], reverse=True))
    bigram_counter = dict(sorted(bigram_counter.items(), key=lambda k_v: k_v[1], reverse=True))
    ent_who_counter = dict(sorted(ent_who_counter.items(), key=lambda k_v: k_v[1], reverse=True))
    ent_what_counter = dict(sorted(ent_what_counter.items(), key=lambda k_v: k_v[1], reverse=True))

    print("Unigram Counter: ", unigram_counter)
    print()
    print("Bigram Counter: ", bigram_counter)
    print()
    print("ent_what Counter: ", ent_what_counter)
    print()
    print("ent_who Counter: ", ent_who_counter)
    print()
    print("Good pairs: ", good_pairs)

    

    


        
    

