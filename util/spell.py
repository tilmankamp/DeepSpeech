import re
import kenlm
from collections import Counter

# Load language model
MODEL = kenlm.Model('./data/lm/lm.arpa')

def words(text):
    "List of words in text."
    return re.findall(r'\w+', text.lower())

# Calculate word statistics
WORDS = Counter(words(open('./data/spell/words.txt').read()))

def perplexity(sentence, N=sum(WORDS.values())): 
    "Perplexity of `sentence`."
    return MODEL.score(sentence, bos = False, eos = False)

def correction(sentence):
    "Most probable spelling correction for sentence."
    return max(candidate_sentences(sentence), key=perplexity)

def candidate_sentences(sentence):
    "Generate possible spelling corrections for sentence."
    layer = [[]]
    for word in words(sentence):
        layer = [node + [cword] for cword in candidate_words(word) for node in layer]
    return [' '.join(sentence) for sentence in layer]

def candidate_words(word): 
    "Generate possible spelling corrections for word."
    return (known_words([word]) or known_words(edits1(word)) or known_words(edits2(word)) or [word])

def known_words(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word):
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))
    
