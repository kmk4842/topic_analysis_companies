import spacy
from spacy.tokens import Token
from gensim import corpora
from gensim.models.ldamodel import LdaModel
import pyLDAvis.gensim
from pprint import pprint
import os.path
import pickle
import matplotlib.pyplot as plt
from wordcloud import WordCloud  # can generate clouds as images.
import numpy as np
import re
import logging

# Switches for parts of programme
extract = True
process = True
load_texts = True
visualize = True
inference = True

# Make logs to adjust LDA Model:
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def excerpts(regex_string, no_lines, file_name):
    '''
    Read the source document and extract excerpts around a keyword.
    :param regex_string: the search string
    :param no_lines: int lines before/after the search string
    :param file_name: string file to read from
    :return: list of excerpt strings
    '''
    line_stack = []
    output = []
    found = 0
    with open(file_name, encoding="utf-8") as f:
        for line in f:
            line = str(line.strip()).lower()
            # Make sure all the characters are mapped to alphabetical, remove punctuation along the way.
            line_verified = ' '.join([word if word.isalpha() else word[:-1] if word[:-1].isalpha() else word[
                                                                                                        1:] if word[
                                                                                                               1:].isalpha() else ""
                                      for word in line.split()])
            line = line_verified
            line_stack.append(line)
            if (re.search(regex_string, line) is not None) & (found == 0):
                # turn the "found" switch on but only if it's off
                found = 1
            elif found == 1:
                if len(line_stack) < no_lines * 2 + 1:
                    continue
                elif len(line_stack) == no_lines * 2 + 1:
                    # output excerpt by joining lines and clear stack if stack full and keyword found
                    # turn switch off
                    output.append(' '.join(line_stack))
                    line_stack.clear()
                    found = 0
                else:
                    pass
            elif len(line_stack) == no_lines + 1:
                line_stack.pop(0)
            else:
                continue
    return output


### EXTRACTING TEXTS ###

# Regex for multiple synonyms of change and strategy, lower case for strategy to avoid "strategic report". Tested with https://regexr.com/
# Bigrams with strategy could be interesting, Strategy by itself is not, I remove it.

def save_list(text_list, file_name):
    with open(file_name, 'wt', encoding="utf-8") as output:
        for t in text_list:
            output.write(str(t) + '\n')


if extract:
    texts_raw = excerpts("([cC]hang[a-z]*)|([sS]hift[a-z]*)|([tT]ransform[a-z]*)|([tT]ransition[a-z]*)", 3,
                         "./data/merged2010_2016.txt")

    print('Number of excerpts: %d' % len(texts_raw))

    save_list(texts_raw, "texts_raw_out.txt")
else:
    pass


### PRE-PROCESSING ###
# I used to follow https://towardsdatascience.com/building-a-topic-modeling-pipeline-with-spacy-and-gensim-c5dc03ffc619
# BUT they break the pipe to reconstruct from elements each time
# This script implements the normal pipe and adds a custom attribute to token
# Tokens are excluded only at the end, when making a gensim-friendly list of texts.

def get_is_excluded(token):
    is_excluded = False
    if token.is_stop or token.is_punct or token.ent_type_ or token.lemma_ == '-PRON-':
        is_excluded = True
    if re.search(
            "(annual)|(report)|(account[s]*)|(ceo)|(CEO)|(chairman)|([mM]anagement)|([bB]oard)|([dD]irector)|([Gg]roup)|(plc)|(PLC)|([cC]ompany)|([cC]hief)|([eE]xecutive)|([oO]fficer)|(financial)|(statement)",
            token.lemma_):
        is_excluded = True
    if len(token.text) < 4 or token.is_alpha is False:
        is_excluded = True
    return is_excluded


Token.set_extension('is_excluded', getter=get_is_excluded)

# Disable sentence parsing and dependency parsing, the data does not include that.
nlp = spacy.load("en_core_web_sm", disable=["senter", "parser"])


def corpuser(texts_list):
    """
    Pre-process texts using spaCy pipeline defined earlier.
    :param texts_list: a list of strings
    :return: list of spaCy docs.
    """
    output = []
    for t in texts_list:
        output.append(nlp(t))
    return output


def gensimmer(list_of_docs):
    """
    Process spaCy docs into lists of strings, excluding unwanted tokens and short docs.
    :param list_of_docs: a list of spacy docs
    :return: list of string texts.
    """
    output = []
    for doc in list_of_docs:
        new_text = []
        for token in doc:
            if not token._.is_excluded:
                new_text.append(token.lemma_)
            else:
                pass
        if len(new_text) > 10:
            output.append(new_text)
        else:
            continue
    return output


if process:
    corpus_spacy = corpuser(texts_raw)
    # consider saving the spacy corpus for reference later... https://spacy.io/usage/saving-loading
    texts = gensimmer(corpus_spacy)
    save_list(texts, "texts_processed_out.txt")
    with open("text_gensim_pickle", "wb") as f:
        pickle.dump(texts, f)
else:
    pass

if load_texts:
    with open("text_gensim_pickle", "rb") as f:
        texts = pickle.load(f)
else:
    pass

### GENSIM ###
# Tutorial at https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html#sphx-glr-auto-examples-tutorials-run-lda-py

# Create Dictionary
dictionary = corpora.Dictionary(texts)
print('Number of unique tokens in raw dictionary: %d' % len(dictionary))

# Filter out rare and very common words
dictionary.filter_extremes(no_below=100, no_above=0.25)

# Remove gaps
dictionary.compactify()

# Vector corpus
corpus = [dictionary.doc2bow(t) for t in texts]

print('Number of unique tokens in filtered dictionary: %d' % len(dictionary))
print('Number of documents in corpus: %d' % len(corpus))

# Make an index to word dictionary - not sure we need this.
temp = dictionary[0]  # This is only to "load" the dictionary.
id2word = dictionary.id2token

# Set model parameters - chunking may cause trouble but it saves memory,
# Passes help the model converge, but this affect parameters, rather than terms selected.
# I increased the number of passes once I decided on the number of topics.
# It would be good to set the random seed too.
# for x in range(5, 21):
# x = 12
# if x == 12:
for x in (6, 12):
    num_topics = x
    chunksize = 40000
    passes = 20
    iterations = 400
    eval_every = None  # Don't evaluate model perplexity, takes too much time, unless you want to determine the right number of passes/iterations - check for convergence in the log.

    # Train LDA Model
    # Check in the log if documents converge by the final pass.
    # You may want to set random_state fore replicability using https://www.random.org/
    lda_model = LdaModel(
        corpus=corpus,
        id2word=id2word,
        chunksize=chunksize,
        alpha='auto',
        eta='auto',
        iterations=iterations,
        num_topics=num_topics,
        passes=passes,
        eval_every=eval_every
    )

    # Pickle the model for later use
    pickle.dump(lda_model, open(os.path.join('./results/lda_save_' + str(num_topics)+'.pk'), 'wb'))

    print('The top 10 keywords in each topic')
    pprint(lda_model.print_topics(num_words=10))

    # Topic coherence https://rare-technologies.com/what-is-topic-coherence/
    top_topics = lda_model.top_topics(corpus)  # , num_words=20)
    avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
    print('Average topic coherence: %.4f.' % avg_topic_coherence)
    print('Top topics and their coherence:')
    pprint(top_topics)

    # Comparing LDA models
    # https://radimrehurek.com/gensim/auto_examples/howtos/run_compare_lda.html

    # LDA Results Visual Analysis
    if visualize:
        #    pyLDAvis.enable_notebook()
        lda_res_path = os.path.join('./results/lda_pyldavis_' + str(num_topics))
        prepped_results = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
        with open(lda_res_path, 'wb') as f:
            pickle.dump(prepped_results, f)
        with open(lda_res_path, 'rb') as f:
            prepped_results = pickle.load(f)
        pyLDAvis.save_html(prepped_results, './results/ldavis_' + str(num_topics) + '.html')

        # Create word cloud in grey scale at 300 dpi for publication

        def grey_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            return "hsl(0, 0%%, %d%%)" % np.random.randint(60, 95)


        for t in range(lda_model.num_topics):
            plt.figure()
            wc = WordCloud(background_color='black', color_func=grey_color_func).fit_words(
                dict(lda_model.show_topic(t, 200)))
            plt.imshow(wc)
            plt.axis("off")
            plt.title("Topic #" + str(t))
            # plt.show()
            plt.savefig(os.path.join('./results/wordcloud_' + str(num_topics) + '_' + str(t) + '.png'), format='png',
                        dpi=300)
    else:
        pass

if inference:
    # Infer topic distribution in the data to find examples
    # lda_model = last model from above
    # unpickle another model that you prefer
    # lda_model = pickle.load(open(os.path.join('./results/lda_save_' + str(num_topics)+'.pk'), 'rb'))
    topic_vector = []
    for d in range(1000):
        topic_vector.append(lda_model[corpus[d]])
    save_list(topic_vector, './results/lda_trained_12_vectors.txt')
    # it would be best to print out the excerpt alongside - that can be done with topn
    # that would make best sense if reaching to original corpus - but no direct index to corpus_spacy
else:
    pass
