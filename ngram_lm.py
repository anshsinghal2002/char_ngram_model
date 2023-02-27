import math, random

# PLEASE do not delete or modify the comments that divide the code
# into sections, like the following comment.

################################################################################
# Utility Functions
################################################################################

COUNTRY_CODES = ['af', 'cn', 'de', 'fi', 'fr', 'in', 'ir', 'pk', 'za']

def start_pad(c):
    ''' Returns a padding string of length c to append to the front of text
        as a pre-processing step to building n-grams. c = n-1 '''
    return '~' * c

def ngrams(c, text):
    ''' Returns the ngrams of the text as tuples where the first element is
        the length-c context and the second is the character '''
    text = start_pad(c)+text
    return [(text[i:i+c],text[i+c]) for i in range(0,len(text)-c)]


def create_ngram_model(model_class, path, c=2, k=0):
    ''' Creates and returns a new n-gram model trained on the entire text
        found in the path file '''
    model = model_class(c, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        model.update(f.read())
    return model

def create_ngram_model_lines(model_class, path, c=2, k=0):
    '''Creates and returns a new n-gram model trained line by line on the
        text found in the path file. '''
    model = model_class(c, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f.readlines():
            model.update(line.strip())
    return model

################################################################################
# Basic N-Gram Model
################################################################################

class NgramModel(object):
    ''' A basic n-gram model using add-k smoothing '''

    def __init__(self, c, k):
        self.c = c
        self.k = k
        self.vocab = set()
        self.ngrams = {}

    def get_vocab(self):
        return self.vocab

    def update_frequencies(self, ngram, char):
        if ngram in self.ngrams.keys():
            self.ngrams[ngram].update(char)
        else:
            self.ngrams.update({ngram:NgramKey(char)})

    def update(self, text):
        '''self.update_frequencies([text[i:i+self.c] for i in range(0,len(text)-self.c)])'''
        for char in text:
            self.vocab.add(char)
        for ngram,char in ngrams(self.c,text):
            self.update_frequencies(ngram,char)

    def prob(self, context, char):
        if context not in self.ngrams.keys():
            return 1/len(self.get_vocab())
        return (self.ngrams[context].get_char_frequency(char) + self.k)/(self.ngrams[context].get_frequency() + self.k*len(self.get_vocab()))

    def random_char(self, context):
        ''' Returns a random character based on the given context and the 
            n-grams learned by this model '''
        r = random.random()
        prob_sum = 0
        for char in self.get_vocab():
            p_i = self.prob(context,char)
            if prob_sum+p_i>r:
                return char
            prob_sum+=p_i
        return None

        

    def random_text(self, length):
        if length==0:
            return ""
        to_return = start_pad(self.c)
    
        for i in range(length):
            to_return+=self.random_char(to_return[i:i+self.c])

        return to_return[self.c:]

    def perplexity(self, text):
        ''' Returns the perplexity of text based on the n-grams learned by
            this model '''

        # if num_ch==0:
        #     return float('inf')
        # text+=start_pad(self.c)
        # log_sum = 0
        # for i in range(num_ch):
        #     char_prob = self.prob(text[i:i+self.c],text[i+self.c])
        #     if char_prob==0:
        #         return float('inf')
        #     log_sum+=math.log(char_prob,2)
        # entropy = (-1/num_ch)*log_sum
        # return 2**(entropy)

        n_grams = ngrams(self.c,text)
        log_sum = 0

        for context,char in n_grams:
            char_prob = self.prob(context,char)
            if char_prob==0:
                return float('inf')
            log_sum+=math.log(char_prob,2)

        entropy = (-1/len(text))*(log_sum)
        return 2**(entropy)
        

class NgramKey():

    def __init__(self, char):
        self.frequency = 1
        self.chars = {char:1}
    def update(self, char):
        if char in self.chars.keys():
            self.chars[char]+=1
        else:
            self.chars.update({char:1})
        self.frequency+=1
    def get_frequency(self):
        return self.frequency
    def get_char_frequency(self,char):
        if char in self.chars.keys():
            return self.chars[char]
        else:
            return 0


################################################################################
# N-Gram Model with Interpolation
################################################################################

class NgramModelWithInterpolation(NgramModel):
    ''' An n-gram model with interpolation '''

    def __init__(self, c, k):
        super().__init__(c,k)
        self.lower_ngrams = [NgramModel(i,k) for i in range(0,c)]
        self.weights = [1/(c+1) for _ in range(c+1)]
        # TO RANDOMIZE WEIGHTS:
        # self.weights_unscaled = [random.randint(1,10) for _ in range(c+1)]
        # sum_weights = sum(self.weights_unscaled)
        # self.weights = [i/sum_weights for i in self.weights_unscaled]

    def get_vocab(self):
        return super().get_vocab()

    def update(self, text):
        for lower_ngram in self.lower_ngrams:
            lower_ngram.update(text)
        super().update(text)


    def prob(self, context, char):
        prob_sum=(super().prob(context,char))*self.weights[-1]
        for i in range(len(self.lower_ngrams)):
            lower_context,_ = ngrams(self.lower_ngrams[i].c,context+char)[-1]
            prob_sum+= self.weights[i]*self.lower_ngrams[i].prob(lower_context,char)
        return prob_sum


class Language_Model_Tester():
    def __init__(self, c,k,interpolation=True):
        # self.train_paths = ['cities_train'+country+'.txt' for country in COUNTRY_CODES]
        self.interpolation = interpolation
        self.c = c
        self.k = k

    def train_models(self):
        if self.interpolation:
            self.models = {country:create_ngram_model(NgramModelWithInterpolation,'cities_train/'+country+'.txt',self.c,self.k) for country in COUNTRY_CODES}
        else:    
            self.models = {country:create_ngram_model(NgramModel,'cities_train/'+country+'.txt',self.c,self.k) for country in COUNTRY_CODES}
    
    def classify(self):
        self.city_val = {}
        for country in COUNTRY_CODES:
            countries = {i:0 for i in COUNTRY_CODES}
            path = 'cities_val/'+country+'.txt'
            with open(path, encoding='utf-8', errors='ignore') as f:
                for city in f.readlines():
                    city = city.strip()
                    min_perplexity = float('inf')
                    for country_test in COUNTRY_CODES:
                        p = self.models[country_test].perplexity(city)
                        if p<=min_perplexity:
                            min_perplexity = p
                            country_chosen = country_test
                    countries[country_chosen]+=1
            self.city_val[country]=countries
        print ('Classification Complete')

    def evaluate_models(self):
        precision_sum=0
        ctr=0
        for country in COUNTRY_CODES:
            print (self.evaluate_model(country))
            precision_sum+=self.get_precision(country)
            ctr+=1
        print ('Avg Precision: ',precision_sum/ctr)

    def evaluate_model(self,country_code):
        return {'country':country_code,'precision':self.get_precision(country_code),'recall':self.get_recall(country_code)}

    def get_precision(self,country_code):
        return self.city_val[country_code][country_code]/sum([self.city_val[other_country][country_code] for other_country in COUNTRY_CODES])

    def get_recall(self,country_code):
        return self.city_val[country_code][country_code]/sum([self.city_val[country_code][other_country] for other_country in COUNTRY_CODES])

################################################################################
# Your N-Gram Model Experimentations
################################################################################

# Add all code you need for testing your language model as you are
# developing it as well as your code for running your experiments
# here.
#
# Hint: it may be useful to encapsulate it into multiple functions so
# that you can easily run any test or experiment at any time.

def test_ngram_init():
    m = NgramModel(1,0)
    m.update('abab')
    print(m.get_vocab())
    m.update('abcd')
    print(m.get_vocab())
    print(m.prob('a','b'))
    print(m.prob('~','c'))
    print(m.prob('b','c'))

def test_random_text():
    m = NgramModel(1,0)
    m.update('abab')
    m.update('abcd')
    random.seed(1)
    print ( m.random_text(25))

def test_random_text_shakespeare():
    print ('\nbigram\n')

    m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 2)
    print (m.random_text(250))
    print ('\ntrigram\n')

    m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 3)
    print (m.random_text(250))
    print ('\nc=4\n')

    m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 4)
    print (m.random_text(250))
    print ('\nc=7\n')

    m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 7)
    print (m.random_text(250))
    print ('\nWITH LINES\n')
    m = create_ngram_model_lines(NgramModel, 'shakespeare_input.txt', 7)
    print (m.random_text(250))

# test_random_text_shakespeare()

# m = NgramModel(1,0)
# m.update('abab')
# m.update('abcd')
# print (m.perplexity('abcd'))
# print (m.perplexity('abca'))
# print (m.perplexity('abcda'))

# m = NgramModel(1,1)
# m.update('abab')
# m.update('abcd')
# print (m.prob('a','a'))
# print (m.prob('a','b'))
# print (m.prob('c','d'))
# print (m.prob('d','a'))

def test_similar_unsimilar_text():
    m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 7,2)
    with open('shakespeare_input.txt', encoding='utf-8', errors='ignore') as f:    
        print ('Shakespeare Training Set: ',m.perplexity(f.read()))
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:    
        print ('Shakespeare Sonnet Set: ',m.perplexity(f.read()))
    with open('nytimes_article.txt', encoding='utf-8', errors='ignore') as f:    
        print ('NYC Article: ',m.perplexity(f.read()))

# test_similar_unsimilar_text()

def test_ngram_interpolation():
    # m = NgramModelWithInterpolation(1, 0)
    # m.update('abab')
    # print (m.prob('a','a'))
    # print (m.prob('a','b'))

    m = NgramModelWithInterpolation(2,1)
    m.update('abab')
    m.update('abcd')
    print("m.prob('~a','b'): ",m.prob('~a','b'))
    print("m.prob('ba','b'): ",m.prob('ba','b'))
    print("m.prob('ba','b'): ", m.prob('ba','b'))
    print("m.prob('bc','d'): ",m.prob('bc','d'))

# test_ngram_interpolation()

def test_different_weights():
    m = NgramModelWithInterpolation(2,1)
    print (m.weights)
    m.update('abab')
    m.update('abcd')
    print("m.prob('~a','b'): ",m.prob('~a','b'))
    print("m.prob('ba','b'): ",m.prob('ba','b'))
    print("m.prob('ba','b'): ", m.prob('ba','b'))
    print("m.prob('bc','d'): ",m.prob('bc','d'))

# test_different_weights()

def test_country_classification():

    # m=create_ngram_model(NgramModel,'cities_train/de.txt',7,2)
    # print (m.random_text(10))

    T = Language_Model_Tester(6,1)
    T.train_models()
    # print (T.models['af'].get_vocab())
    T.classify()
    print ('c = ',T.c)
    print ('k = ',T.k)
    print ('Interpolation=',T.interpolation)
    print (T.evaluate_models())

test_country_classification()