# -*- coding: utf-8 -*-
from conllu_parser import parse
from nltk.util import ngrams
import pprint as pp
import re
import sklearn_crfsuite
from sklearn_crfsuite import metrics
import _pickle as pickle
from collections import OrderedDict

class FeatureExtr():
    '''
    Класс для извлечения признаков слов.
    '''
    def __init__(self, features = []):
        self.features = features    

    def download(self, filename):
        '''
        Загрузка файла в формате conllu и его парсинг.
        '''
        with open(filename, 'r', encoding='utf-8') as f:
            data = f.read()
        result = parse(data)
        return result
    
    def download_list(self, filename):
        '''
        Загрузка списков из файлов (список грамматических категорий, конечные списки некоторых частей речи).
        '''
        with open(filename) as c:
            data = c.read()
            alist = data.split('\n')
        return alist
    
    def all_words_features(self, word):
        '''
        Получение признаков, которые нужно извлечь для любого слова, вне зависимости от его положения в окне.
        Это такие признаки, как:
            1) токен в uppercase (или нет);
            2) токен с большой буквы (или нет);
            3) токен - это цифра (или нет);
            4) первая и последняя буквы;
            5) если длина слова > 1, то префиксы и суффиксы длины от 2 до 4 символов.
        '''
        word_features = [word.isupper(), word.istitle(), word.isdigit(), word[0], word[-1]]     # признаки 1-4
        if len(word) > 1:                                     # префиксы и суффиксы в зависимости от длины слова
            word_features.extend([word[:2],word[-2:]])
        if len(word) > 2:
            word_features.extend([word[:3],word[-3:]])
        if len(word) > 3:
            word_features.extend([word[:4], word[-4:]])
        return word_features

    def make_right_context_features(self,sent,i):
        '''
        Функция, формирующая признаки правого контекста текущего токена. 
        Это те признаки, которые извлекаются с помощью функции all_words_features.
        Функция также формирует список слов в правом контексте. 
        В дальнейшем это требуется для формирования списка всех слов окна, который передаётся в функцию ngrams.
        '''
        word1 = sent[i+1]['form']
        word1_feat = self.all_words_features(word1)
        r_context = [word1]
        r_context_features = dict(zip(['+1:word_is_upper', '+1:word_is_title', '+1:word_is_digit',
                                       '+1:pref[0]','+1:suf[-1]',
                                        '+1:pref[:2]','+1:suf[-2:]','+1:pref[:3]','+1:suf[-3:]',
                                        '+1:pref[:4]','+1:suf[-4:]'], word1_feat))
        if i == len(sent)-3:
            word2 = sent[i+2]['form']
            word2_feat = (self.all_words_features(word2))            
            r_context.append(word2)
            w2_features = dict(zip(['+2:word_is_upper', '+2:word_is_title', '+2:word_is_digit',
                                    '+2:pref[0]','+2:suf[-1]',
                                '+2:pref[:2]','+2:suf[-2:]','+2:pref[:3]','+2:suf[-3:]',
                                '+2:pref[:4]','+2:suf[-4:]'], word2_feat))
            r_context_features.update(w2_features)
        if i < len(sent)-3:
            word2 = sent[i+2]['form']
            word2_feat = (self.all_words_features(word2))            
            word3 = sent[i+3]['form']
            word3_feat = (self.all_words_features(word3))            
            r_context.extend([word2,word3])
            w2_features = dict(zip(['+2:word_is_upper', '+2:word_is_title', '+2:word_is_digit',
                                    '+2:pref[0]','+2:suf[-1]',
                                '+2:pref[:2]','+2:suf[-2:]','+2:pref[:3]','+2:suf[-3:]',
                                '+2:pref[:4]','+2:suf[-4:]'], word2_feat))
            w3_features = dict(zip(['+3:word_is_upper', '+3:word_is_title', '+3:word_is_digit',
                                    '+3:pref[0]','+3:suf[-1]',
                                '+3:pref[:2]','+3:suf[-2:]','+3:pref[:3]','+3:suf[-3:]',
                                '+3:pref[:4]','+3:suf[-4:]'], word3_feat))
            r_context_features.update(w2_features)
            r_context_features.update(w3_features)
        return r_context_features, r_context
        
    def make_left_context_features(self,sent,i):
        '''
        То же, что make_right_context_features, только для левого контекста.
        '''
        word1 = sent[i-1]['form']
        word1_feat = self.all_words_features(word1)
        l_context = [word1]
        l_context_features = dict(zip(['-1:word_is_upper', '-1:word_is_title', '-1:word_is_digit','-1:pref[0]','-1:suf[-1]',
                                '-1:pref[:2]','-1:suf[-2:]','-1:pref[:3]','-1:suf[-3:]','-1:pref[:4]','-1:suf[-4:]'], word1_feat))
        if i == 2:
            word2 = sent[i-2]['form']
            word2_feat = self.all_words_features(word2)           
            l_context.insert(0,word2)
            w2_features = dict(zip(['-2:word_is_upper', '-2:word_is_title', '-2:word_is_digit','-2:pref[0]','-2:suf[-1]',
                                '-2:pref[:2]','-2:suf[-2:]','-2:pref[:3]','-2:suf[-3:]','-2:pref[:4]','-2:suf[-4:]'], word2_feat))
            l_context_features.update(w2_features)
        if i > 2:
            word2 = sent[i-2]['form']        
            word2_feat = self.all_words_features(word2)      
            word3 = sent[i-3]['form']
            word3_feat = self.all_words_features(word3)           
            l_context.insert(0, word2)
            l_context.insert(0, word3)
            w2_features = dict(zip(['-2:word_is_upper', '-2:word_is_title', '-2:word_is_digit','-2:pref[0]','-2:suf[-1]',
                                '-2:pref[:2]','-2:suf[-2:]','-2:pref[:3]','-2:suf[-3:]','-2:pref[:4]','-2:suf[-4:]'], word2_feat))
            w3_features = dict(zip(['-3:word_is_upper', '-3:word_is_title', '-3:word_is_digit','-3:pref[0]','-3:suf[-1]',
                                '-3:pref[:2]','-3:suf[-2:]','-3:pref[:3]','-3:suf[-3:]','-3:pref[:4]','-3:suf[-4:]'], word3_feat))
            l_context_features.update(w2_features)
            l_context_features.update(w3_features)
        return l_context_features, l_context
        
    def ngrams(self, window):
        '''
        Признаки-биграммы и признаки-триграммы.
        '''
        ngrams_2 = list(ngrams(window, 2))
        ngrams_3 = list(ngrams(window, 3))
        bigrams = ['{} {}'.format(ngrams_2[i][0], ngrams_2[i][1]) for i in range(len(ngrams_2))]
        trigrams = ['{} {} {}'.format(ngrams_3[i][0], ngrams_3[i][1], ngrams_3[i][2]) for i in range(len(ngrams_3))]
        bigr_features = dict(zip(['bi_1', 'bi_2', 'bi_3', 'bi_4', 'bi_5', 'bi_6'], bigrams))
        trigr_features = dict(zip(['tri_1', 'tri_2', 'tri_3', 'tri_4', 'tri_5'], trigrams))
        return bigr_features, trigr_features
        
    def word2features(self, sent, i, postags = False):
        '''
        Функция, формирующая полный список признаков:
            1) токен в uppercase (или нет);
            2) токен с большой буквы (или нет);
            3) токен - это цифра (или нет);
            4) первая и последняя буквы;
            5) если длина слова > 1, то префиксы и суффиксы длины от 2 до 4 символов;
            6) всё вышеперечисленное для правого контекста;
            7) всё вышеперечисленное для левого контекста;
            8) сам токен в lowercase;
            9) bias = 1.0;
            10) если токен является началом предложения, 'BOS' = True;
            11) если токен является началом предложения, 'EOS' = True;
            12) биграммы;
            13) триграммы.
        Если значение postags = True, то в качестве признака добавляется postag 
        (для обучения классификаторов, предсказывающих грамматические категории).
        '''
        word = sent[i]['form']
        word_feat = self.all_words_features(word)
        features = dict(zip(['word_is_upper', 'word_is_title', 'word_is_digit','pref[0]','suf[-1]',
                             'pref[:2]','suf[-2:]','pref[:3]','suf[-3:]','pref[:4]','suf[-4:]'], word_feat))
        features.update({'word': word.lower(),
                        'bias': 1.0})
        if i == 0:  
            features['BOS'] = True
            right_context = self.make_right_context_features(sent,i)
            features.update(right_context[0])
            window = right_context[1]
            window.insert(0, word)
        elif i == len(sent)-1:
            features['EOS'] = True
            left_context = self.make_left_context_features(sent,i)
            features.update(left_context[0])
            window = left_context[1]
            window.append(word)
        else:
            left_context = self.make_left_context_features(sent,i)
            features.update(left_context[0])
            right_context = self.make_right_context_features(sent,i)
            features.update(right_context[0])
            window = left_context[1]
            window.append(word)
            window.extend(right_context[1])
        ngrams = self.ngrams(window) 
        features.update(ngrams[0])
        features.update(ngrams[1])
        if postags == True:
            features.update({'postag': sent[i]['upostag']})
        return features
        
    def sent2features(self, sent, postags = False):
        '''
        Все признаки для одного предложения.
        '''
        return [self.word2features(sent, i, postags) for i in range(len(sent))]

    def word2label_gc(self, word, category):
        '''
        Классы для одного слова, если классы определяются по грамматической категории.
        На вход подаётся слово и интересующая ГК.
        Если слово соответствует переданной ГК, то метка класса определяется по её значению, иначе - О.
        '''
        if word['feats'] is not None and category in word['feats']:
            label = word['feats'][category]
        else:
            label = 'O'
        return label
    
    def sent2labels(self, sent, category, pos = False):
        '''
        Все классы для одного предложения.
        '''
        if pos == True:
            sent_labels = [sent[i]['upostag'] for i in range(len(sent))]
        else:
            sent_labels = [self.word2label_gc(sent[i], category) for i in range(len(sent))]
        return sent_labels
        
class Classifier():
    '''
    Класс для обучения.
    '''
    def __init__(self):
        self.y_pred = []
        
    def training(self, X_train, y_train):
        self.crf = sklearn_crfsuite.CRF()
        self.crf.fit(X_train, y_train)
        
    def predict(self, model, X_test):
        y_pred = model.predict(X_test)
#        labels = list(self.crf.classes_)
#        labels.remove('O')    
        #print(metrics.flat_classification_report(y_test, y_pred, digits=3)) # убрать y_test
        return y_pred
        
    def add_pos_features(self, X, y_pred):
        pos_labels = y_pred.copy()
        for sent_ind in range(len(X)):
            sent_labels = pos_labels.pop(0)
            for word_ind in range(len(X[sent_ind])):
                X[sent_ind][word_ind].update({'postag': sent_labels[word_ind]})
        return X
        
    def pickle_model(self, name):
        '''
        Pickle модели.
        '''
        with open(name, 'wb') as f:
            pickle.dump(self.crf, f)
        
if __name__ == '__main__':
    feat_extr = FeatureExtr()   # объект для извлечения признаков
    clfr_pos = Classifier()     # классификатор для частей речи
    clfr_gc = Classifier()      # классификатор для грам. категорий
    result_train = feat_extr.download('GIKRYA_train1.conllu')   # обучающая выборка
    categories = feat_extr.download_list('categories.txt')      # список грам. категорий
    X_train = [feat_extr.sent2features(sent) for sent in result_train]      # множество признаков (без постэгов)
    y_train = [feat_extr.sent2labels(sent, None, True) for sent in result_train]    # классы - части речи
    clfr_pos.training(X_train, y_train)     # обучение модели для частей речи и её сохранение
    clfr_pos.pickle_model('pos.pickle')
    X_train_new = [feat_extr.sent2features(sent, True) for sent in result_train]    # корректировка множества признаков: 
                                                                                    #добавление к уже имеющимся признака postag
    for category in categories:      # цикл, создающий модели для грам. категорий
        y_train = [feat_extr.sent2labels(sent, category) for sent in result_train]
        clfr_gc.training(X_train_new, y_train)
        clfr_gc.pickle_model('{}.pickle'.format(category))
    '''
    Определение тегов на тестовой выборке.
    '''
    result_test = feat_extr.download('GIKRYA_test1.conllu')     # тестовая выборка
    X_test = [feat_extr.sent2features(sent) for sent in result_test]        # множество признаков
    #y_test = [feat_extr.sent2labels(sent, None, True) for sent in result_test]      # классы - грам. значения грам. категорий
    def load_model(name):
        '''
        Загрузка модели.
        '''
        with open(name, 'rb') as f:
            model = pickle.load(f)
        return model
    pos_model = load_model('pos.pickle')
    pos_pred = clfr_pos.predict(pos_model, X_test)      # определение постэгов на тестовом множестве
    X_test_new = clfr_pos.add_pos_features(X_test, pos_pred)    # добавление полученных постэгов в качестве признаков
    pred_categories = {}
    for category in categories:     # определение грамматических категорий
        #y_test = [feat_extr.sent2labels(sent, category) for sent in result_test]
        gc_model = load_model('{}.pickle'.format(category))
        gc_pred = clfr_gc.predict(gc_model, X_test_new)
        pred_categories.update({category: gc_pred})   # словарь со всеми полученными грам. значениями  
    '''
    Теггирование
    '''
    adp =  feat_extr.download_list('ADP.txt')   # загрузка конечных списков некоторых частей речи 
    conj = feat_extr.download_list('CONJ.txt')
    det = feat_extr.download_list('DET.txt')
    h =  feat_extr.download_list('H.txt')
    part =  feat_extr.download_list('PART.txt')
    pron = feat_extr.download_list('PRON.txt')
    
    for sent_i, sent in enumerate(result_test):    # замена тегов из исходной тестовой выборки на предсказанные
        sent_pos_labels = pos_pred.pop(0)       # частеречные теги для одного предложения
        sent_gc_labels = [pred_categories[category].pop(0) for category in categories]      # теги грам. категорий для одного предложения
        for word_i, word in enumerate(result_test[sent_i]):
            if word['upostag'] != sent_pos_labels[word_i]:      # если постэг в исходной выборке не совпадает с предсказанным,
                                                                # заменить его на предсказанный
                word['upostag'] = sent_pos_labels[word_i]
            if word['lemma'] in adp == 1:     # если слово есть в одном из конечных списков, заменить предсказанный тег нужным
                word['upostag'] = 'ADP'
            if word['lemma'] in conj == 1:
                word['upostag'] = 'CONJ'
            if word['lemma'] in det == 1:
                word['upostag'] = 'DET'
            if word['lemma'] in h == 1:
                word['upostag'] = 'H'
            if word['lemma'] in part == 1:
                word['upostag'] = 'PART'
            if word['lemma'] in pron == 1:
                word['upostag'] = 'PRON'
            for cat_i, category in enumerate(categories):   # замена грам. категорий (ГК)
                if word['feats'] is not None \
                    and category in word['feats'] \
                    and word['feats'][category] != sent_gc_labels[cat_i][word_i]:   # если в исходной выборке у слова указаны какие-либо ГК
                                                                                    # и если очередная категория есть в списке
                                                                                    # и есть значение этой ГК не равно предсказанному
                        if sent_gc_labels[cat_i][word_i] != 'O':        # если предсказан не О-класс, то заменить
                            word['feats'][category] = sent_gc_labels[cat_i][word_i]
                        else:                                           # иначе удалить категорию
                            del word['feats'][category]
                if word['feats'] is not None \
                    and category not in word['feats']:      # если в исходной выборке у слова указаны какие-либо ГК
                                                            # но очередной ГК нет в списке
                        if sent_gc_labels[cat_i][word_i] != 'O':        # если предсказан не О-класс, то вставить новое значение в словарь
                            word['feats'][category] = sent_gc_labels[cat_i][word_i]
                if word['feats'] is None and sent_gc_labels[cat_i][word_i] != 'O':  # если в исх. выборке не указаны никакие ГК, 
                                                                                    # но модель что-то предсказала, то добавить словарь
                    word['feats'] = OrderedDict([(category, sent_gc_labels[cat_i][word_i])])
    with open('results.txt', 'w', encoding='utf-8') as result:      # запись в файл
        for sent in result_test:
            for word in sent:
                result.write('{}\t{}\t{}\t{}\t'.format(word['id'], word['form'], word['lemma'], word['upostag']))
                if word['feats'] is not None and word['feats'] != OrderedDict():
                    keys_list = word['feats'].keys()
                    for i, key in enumerate(keys_list):
                        if i < len(keys_list)-1:
                            result.write('{}={}|'.format(key, word['feats'][key]))
                        else:
                            result.write('{}={}\n'.format(key, word['feats'][key]))
                else:
                    result.write('_\n')
            result.write('\n')
    
    
    
    
    
    
    
    
    
    
    
