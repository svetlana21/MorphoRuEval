# -*- coding: utf-8 -*-
from conllu.parser import parse
from nltk.util import ngrams
import pprint as pp

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
        with open(filename) as f:
            data = f.read()
        result = parse(data)
        pp.pprint(result)
        return result
    
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
        if len(word)>1:                                     # префиксы и суффиксы в зависимости от длины слова
            word_features.extend([word[:2],word[-2:]])
        if len(word)>2:
            word_features.extend([word[:3],word[-3:]])
        if len(word)>3:
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
        
    def word2features(self,sent,i):
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
        pp.pprint(features)
        return features
        
if __name__ == '__main__':
    feat_extr = FeatureExtr()
    result = feat_extr.download('test_syntagrus.conllu')
    for sent in result:
        for i in range(len(sent)):
            feat_extr.word2features(sent, i)