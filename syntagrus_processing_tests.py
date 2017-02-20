# -*- coding: utf-8 -*-
import unittest
import pprint as pp
from collections import OrderedDict

from syntagrus_processing import FeatureExtr

class TestFeatureExtr(unittest.TestCase):

    def setUp(self):
        self.test_feature_extr = FeatureExtr()
        self.test_sent = [OrderedDict([('id', 1),
               ('form', 'Оторвавшись'),
               ('lemma', 'отрываться'),
               ('upostag', 'VERB'),
               ('xpostag', None),
               ('feats',
                OrderedDict([('Aspect', 'Perf'),
                             ('Tense', 'Past'),
                             ('VerbForm', 'Trans'),
                             ('Voice', 'Act')])),
               ('head', 6),
               ('deprel', 'advcl'),
               ('deps', None),
               ('misc', None)]),
  OrderedDict([('id', 2),
               ('form', 'от'),
               ('lemma', 'от'),
               ('upostag', 'ADP'),
               ('xpostag', None),
               ('feats', None),
               ('head', 3),
               ('deprel', 'case'),
               ('deps', None),
               ('misc', None)]),
  OrderedDict([('id', 3),
               ('form', 'бумаг'),
               ('lemma', 'бумага'),
               ('upostag', 'NOUN'),
               ('xpostag', None),
               ('feats',
                OrderedDict([('Animacy', 'Inan'),
                             ('Case', 'Gen'),
                             ('Gender', 'Fem'),
                             ('Number', 'Plur')])),
               ('head', 1),
               ('deprel', 'nmod'),
               ('deps', None),
               ('misc', None)]),
  OrderedDict([('id', 4),
               ('form', ','),
               ('lemma', ','),
               ('upostag', 'PUNCT'),
               ('xpostag', ','),
               ('feats', None),
               ('head', 3),
               ('deprel', 'punct'),
               ('deps', None),
               ('misc', None)]),
  OrderedDict([('id', 5),
               ('form', 'он'),
               ('lemma', 'он'),
               ('upostag', 'PRON'),
               ('xpostag', None),
               ('feats',
                OrderedDict([('Case', 'Nom'),
                             ('Gender', 'Masc'),
                             ('Number', 'Sing'),
                             ('Person', '3')])),
               ('head', 6),
               ('deprel', 'nsubj'),
               ('deps', None),
               ('misc', None)]),
  OrderedDict([('id', 6),
               ('form', 'взглянул'),
               ('lemma', 'взглядывать'),
               ('upostag', 'VERB'),
               ('xpostag', None),
               ('feats',
                OrderedDict([('Aspect', 'Perf'),
                             ('Gender', 'Masc'),
                             ('Mood', 'Ind'),
                             ('Number', 'Sing'),
                             ('Tense', 'Past'),
                             ('VerbForm', 'Fin'),
                             ('Voice', 'Act')])),
               ('head', 0),
               ('deprel', 'root'),
               ('deps', None),
               ('misc', None)]),
  OrderedDict([('id', 7),
               ('form', 'на'),
               ('lemma', 'на'),
               ('upostag', 'ADP'),
               ('xpostag', None),
               ('feats', None),
               ('head', 8),
               ('deprel', 'case'),
               ('deps', None),
               ('misc', None)]),
  OrderedDict([('id', 8),
               ('form', 'Ефимову'),
               ('lemma', 'ефимова'),
               ('upostag', 'PROPN'),
               ('xpostag', None),
               ('feats',
                OrderedDict([('Animacy', 'Anim'),
                             ('Case', 'Acc'),
                             ('Gender', 'Fem'),
                             ('Number', 'Sing')])),
               ('head', 6),
               ('deprel', 'nmod'),
               ('deps', None),
               ('misc', None)]),
  OrderedDict([('id', 9),
               ('form', '.'),
               ('lemma', '.'),
               ('upostag', 'PUNCT'),
               ('xpostag', '.'),
               ('feats', None),
               ('head', 6),
               ('deprel', 'punct'),
               ('deps', None),
               ('misc', None)])]

    def test_all_words_features_title(self):
        '''
        Тест функции all_words_features. Проверка правильности определения слова с заглавной буквы 
        и всех суффиксов и префиксов длиной от 2 до 4 у токена, длина которого > 3.
        '''
        test_data = 'Оторвавшись'
        true_result = [False, True, False, 'О', 'ь', 'От', 'сь', 'Ото', 'ись', 'Отор', 'шись']
        fact_result = self.test_feature_extr.all_words_features(test_data)
        self.assertEqual(true_result, fact_result)
    
    def test_all_words_features_digit(self):
        '''
        Тест функции all_words_features. Проверка правильности определения токена-цифры 
        и всех суффиксов и префиксов длиной от 2 до 4 у токена, длина которого <= 3.
        '''
        test_data = '123'
        true_result = [False, False, True, '1', '3', '12', '23', '123', '123']
        fact_result = self.test_feature_extr.all_words_features(test_data)
        self.assertEqual(true_result, fact_result)
        
    def test_all_words_features_lowercase(self):
        '''
        Тест функции all_words_features. Проверка правильности определения токена lowercase.
        '''
        test_data = 'она'
        true_result = [False, False, False, 'о', 'а', 'он', 'на', 'она', 'она']
        fact_result = self.test_feature_extr.all_words_features(test_data)
        self.assertEqual(true_result, fact_result)
        
    def test_all_words_features_uppercase(self):
        '''
        Тест функции all_words_features. Проверка правильности определения токена uppercase.
        '''
        test_data = 'ЗАО'
        true_result = [True, False, False, 'З', 'О', 'ЗА', 'АО', 'ЗАО', 'ЗАО']
        fact_result = self.test_feature_extr.all_words_features(test_data)
        self.assertEqual(true_result, fact_result)
        
    def test_full_right_context(self):
        '''
        Тест функции make_right_context_features при условии, что правый контекст будет полным (длиной 3 токена)
        '''
        test_i = 0
        true_result = ({'+1:pref[0]':'о', '+1:pref[:2]': 'от',
                        '+1:suf[-1]':'т', '+1:suf[-2:]':'от', 
                        '+1:word_is_digit':False, '+1:word_is_title':False, '+1:word_is_upper':False,
                        '+2:pref[0]':'б','+2:pref[:2]':'бу', '+2:pref[:3]':'бум','+2:pref[:4]':'бума',
                        '+2:suf[-1]':'г','+2:suf[-2:]':'аг','+2:suf[-3:]':'маг','+2:suf[-4:]':'умаг', 
                        '+2:word_is_digit':False, '+2:word_is_title':False, '+2:word_is_upper':False,
                        '+3:pref[0]':',', '+3:suf[-1]':',', 
                        '+3:word_is_digit':False, '+3:word_is_title':False, '+3:word_is_upper':False}, 
                        ['от', 'бумаг', ','])
        fact_result = self.test_feature_extr.make_right_context_features(self.test_sent, test_i)
        self.assertEqual(true_result[0], fact_result[0])
        self.assertEqual(true_result[1], fact_result[1])
        
    def test_1word_right_context(self):
        '''
        Тест функции make_right_context_features при условии, что правый контекст будет длиной в 1 токен
        (текущий токен - предпоследний токен предложения).
        '''
        test_i = 7
        true_result = ({'+1:pref[0]':'.', '+1:suf[-1]':'.',
                        '+1:word_is_digit':False, '+1:word_is_title':False, '+1:word_is_upper':False}, 
                        ['.'])
        fact_result = self.test_feature_extr.make_right_context_features(self.test_sent, test_i)
        self.assertEqual(true_result[0], fact_result[0])
        self.assertEqual(true_result[1], fact_result[1])
    
    def test_2word_right_context(self):
        '''
        Тест функции make_right_context_features при условии, что правый контекст будет длиной 2 токена.
        '''
        test_i = len(self.test_sent)-3
        true_result = ({'+1:pref[0]':'Е', '+1:pref[:2]':'Еф', '+1:pref[:3]':'Ефи', '+1:pref[:4]':'Ефим',
                        '+1:suf[-1]':'у', '+1:suf[-2:]':'ву', '+1:suf[-3:]':'ову', '+1:suf[-4:]':'мову', 
                        '+1:word_is_digit':False, '+1:word_is_title':True, '+1:word_is_upper':False,
                        '+2:pref[0]':'.', '+2:suf[-1]':'.', 
                        '+2:word_is_digit':False, '+2:word_is_title':False, '+2:word_is_upper':False}, 
                        ['Ефимову', '.'])
        fact_result = self.test_feature_extr.make_right_context_features(self.test_sent, test_i)
        self.assertEqual(true_result[0], fact_result[0])
        self.assertEqual(true_result[1], fact_result[1])

    def test_full_left_context(self):
        '''
        Тест функции make_left_context_features при полном левом контексте.
        '''
        test_i = 5
        true_result = ({'-1:pref[0]':'о', '-1:pref[:2]': 'он',
                        '-1:suf[-1]':'н', '-1:suf[-2:]':'он', 
                        '-1:word_is_digit':False, '-1:word_is_title':False, '-1:word_is_upper':False,
                        '-2:pref[0]':',', '-2:suf[-1]':',',
                        '-2:word_is_digit':False, '-2:word_is_title':False, '-2:word_is_upper':False,
                        '-3:pref[0]':'б', '-3:pref[:2]':'бу', '-3:pref[:3]':'бум', '-3:pref[:4]':'бума', 
                        '-3:suf[-1]':'г', '-3:suf[-2:]':'аг','-3:suf[-3:]':'маг','-3:suf[-4:]':'умаг', 
                        '-3:word_is_digit':False, '-3:word_is_title':False, '-3:word_is_upper':False}, 
                        ['бумаг', ',', 'он'])
        fact_result = self.test_feature_extr.make_left_context_features(self.test_sent, test_i)
        self.assertEqual(true_result[0], fact_result[0])
        self.assertEqual(true_result[1], fact_result[1])
        
    def test_1word_left_context(self):
        '''
        Тест функции make_left_context_features. Левый контекст длиной в 1 токен
        (текущий  токен - 2-ой токен в предложении).
        '''
        test_i = 1
        true_result = ({'-1:pref[0]':'О', '-1:pref[:2]':'От', '-1:pref[:3]':'Ото', '-1:pref[:4]':'Отор',
                        '-1:suf[-1]':'ь', '-1:suf[-2:]':'сь', '-1:suf[-3:]':'ись', '-1:suf[-4:]':'шись', 
                        '-1:word_is_digit':False, '-1:word_is_title':True, '-1:word_is_upper':False}, 
                        ['Оторвавшись'])
        fact_result = self.test_feature_extr.make_left_context_features(self.test_sent, test_i)
        self.assertEqual(true_result[0], fact_result[0])
        self.assertEqual(true_result[1], fact_result[1])
        
    def test_2word_left_context(self):
        '''
        Тест функции make_left_context_features. Левый контекст длиной в 2 токена.
        '''
        test_i = 2
        true_result = ({'-1:pref[0]':'о', '-1:pref[:2]': 'от',
                        '-1:suf[-1]':'т', '-1:suf[-2:]':'от', 
                        '-1:word_is_digit':False, '-1:word_is_title':False, '-1:word_is_upper':False,
                        '-2:pref[0]':'О', '-2:pref[:2]':'От', '-2:pref[:3]':'Ото', '-2:pref[:4]':'Отор',
                        '-2:suf[-1]':'ь', '-2:suf[-2:]':'сь', '-2:suf[-3:]':'ись', '-2:suf[-4:]':'шись', 
                        '-2:word_is_digit':False, '-2:word_is_title':True, '-2:word_is_upper':False}, 
                        ['Оторвавшись', 'от'])
        fact_result = self.test_feature_extr.make_left_context_features(self.test_sent, test_i)
        self.assertEqual(true_result[0], fact_result[0])
        self.assertEqual(true_result[1], fact_result[1])
        
    def test_full_window_ngrams(self):
        '''
        Тест функции ngrams при полном окне в 7 токенов.
        '''
        test_data = ['Оторвавшись', 'от', 'бумаг', ',', 'он', 'взглянул', 'на']
        true_result = ({'bi_1':'Оторвавшись от', 'bi_2':'от бумаг', 'bi_3':'бумаг ,', 
                        'bi_4':', он', 'bi_5':'он взглянул', 'bi_6':'взглянул на'},
                        {'tri_1':'Оторвавшись от бумаг', 'tri_2':'от бумаг ,', 'tri_3':'бумаг , он', 
                        'tri_4':', он взглянул', 'tri_5':'он взглянул на'})
        fact_result = self.test_feature_extr.ngrams(test_data)
        pp.pprint(fact_result)
        self.assertEqual(true_result[0], fact_result[0])
        self.assertEqual(true_result[1], fact_result[1])
        
    def test_small_window_ngrams(self):
        '''
        Тест функции ngrams при неполном окне длиной < 7 токенов.
        '''
        test_data = ['Оторвавшись', 'от', 'бумаг', ',']
        true_result = ({'bi_1':'Оторвавшись от', 'bi_2':'от бумаг', 'bi_3':'бумаг ,'},
                        {'tri_1':'Оторвавшись от бумаг', 'tri_2':'от бумаг ,'})
        fact_result = self.test_feature_extr.ngrams(test_data)
        self.assertEqual(true_result[0], fact_result[0])
        self.assertEqual(true_result[1], fact_result[1])
        
    def test_word2features(self):
        '''
        Тест функции word2features.
        '''
        test_i = 0
        true_result = {'+1:pref[0]':'о', '+1:pref[:2]': 'от',
                        '+1:suf[-1]':'т', '+1:suf[-2:]':'от', 
                        '+1:word_is_digit':False, '+1:word_is_title':False, '+1:word_is_upper':False,
                        '+2:pref[0]':'б','+2:pref[:2]':'бу', '+2:pref[:3]':'бум','+2:pref[:4]':'бума',
                        '+2:suf[-1]':'г','+2:suf[-2:]':'аг','+2:suf[-3:]':'маг','+2:suf[-4:]':'умаг', 
                        '+2:word_is_digit':False, '+2:word_is_title':False, '+2:word_is_upper':False,
                        '+3:pref[0]':',', '+3:suf[-1]':',', 
                        '+3:word_is_digit':False, '+3:word_is_title':False, '+3:word_is_upper':False,
                        'bi_1':'Оторвавшись от', 'bi_2':'от бумаг', 'bi_3':'бумаг ,',
                        'bias':1.0,
                        'BOS':True,
                        'pref[0]':'О', 'pref[:2]':'От', 'pref[:3]':'Ото', 'pref[:4]':'Отор',
                        'suf[-1]':'ь', 'suf[-2:]':'сь', 'suf[-3:]':'ись', 'suf[-4:]':'шись',
                        'tri_1':'Оторвавшись от бумаг', 'tri_2':'от бумаг ,',
                        'word':'Оторвавшись',
                        'word_is_digit':False, 'word_is_title':True, 'word_is_upper':False}
        fact_result = self.test_feature_extr.word2features(self.test_sent, test_i)
        pp.pprint(fact_result)
        self.assertCountEqual(true_result, fact_result)

if __name__ == '__main__':
    unittest.main()