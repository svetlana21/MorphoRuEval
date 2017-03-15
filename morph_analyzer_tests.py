# -*- coding: utf-8 -*-
import unittest
import pprint as pp
from collections import OrderedDict

from morph_analyzer import FeatureExtr

class TestFeatureExtr(unittest.TestCase):

    def setUp(self):
        self.test_feature_extr = FeatureExtr()
        self.test_sent = [OrderedDict([('id', 1),
               ('form', 'Оторвавшись'),
               ('lemma', 'отрываться'),
               ('upostag', 'VERB'),
               ('feats',
                OrderedDict([('Aspect', 'Perf'),
                             ('Tense', 'Past'),
                             ('VerbForm', 'Trans'),
                             ('Voice', 'Act')]))]),
  OrderedDict([('id', 2),
               ('form', 'от'),
               ('lemma', 'от'),
               ('upostag', 'ADP'),
               ('feats', None)]),
  OrderedDict([('id', 3),
               ('form', 'бумаг'),
               ('lemma', 'бумага'),
               ('upostag', 'NOUN'),
               ('feats',
                OrderedDict([('Animacy', 'Inan'),
                             ('Case', 'Gen'),
                             ('Gender', 'Fem'),
                             ('Number', 'Plur')]))]),
  OrderedDict([('id', 4),
               ('form', ','),
               ('lemma', ','),
               ('upostag', 'PUNCT'),
               ('feats', None)]),
  OrderedDict([('id', 5),
               ('form', 'он'),
               ('lemma', 'он'),
               ('upostag', 'PRON'),
               ('feats',
                OrderedDict([('Case', 'Nom'),
                             ('Gender', 'Masc'),
                             ('Number', 'Sing'),
                             ('Person', '3')]))]),
  OrderedDict([('id', 6),
               ('form', 'взглянул'),
               ('lemma', 'взглядывать'),
               ('upostag', 'VERB'),
               ('feats',
                OrderedDict([('Aspect', 'Perf'),
                             ('Gender', 'Masc'),
                             ('Mood', 'Ind'),
                             ('Number', 'Sing'),
                             ('Tense', 'Past'),
                             ('VerbForm', 'Fin'),
                             ('Voice', 'Act')]))]),
  OrderedDict([('id', 7),
               ('form', 'на'),
               ('lemma', 'на'),
               ('upostag', 'ADP'),
               ('feats', None)]),
  OrderedDict([('id', 8),
               ('form', 'Ефимову'),
               ('lemma', 'ефимова'),
               ('upostag', 'PROPN'),
               ('feats',
                OrderedDict([('Animacy', 'Anim'),
                             ('Case', 'Acc'),
                             ('Gender', 'Fem'),
                             ('Number', 'Sing')]))]),
  OrderedDict([('id', 9),
               ('form', '.'),
               ('lemma', '.'),
               ('upostag', 'PUNCT'),
               ('feats', None)])]

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
        
    def test_word2features_without_postag(self):
        '''
        Тест функции word2features (без признака postag и с ним).
        '''
        test_i = 0
        true_result_without_postag = {'+1:pref[0]':'о', '+1:pref[:2]': 'от',
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
        true_result_with_postag = {'+1:pref[0]':'о', '+1:pref[:2]': 'от',
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
                        'postag':'VERB',
                        'pref[0]':'О', 'pref[:2]':'От', 'pref[:3]':'Ото', 'pref[:4]':'Отор',
                        'suf[-1]':'ь', 'suf[-2:]':'сь', 'suf[-3:]':'ись', 'suf[-4:]':'шись',
                        'tri_1':'Оторвавшись от бумаг', 'tri_2':'от бумаг ,',
                        'word':'Оторвавшись',
                        'word_is_digit':False, 'word_is_title':True, 'word_is_upper':False}
        fact_result_without_postag = self.test_feature_extr.word2features(self.test_sent, test_i)
        fact_result_with_postag = self.test_feature_extr.word2features(self.test_sent, test_i, True)
        self.assertCountEqual(true_result_without_postag, fact_result_without_postag)
        self.assertCountEqual(true_result_with_postag, fact_result_with_postag)

    def test_word2label_gc(self):
        '''
        Тест для функции word2label_pos.
        Рассматриваются случаи:
        1) когда у слова есть грам. признаки, но подаваемой на вход ГК нет в их списке (класс О);
        2) когда у слова нет грам. признаков ('feats':None, класс О);
        3) когда подаваемая на вход ГК есть в списке грам. признаков слова (класс, соответствующий грам. значению ГК).
        '''
        test_word1 = self.test_sent[0]
        test_word2 = self.test_sent[1]
        test_word3 = self.test_sent[2]
        test_cat = ('Case')
        true_result1 = 'O'
        true_result2 = 'O'
        true_result3 = 'Gen'
        fact_result1 = self.test_feature_extr.word2label_gc(test_word1, test_cat)
        fact_result2 = self.test_feature_extr.word2label_gc(test_word2, test_cat)
        fact_result3 = self.test_feature_extr.word2label_gc(test_word3, test_cat)
        self.assertEqual(true_result1, fact_result1)
        self.assertEqual(true_result2, fact_result2)
        self.assertEqual(true_result3, fact_result3)
        
    def test_add_pos_features(self):
        X = [[{'+1:pref[0]':'о', '+1:pref[:2]': 'от',
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
                        'word_is_digit':False, 'word_is_title':True, 'word_is_upper':False},
                        {'+1:pref[0]': 'б', '+1:pref[:2]': 'бу', '+1:pref[:3]': 'бум', '+1:pref[:4]': 'бума', '+1:suf[-1]': 'г',
                         '+1:suf[-2:]': 'аг', '+1:suf[-3:]': 'маг', '+1:suf[-4:]': 'умаг', '+1:word_is_digit': False,
                         '+1:word_is_title': False, '+1:word_is_upper': False, '+2:pref[0]': ',', '+2:suf[-1]': ',',
                         '+2:word_is_digit': False, '+2:word_is_title': False, '+2:word_is_upper': False,
                         '+3:pref[0]': 'о', '+3:pref[:2]': 'он', '+3:suf[-1]': 'н', '+3:suf[-2:]': 'он',
                         '+3:word_is_digit': False, '+3:word_is_title': False, '+3:word_is_upper': False,
                         '-1:pref[0]': 'О', '-1:pref[:2]': 'От', '-1:pref[:3]': 'Ото', '-1:pref[:4]': 'Отор',
                         '-1:suf[-1]': 'ь', '-1:suf[-2:]': 'сь', '-1:suf[-3:]': 'ись', '-1:suf[-4:]': 'шись',
                         '-1:word_is_digit': False, '-1:word_is_title': True, '-1:word_is_upper': False,
                         'bi_1': 'Оторвавшись от', 'bi_2': 'от бумаг', 'bi_3': 'бумаг ,', 'bi_4': ', он', 'bias': 1.0,
                         'pref[0]': 'о', 'pref[:2]': 'от', 'suf[-1]': 'т', 'suf[-2:]': 'от', 'tri_1': 'Оторвавшись от бумаг',
                         'tri_2': 'от бумаг ,', 'tri_3': 'бумаг , он', 'word': 'от', 
                         'word_is_digit': False, 'word_is_title': False, 'word_is_upper': False}],
                         [{'+1:pref[0]': ',', '+1:suf[-1]': ',', '+1:word_is_digit': False, '+1:word_is_title': False,
                           '+1:word_is_upper': False, '+2:pref[0]': 'о', '+2:pref[:2]': 'он', '+2:suf[-1]': 'н', '+2:suf[-2:]': 'он',
                           '+2:word_is_digit': False, '+2:word_is_title': False, '+2:word_is_upper': False,
                           '+3:pref[0]': 'в', '+3:pref[:2]': 'вз', '+3:pref[:3]': 'взг', '+3:pref[:4]': 'взгл',
                           '+3:suf[-1]': 'л', '+3:suf[-2:]': 'ул', '+3:suf[-3:]': 'нул', '+3:suf[-4:]': 'янул',
                           '+3:word_is_digit': False, '+3:word_is_title': False, '+3:word_is_upper': False,
                           '-1:pref[0]': 'о', '-1:pref[:2]': 'от', '-1:suf[-1]': 'т', '-1:suf[-2:]': 'от',
                           '-1:word_is_digit': False, '-1:word_is_title': False, '-1:word_is_upper': False,
                           '-2:pref[0]': 'О', '-2:pref[:2]': 'От', '-2:pref[:3]': 'Ото', '-2:pref[:4]': 'Отор',
                           '-2:suf[-1]': 'ь', '-2:suf[-2:]': 'сь', '-2:suf[-3:]': 'ись', '-2:suf[-4:]': 'шись',
                           '-2:word_is_digit': False, '-2:word_is_title': True, '-2:word_is_upper': False,
                           'bi_1': 'Оторвавшись от', 'bi_2': 'от бумаг', 'bi_3': 'бумаг ,', 'bi_4': ', он', 'bi_5': 'он взглянул',
                           'bias': 1.0, 'pref[0]': 'б', 'pref[:2]': 'бу', 'pref[:3]': 'бум', 'pref[:4]': 'бума',
                           'suf[-1]': 'г', 'suf[-2:]': 'аг', 'suf[-3:]': 'маг', 'suf[-4:]': 'умаг',
                           'tri_1': 'Оторвавшись от бумаг', 'tri_2': 'от бумаг ,', 'tri_3': 'бумаг , он', 'tri_4': ', он взглянул',
                           'word': 'бумаг', 'word_is_digit': False, 'word_is_title': False, 'word_is_upper': False},
                           {'+1:pref[0]': 'о', '+1:pref[:2]': 'он', '+1:suf[-1]': 'н', '+1:suf[-2:]': 'он',
                           '+1:word_is_digit': False, '+1:word_is_title': False, '+1:word_is_upper': False,
                           '+2:pref[0]': 'в', '+2:pref[:2]': 'вз', '+2:pref[:3]': 'взг', '+2:pref[:4]': 'взгл',
                           '+2:suf[-1]': 'л', '+2:suf[-2:]': 'ул', '+2:suf[-3:]': 'нул', '+2:suf[-4:]': 'янул',
                           '+2:word_is_digit': False, '+2:word_is_title': False, '+2:word_is_upper': False,
                           '+3:pref[0]': 'н', '+3:pref[:2]': 'на', '+3:suf[-1]': 'а', '+3:suf[-2:]': 'на',
                           '+3:word_is_digit': False, '+3:word_is_title': False, '+3:word_is_upper': False,
                           '-1:pref[0]': 'б', '-1:pref[:2]': 'бу', '-1:pref[:3]': 'бум', '-1:pref[:4]': 'бума',
                           '-1:suf[-1]': 'г', '-1:suf[-2:]': 'аг', '-1:suf[-3:]': 'маг', '-1:suf[-4:]': 'умаг',
                           '-1:word_is_digit': False, '-1:word_is_title': False, '-1:word_is_upper': False,
                           '-2:pref[0]': 'о', '-2:pref[:2]': 'от', '-2:suf[-1]': 'т', '-2:suf[-2:]': 'от',
                           '-2:word_is_digit': False, '-2:word_is_title': False, '-2:word_is_upper': False,
                           '-3:pref[0]': 'О', '-3:pref[:2]': 'От', '-3:pref[:3]': 'Ото', '-3:pref[:4]': 'Отор',
                           '-3:suf[-1]': 'ь', '-3:suf[-2:]': 'сь', '-3:suf[-3:]': 'ись', '-3:suf[-4:]': 'шись',
                           '-3:word_is_digit': False, '-3:word_is_title': True, '-3:word_is_upper': False,
                           'bi_1': 'Оторвавшись от', 'bi_2': 'от бумаг', 'bi_3': 'бумаг ,', 'bi_4': ', он',
                           'bi_5': 'он взглянул', 'bi_6': 'взглянул на', 'bias': 1.0, 'pref[0]': ',', 'suf[-1]': ',',
                           'tri_1': 'Оторвавшись от бумаг', 'tri_2': 'от бумаг ,', 'tri_3': 'бумаг , он',
                           'tri_4': ', он взглянул', 'tri_5': 'он взглянул на', 'word': ',', 
                           'word_is_digit': False, 'word_is_title': False, 'word_is_upper': False}]]
        y_pred = [['VERB','ADP'], ['NOUN','PUNCT']]
        true_result = [[{'+1:pref[0]':'о', '+1:pref[:2]': 'от',
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
                        'postag':'VERB',
                        'pref[0]':'О', 'pref[:2]':'От', 'pref[:3]':'Ото', 'pref[:4]':'Отор',
                        'suf[-1]':'ь', 'suf[-2:]':'сь', 'suf[-3:]':'ись', 'suf[-4:]':'шись',
                        'tri_1':'Оторвавшись от бумаг', 'tri_2':'от бумаг ,',
                        'word':'Оторвавшись',
                        'word_is_digit':False, 'word_is_title':True, 'word_is_upper':False},
                        {'+1:pref[0]': 'б', '+1:pref[:2]': 'бу', '+1:pref[:3]': 'бум', '+1:pref[:4]': 'бума', '+1:suf[-1]': 'г',
                         '+1:suf[-2:]': 'аг', '+1:suf[-3:]': 'маг', '+1:suf[-4:]': 'умаг', '+1:word_is_digit': False,
                         '+1:word_is_title': False, '+1:word_is_upper': False, '+2:pref[0]': ',', '+2:suf[-1]': ',',
                         '+2:word_is_digit': False, '+2:word_is_title': False, '+2:word_is_upper': False,
                         '+3:pref[0]': 'о', '+3:pref[:2]': 'он', '+3:suf[-1]': 'н', '+3:suf[-2:]': 'он',
                         '+3:word_is_digit': False, '+3:word_is_title': False, '+3:word_is_upper': False,
                         '-1:pref[0]': 'О', '-1:pref[:2]': 'От', '-1:pref[:3]': 'Ото', '-1:pref[:4]': 'Отор',
                         '-1:suf[-1]': 'ь', '-1:suf[-2:]': 'сь', '-1:suf[-3:]': 'ись', '-1:suf[-4:]': 'шись',
                         '-1:word_is_digit': False, '-1:word_is_title': True, '-1:word_is_upper': False,
                         'bi_1': 'Оторвавшись от', 'bi_2': 'от бумаг', 'bi_3': 'бумаг ,', 'bi_4': ', он', 'bias': 1.0, 'postag':'ADP',
                         'pref[0]': 'о', 'pref[:2]': 'от', 'suf[-1]': 'т', 'suf[-2:]': 'от', 'tri_1': 'Оторвавшись от бумаг',
                         'tri_2': 'от бумаг ,', 'tri_3': 'бумаг , он', 'word': 'от', 
                         'word_is_digit': False, 'word_is_title': False, 'word_is_upper': False}],
                         [{'+1:pref[0]': ',', '+1:suf[-1]': ',', '+1:word_is_digit': False, '+1:word_is_title': False,
                           '+1:word_is_upper': False, '+2:pref[0]': 'о', '+2:pref[:2]': 'он', '+2:suf[-1]': 'н', '+2:suf[-2:]': 'он',
                           '+2:word_is_digit': False, '+2:word_is_title': False, '+2:word_is_upper': False,
                           '+3:pref[0]': 'в', '+3:pref[:2]': 'вз', '+3:pref[:3]': 'взг', '+3:pref[:4]': 'взгл',
                           '+3:suf[-1]': 'л', '+3:suf[-2:]': 'ул', '+3:suf[-3:]': 'нул', '+3:suf[-4:]': 'янул',
                           '+3:word_is_digit': False, '+3:word_is_title': False, '+3:word_is_upper': False,
                           '-1:pref[0]': 'о', '-1:pref[:2]': 'от', '-1:suf[-1]': 'т', '-1:suf[-2:]': 'от',
                           '-1:word_is_digit': False, '-1:word_is_title': False, '-1:word_is_upper': False,
                           '-2:pref[0]': 'О', '-2:pref[:2]': 'От', '-2:pref[:3]': 'Ото', '-2:pref[:4]': 'Отор',
                           '-2:suf[-1]': 'ь', '-2:suf[-2:]': 'сь', '-2:suf[-3:]': 'ись', '-2:suf[-4:]': 'шись',
                           '-2:word_is_digit': False, '-2:word_is_title': True, '-2:word_is_upper': False,
                           'bi_1': 'Оторвавшись от', 'bi_2': 'от бумаг', 'bi_3': 'бумаг ,', 'bi_4': ', он', 'bi_5': 'он взглянул',
                           'bias': 1.0, 'postag':'NOUN', 'pref[0]': 'б', 'pref[:2]': 'бу', 'pref[:3]': 'бум', 'pref[:4]': 'бума',
                           'suf[-1]': 'г', 'suf[-2:]': 'аг', 'suf[-3:]': 'маг', 'suf[-4:]': 'умаг',
                           'tri_1': 'Оторвавшись от бумаг', 'tri_2': 'от бумаг ,', 'tri_3': 'бумаг , он', 'tri_4': ', он взглянул',
                           'word': 'бумаг', 'word_is_digit': False, 'word_is_title': False, 'word_is_upper': False},
                           {'+1:pref[0]': 'о', '+1:pref[:2]': 'он', '+1:suf[-1]': 'н', '+1:suf[-2:]': 'он',
                           '+1:word_is_digit': False, '+1:word_is_title': False, '+1:word_is_upper': False,
                           '+2:pref[0]': 'в', '+2:pref[:2]': 'вз', '+2:pref[:3]': 'взг', '+2:pref[:4]': 'взгл',
                           '+2:suf[-1]': 'л', '+2:suf[-2:]': 'ул', '+2:suf[-3:]': 'нул', '+2:suf[-4:]': 'янул',
                           '+2:word_is_digit': False, '+2:word_is_title': False, '+2:word_is_upper': False,
                           '+3:pref[0]': 'н', '+3:pref[:2]': 'на', '+3:suf[-1]': 'а', '+3:suf[-2:]': 'на',
                           '+3:word_is_digit': False, '+3:word_is_title': False, '+3:word_is_upper': False,
                           '-1:pref[0]': 'б', '-1:pref[:2]': 'бу', '-1:pref[:3]': 'бум', '-1:pref[:4]': 'бума',
                           '-1:suf[-1]': 'г', '-1:suf[-2:]': 'аг', '-1:suf[-3:]': 'маг', '-1:suf[-4:]': 'умаг',
                           '-1:word_is_digit': False, '-1:word_is_title': False, '-1:word_is_upper': False,
                           '-2:pref[0]': 'о', '-2:pref[:2]': 'от', '-2:suf[-1]': 'т', '-2:suf[-2:]': 'от',
                           '-2:word_is_digit': False, '-2:word_is_title': False, '-2:word_is_upper': False,
                           '-3:pref[0]': 'О', '-3:pref[:2]': 'От', '-3:pref[:3]': 'Ото', '-3:pref[:4]': 'Отор',
                           '-3:suf[-1]': 'ь', '-3:suf[-2:]': 'сь', '-3:suf[-3:]': 'ись', '-3:suf[-4:]': 'шись',
                           '-3:word_is_digit': False, '-3:word_is_title': True, '-3:word_is_upper': False,
                           'bi_1': 'Оторвавшись от', 'bi_2': 'от бумаг', 'bi_3': 'бумаг ,', 'bi_4': ', он',
                           'bi_5': 'он взглянул', 'bi_6': 'взглянул на', 'bias': 1.0, 'postag':'PUNCT', 'pref[0]': ',', 'suf[-1]': ',',
                           'tri_1': 'Оторвавшись от бумаг', 'tri_2': 'от бумаг ,', 'tri_3': 'бумаг , он',
                           'tri_4': ', он взглянул', 'tri_5': 'он взглянул на', 'word': ',', 
                           'word_is_digit': False, 'word_is_title': False, 'word_is_upper': False}]]
        fact_result = self.test_feature_extr.add_pos_features(X, y_pred)
        self.assertCountEqual(true_result, fact_result)
        
from morph_analyzer import Train_Predict

class TestTrain_Predict(unittest.TestCase):

    def setUp(self):
        self.test_train_pred = Train_Predict()
        self.test_sent = [[OrderedDict([('id', 1),
               ('form', 'Оторвавшись'),
               ('lemma', 'отрываться'),
               ('upostag', 'VERB'),
               ('feats',
                OrderedDict([('Tense', 'Past'),
                             ('VerbForm', 'Trans'),
                             ('Voice', 'Act')]))]),
  OrderedDict([('id', 2),
               ('form', 'от'),
               ('lemma', 'от'),
               ('upostag', 'ADP'),
               ('feats', None)])],
  [OrderedDict([('id', 3),
               ('form', 'бумаг'),
               ('lemma', 'бумага'),
               ('upostag', 'NOUN'),
               ('feats',
                OrderedDict([('Animacy', 'Inan'),
                             ('Case', 'Gen'),
                             ('Gender', 'Fem'),
                             ('Number', 'Plur')]))]),
  OrderedDict([('id', 4),
               ('form', ','),
               ('lemma', ','),
               ('upostag', 'PUNCT'),
               ('feats', None)])]]
               
    def test_change_tags(self):
        '''
        Тест функции для теггирования изначально размеченной тестовой выборки.
        Случаи:
        1. Слово 1: первоначальный постэг не совпадает с предсказанным. В word['feats'] изначально есть грам. категории, но они не совпадают с предсказанными.
        2. Слово 2: слово в одном из конечных списков. Предсказанный неверный постэг заменяется на верный.
        word['feats'] изначально None и таковым остаётся.
        3. Слово 3 (с него начинается "предложение 2"): первоначальный постэг совпадает с предсказанным. 
        В word['feats'] есть предсказанная грам. категория Case, но с др. значением.
        Категория Gender предсказана как 'O'. Остальные совпадают.
        4. Слово 4: первоначальный постэг не совпадает с предсказанным. word['feats'] изначально None, но "модель" что-то предсказала.
        '''
        pos_pred_test = [['NOUN', 'PRON'],['NOUN', 'ADJ']]
        pred_categories_test = {'Animacy': [['Anim', 'O'],['Inan', 'O']],
                                 'Case': [['Nom', 'O'],['Acc', 'Loc']],
                                 'Gender': [['O', 'O'],['O', 'Fem']],
                                 'Number': [['O', 'O'],['Plur', 'O']],
                                 'Tense': [['O','O'],['O','O']],
                                 'Person': [['O','O'],['O','O']],
                                 'VerbForm': [['O','O'],['O','O']],
                                 'Mood': [['O','O'],['O','O']],
                                 'Voice': [['O','O'],['O','O']],
                                 'Variant': [['O','O'],['O','O']],
                                 'Degree': [['O','O'],['O','O']],
                                 'Form': [['O','O'],['O','O']]}
        true_result = [[OrderedDict([('id', 1),
               ('form', 'Оторвавшись'),
               ('lemma', 'отрываться'),
               ('upostag', 'NOUN'),
               ('feats',
                OrderedDict([('Animacy', 'Anim'),
                             ('Case', 'Nom')]))]),
  OrderedDict([('id', 2),
               ('form', 'от'),
               ('lemma', 'от'),
               ('upostag', 'ADP'),
               ('feats', OrderedDict())])],
  [OrderedDict([('id', 3),
               ('form', 'бумаг'),
               ('lemma', 'бумага'),
               ('upostag', 'NOUN'),
               ('feats',
                OrderedDict([('Animacy', 'Inan'),
                             ('Case', 'Acc'),
                             ('Number', 'Plur')]))]),
  OrderedDict([('id', 4),
               ('form', ','),
               ('lemma', ','),
               ('upostag', 'ADJ'),
               ('feats', 
                OrderedDict([('Case', 'Loc'),
                            ('Gender', 'Fem')]))])]]
        fact_result = self.test_train_pred.add_tags(self.test_sent, pos_pred_test, pred_categories_test)
        self.assertCountEqual(true_result, fact_result)
        
    def test_add_tags(self):
        '''
        Тест функции для теггирования изначально неразмеченной тестовой выборки.
        '''
        test_sent = [[OrderedDict([('id', 1),
               ('form', 'Оторвавшись')]),
                OrderedDict([('id', 2),
               ('form', 'от')])],
                [OrderedDict([('id', 3),
               ('form', 'бумаг')]),
                OrderedDict([('id', 4),
               ('form', ',')])]]  
        pos_pred_test = [['NOUN', 'PRON'],['NOUN', 'ADJ']]
        pred_categories_test = {'Animacy': [['Anim', 'O'],['Inan', 'O']],
                                 'Case': [['Nom', 'O'],['Acc', 'Loc']],
                                 'Gender': [['O', 'O'],['O', 'Fem']],
                                 'Number': [['O', 'O'],['Plur', 'O']],
                                 'Tense': [['O','O'],['O','O']],
                                 'Person': [['O','O'],['O','O']],
                                 'VerbForm': [['O','O'],['O','O']],
                                 'Mood': [['O','O'],['O','O']],
                                 'Voice': [['O','O'],['O','O']],
                                 'Variant': [['O','O'],['O','O']],
                                 'Degree': [['O','O'],['O','O']],
                                 'Form': [['O','O'],['O','O']]}
        true_result = [[OrderedDict([('id', 1),
               ('form', 'Оторвавшись'),
               ('upostag', 'NOUN'),
               ('feats',
                OrderedDict([('Animacy', 'Anim'),
                             ('Case', 'Nom')]))]),
  OrderedDict([('id', 2),
               ('form', 'от'),
               ('upostag', 'ADP'),
               ('feats', OrderedDict())])],
  [OrderedDict([('id', 3),
               ('form', 'бумаг'),
               ('upostag', 'NOUN'),
               ('feats',
                OrderedDict([('Animacy', 'Inan'),
                             ('Case', 'Acc'),
                             ('Number', 'Plur')]))]),
  OrderedDict([('id', 4),
               ('form', ','),
               ('upostag', 'ADJ'),
               ('feats', 
                OrderedDict([('Case', 'Loc'),
                            ('Gender', 'Fem')]))])]]
        fact_result = self.test_train_pred.add_tags(test_sent, pos_pred_test, pred_categories_test, False)
        pp.pprint(fact_result)
        self.assertCountEqual(true_result, fact_result)

if __name__ == '__main__':
    unittest.main()
