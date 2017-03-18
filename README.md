# MorphoRuEval
Обработка данных в формате conllu и обучение морфологического анализатора. Обучение было произведено  с помощью библиотеки sklearn-crfsuite (алгоритм ’lbfgs’).

 Алгоритм следующий:
1.	Отдельный классификатор обучается определять часть речи. В качестве признаков используются:
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
           
2.	Для каждой отдельной грамматической категории (Animacy, Case и т.д.) обучается свой классификатор. В качестве признаков подаётся всё вышеперечисленное + часть речи.

3.	При теггировании тестовой выборки сначала определяется часть речи. Полученные частеречные теги добавляются к исходному множеству признаков и идёт определение тегов для каждой отдельной грамматической категории.

Для парсинга формата UD с менее чем 10 колонками был использован готовый пакет conllu 0.3 с внесённым в него рядом изменений: https://github.com/svetlana21/conllu

morph_analyzer.py - морфологический анализатор

morph_analyzer_tests.py - тесты к нему

txt_files.7z - вспомогательные текстовые файлы, использующиеся в работе анализатора



Результаты:

Модели обучены на выборке из OpenCorpora:

Лента

2756 меток из 4179, точность 65.95%

8 предложений из 358, точность 2.23%

Вконтакте

2408 меток из 3877, точность 62.11%

49 предложений из 568, точность 8.63%

Худлит

2397 меток из 4042, точность 59.30%

19 предложений из 394, точность 4.82%

Модели обучены на ГИКРЯ:

новости

3705 меток из 4179, точность 88.66%

107 предложений из 358, точность 29.89%

вконтакте

3268 меток из 3877, точность 84.29%

237 предложений из 568, точность 41.73%

худлит

3444 меток из 4042, точность 85.21%

127 предложений из 394, точность 32.23%
