'''pip install pymorphy2 # установка библиотеки для лемматизации русских слов
   pip install pyLDAvis '''
# импортирование библиотеки pandas для работы с объетками DataFrames 
import pandas as pd 

# имортирование библиотек для обработки текста
import pymorphy2 # импортирование библиотеки для лемматизации русских слов
from nltk.tokenize import word_tokenize # импортирование токенайзера

from nltk import download
from nltk.corpus import stopwords # импортирование объекта со списком стоп слов
from string import punctuation # импортирование строки с символами пунктуации

# импортирование векторайзера для матрицы совместной встречаемости
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
# Библиотека для сжатого сингулярного разложения
from sklearn.decomposition import TruncatedSVD

# импортирование библиотеки для работы с многомерными массивами
import numpy as np

# импортируем библотеку для визуализации данных в виде графиков
import matplotlib.pyplot as plt

download("stopwords") # загрузка дополнительного пакета stopwords
download('punkt') #загрузка дополнительного пакета punkt
# необходимо загрузить файлы kinopoisk-top250.csv и Test.csv
FILE_PATH = "kinopoisk-top250.csv"
data_frame = data_frame.drop(columns=["url_logo","rating"])# удаляем с датасета бесполезные характеристики фильмов
years_frequency = data_frame["year"].value_counts() # запись частотности годов

# записываем года и их частотности в отдельные списки
years = years_frequency.keys() 
frequency = years_frequency.values

fig, axes = plt.subplots(figsize = (16, 8)) # Создание графика и определение его размеров
axes.bar(years,frequency) # Определение метрик для графика (x,y) 
axes.set_title("Частотность выхода фильмов", fontsize=20) # Определение название графика
axes.grid(True) # Определение сетки для графика, для лучшей читабильности

countries = data_frame["country"] # Запись значений со столбца Country

def country_counter(countries): # можно было просто илспользовать метод value_counts()
  counter = {}
  for sample in countries:
    if sample in counter:
      counter[sample] += 1
    else:
      counter[sample] = 1
  return counter

counter = country_counter(countries) # словарь - счетчик встречаемости стран 

countries = list(counter.keys())[:5] # Список 5-ти самых встречаемых стран в кинопроизводстве топ-250 картин
film_number = list(counter.values())[:5] # Количество встреч 5-ти самых встречаемых стран в кинопроизводстве топ-250 картин

fig2, axes2 = plt.subplots(figsize=(6,6)) # Определение графика и  его размеров
explode = (0,0,0.1,0,0) # Определение расстояния долей от центра pie графика 
axes2.pie(film_number, explode = explode, labels = countries, autopct="%1.1f%%") # Определение графика, как pie график и вставка метрик с параметрами отображения 
axes2.set_title("Соотношение кол-ва производства фильмов между разными странами",fontsize = 20) # Определение название графика
plt.show() # Вывод графика

# Конвертирование строки пунктуационных символов во множество
punct = set(punctuation)
# Основываясь на опыте первичной очистке текста от п.с., добавляем во множество еще немного символом (пример: "славе…")
punct.update({"«", "—","»", "…", "..."})

# Конвертирование списка стоп слов во множество
ru_stop = set(stopwords.words("russian"))
# Основываясь на опыте первичной очистке текста от стоп слов, добавляем во множество еще немного стоп слов
ru_stop.update({"её","оба","обе","который","свой","это","весь","то","всё","самый"})

# Определяем объект MorphAnalyzer
morpheus = pymorphy2.MorphAnalyzer() 
overviews = data_frame["overview"] # записываем в переменную контент колонки overvew 

column_title = "setting"
data_frame[column_title] = "" # создание новой колонки с именем setting

def overviews_to_settings(overviews, column_title):
  string = 0
  for sample in overviews:
    counter = {} # словарь-счетчик слов каждого sample
    for word in word_tokenize(sample, language = "russian"): #токенизирование sample 
      if word[-1] == "…": # Фильтр для очистки токенизированных слов от многоточия в конце (пример: "славе…" --> "славе")
        word = word[:-1]
      word = morpheus.parse(word)[0].normalized.word # Лемматизируем слово
      # записываем в словарь слово и его кол-во размещений в sample, либо, если слово уже существует, прибавляем к кол-ву 1
      if not(word in ru_stop or word in punct): 
        if word in counter:
          counter[word] += 1
        else:
          counter[word] = 1
    data_frame[column_title][string] = counter # записываем словарь счетчик в соответствующую sample строку 
    string += 1 # переходим на следующую строку

overviews_to_settings(overviews, column_title)

def sentence_cleaner(sample): # функция, убирающая стоп слова и знаки препинания, а также леммантизирующая оставшиеся слова 
  new_sentence = ""
  for word in word_tokenize(sample):
    if word[-1] == "…":
      word = word[:-1]
    word = morpheus.parse(word)[0].normalized.word
    if not(word in punct or word in ru_stop):
       new_sentence += word + " "
  return new_sentence

cleared_sentences = [sentence_cleaner(sample) for sample in overviews] # список очищенных предложений

word_vectorizer = CountVectorizer() # Определение векторайзера

word_vectorizer.fit(cleared_sentences) # Создание словаря токенов в векторайзере

table_content = word_vectorizer.transform(cleared_sentences) # Векторизируем предложения
terms = word_vectorizer.get_feature_names() # записывание уникальных слов среди текстов
words_amounts = table_content.toarray().sum(axis=0) # частотность каждого уникального слова в аннотициях фильмов

all_samples_counter = dict(zip(terms, words_amounts)) # словарь уникальных слов и их частотность

all_samples_counter = pd.Series(all_samples_counter) # конвертирование словарь в объект типа pd.Series
all_samples_counter = all_samples_counter.sort_values(ascending = False) # сортировка элементов по убыванию частотности элемента
top_10 = all_samples_counter.iloc[:10] # Срез 10-ти наиболее частовстречаемых слов с их частотностью 

fig3, axis3 = plt.subplots() # Определение графика
fig3.suptitle("Топ 10 встречаемых слов") # Определение названия графика
axis3.barh(top_10.index, top_10.values) # Определение метрик для графика
axis3.grid(True) # Определение сетки для графика

y=data_frame['rating_ball']
x=data_frame['setting']

def f_p(x):
  elements = [element for element  in x]
  return ' '.join(elements)

x = x.apply(f_p)

# переводим строковой X в векторный вид
vectorizer = TfidfVectorizer()
X=vectorizer.fit_transform(x)

# разбиваем X и y на тренировочные и тестовые данные
#test_size- выделяет для тестового набора 30%
# Суть параметра random_state (во всех функциях и методах из SciKit-Learn) в воспроизводимых случайных значениях. 
# Т.е. если явно задать значение random_state отличным от None - то генерируемые псевдослучайные величины будут иметь одни и те же значения при каждом вызове.
X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.3,random_state=17)

# проверяем точность рандомного дерева
# max_depth-максимальная шлубина дерева
# predict-передает на предсказание тестовый X_holdout
regr_tree = DecisionTreeRegressor(max_depth=5,random_state=17).fit(X_train,y_train)
pred = regr_tree.predict(X_holdout)
mean_absolute_error(y_holdout, pred)

# среднее значение по рейтингу
# mean-среднее значение по рейтингу
averege=data_frame['rating_ball'].mean()
averege_pred = [averege]*len(y_holdout)
mean_absolute_error(y_holdout, averege_pred)

dataf="Test.xlsx"
dataf=pd.read_excel(dataf)
new_data=dataf['overview']+' '+dataf['actors']+' '+dataf['screenwriter']

#использование других данных для теста (самый лучший день)
# array([8.19314198]),в то время как реальный балл фильма 4.6
# токенизирует столбик с аннотацией
def tt(tit, wt):
  return word_tokenize(tit)
lem=pymorphy2.MorphAnalyzer()
def lemmatize(tt, lem):
  return [lem.parse(i)[0].normalized.word for i in tt]
tok_dataf = new_data.apply(lambda x: tt(x, word_tokenize))
tok_dataf = new_data.apply(word_tokenize)
lem_dataf= tok_dataf.apply(lambda x: lemmatize(x, lem))
x1=lem_dataf.apply(f_p)
X1=vectorizer.transform(x1)
regr_tree.predict(X1)
