import wx
import os
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as NavigationToolbar
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pmdarima
from pmdarima.arima import auto_arima
import datetime as dt
from nltk.corpus import wordnet
import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from statsmodels.tsa.statespace.sarimax import SARIMAX
from wordcloud import WordCloud


class p1(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)

        self.figure = plt.figure()
        self.figure.set_size_inches(12, 6)
        self.canvas = FigureCanvasWxAgg(self, -1, self.figure)
        self.toolbar = NavigationToolbar(self.canvas)
        self.toolbar.Hide()

    def onOpen(self):
        dialog = wx.FileDialog(self, "Otwieranie pliku",
                               style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)

        if dialog.ShowModal() == wx.ID_CANCEL:
            return

        path = dialog.GetPath()
        print(path)
        return path





    def plot1a(self,a):
        plik = p1.onOpen(self)
        plt.clf()
        df = pd.read_csv(plik, names=['value'], header=0)
        model = pmdarima.auto_arima(df.value, start_p=1, start_q=1,
                              test='adf',  # use Trueadftest to find optimal 'd'
                              max_p=3, max_q=3,  # maximum p and q
                              m=7,  # frequency of series
                              d=None,  # let model determine 'd'
                              seasonal=True,  # No Seasonality
                              start_P=0,
                              D=0,
                              trace=True,
                              error_action='ignore',
                              suppress_warnings=True,
                              stepwise=True)
        print(model.summary())
        n_periods = a
        fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
        def forecast_accuracy(forecast, actual):
            mape = np.mean(np.abs(forecast - actual) / np.abs(actual))
            mae = np.mean(np.abs(forecast - actual))  # MAE
            mpe = np.mean((forecast - actual) / actual)  # MPE
            rmse = np.mean((forecast - actual) ** 2) ** .5  # RMSE
            print("mae:", mae, "mpe:", mpe, "rmse:", rmse,"mape:", mape)
            return ({'mae': mae, 'mpe': mpe, 'rmse': rmse})

        def mean_absolute_percentage_error(y_true, y_pred):
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        mape=mean_absolute_percentage_error(df.values,fc)


        print("MAPE",mape)
        forecast_accuracy(fc, df.values)
        index_of_fc = np.arange(len(df.value), len(df.value) + n_periods)
        df2 = pd.DataFrame({'Przewidzana wartosc': fc})
        df1 = df2.head(a)
        f = open("Przewidywanie_Arima.txt", "w")
        f.write(str(df1))
        f.close()
        print(df1)
        fc_series = pd.Series(fc, index=index_of_fc)
        lower_series = pd.Series(confint[:, 0], index=index_of_fc)
        upper_series = pd.Series(confint[:, 1], index=index_of_fc)

        plt.plot(df.value)
        plt.xticks(np.arange(0, len(df.value), 30),rotation=90)
        plt.plot(fc_series, color='darkgreen')
        plt.fill_between(lower_series.index,
                         lower_series,
                         upper_series,
                         color='k', alpha=0.15)
        plt.ylabel("liczba graczy online")
        plt.title("ARIMA")
        plt.subplots_adjust(bottom=0.20)
        plt.plot()
        self.canvas.draw()

    def porownanie(self):
        plt.clf()
        plik = p1.onOpen(self)
        data = pd.read_csv(plik)

        data.describe()

        with open(plik) as csvfile:
            row_count = sum(1 for row in csvfile)
            print(row_count if row_count else 'Empty')


        data['DateTime'] = pd.to_datetime(data['DateTime'])
        data['DateTime'] = data['DateTime'].map(dt.datetime.toordinal)

        X = data['DateTime'].values.reshape(-1, 1)
        Y = data['Players'].values.reshape(-1, 1)

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

        regressor = LinearRegression()
        regressor.fit(X_train, y_train)  # training the algorithm

        # To retrieve the intercept:
        print(regressor.intercept_)
        # For retrieving the slope:
        print(regressor.coef_)

        y_pred = regressor.predict(X_test)
        df = pd.DataFrame({'Wartosci aktualne': y_test.flatten(), 'Wartosci przewidziane': y_pred.flatten()})
        df1 = df.head(10)
        df1.to_csv(r'Regresja_przewidziane_wartosci.csv')
        print(df1)

        def MAPE(forecast, actual):
            mape = np.mean(np.abs(forecast - actual) / np.abs(actual)) * 100
            return mape

        mae1=metrics.mean_absolute_error(y_test, y_pred)
        mse1=metrics.mean_squared_error(y_test, y_pred)
        rmse1=np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        mape1 = MAPE(y_pred, y_test)




        df2 = pd.read_csv(plik)
        relp = os.path.relpath(plik)
        if  relp == 'Bloodstained_data.csv':
            model = SARIMAX(df2['Players'], order=(2,0,1), seasonal_order=(2,0,0,7), trend='n')
        elif relp == 'Resident_data.csv':
            model = SARIMAX(df2['Players'], order=(2,0,0), seasonal_order=(1,0,1,7), trend='n')
        elif relp == 'Divinity2_data.csv':
            model = SARIMAX(df2['Players'], order=(1,0,0), seasonal_order=(2,0,0,7), trend='n')
        elif relp == 'Hellblade_data.csv':
            model = SARIMAX(df2['Players'], order=(2,0,1), seasonal_order=(1,0,1,7), trend='n')
        elif relp == 'Plague_data.csv':
            model = SARIMAX(df2['Players'], order=(3,0,0), seasonal_order=(1,0,0,7), trend='n')
        elif relp == 'DragonDogma_data.csv':
            model = SARIMAX(df2['Players'], order=(3,0,2), seasonal_order=(2,0,0,7), trend='n')

        res = model.fit()
        print(res.summary())


        nauka = res.predict(start=0, end=(row_count-2), Dynamic=True)
        df3 = pd.DataFrame({'Wartosci aktualne': df2['Players'], 'Wartosci przewidziane': nauka})
        df3H=df3.tail(10)
        df3H.to_csv(r'Arima_przewidziane_wartosci.csv')

        mae2 = metrics.mean_absolute_error(df2['Players'], nauka)
        mse2 = metrics.mean_squared_error(df2['Players'], nauka)
        rmse2 = np.sqrt(metrics.mean_squared_error(df2['Players'],nauka))
        mape2 = MAPE(nauka, df2['Players'])

        print(mae1,mse1,rmse1,mape1)
        print(mae2, mse2, rmse2, mape2)
        if mape2<mape1:
            wynik='Dla danej probki model ARIMA jest lepszym modelem'
        elif mape2>mape1:
            wynik = 'Dla danej probki regresja liniowa jest lepszym modelem'

        f = open("Porownanie_Algorytmow.txt", "w")
        f.write("Regresja liniowa:\nRMSE: {} MSE: {} MAE: {} \nMAPE: {}\n\n".format(rmse1,mse1,mae1,mape1))
        f.write("ARIMA:\nRMSE: {} MSE: {} MAE: {} \nMAPE: {}".format(rmse2,mse2,mae2,mape2))
        f.write("\n {}".format(wynik))
        f.close()

        plt.plot(df3)
        plt.yticks([])
        plt.plot(nauka, color="red",label='predykcja')
        plt.ylabel("Ilosc osob online")
        plt.xlabel("Dni od premiery gry")
        self.canvas.draw()

    def textMinin(self):
        plt.clf()
        plik = p1.onOpen(self)
        review = pd.read_csv(plik)

        # tworzenie oznaczenia recenzji o zlym  wydzwieku (ocena < 5)
        review["is_bad_review"] = review["recomend"].apply(lambda x: 1 if x == "Not Recommended" else 0)
        # wybieranie tylko potrzebych kolumn
        review = review[["review", "is_bad_review"]]
        # zamiana danych w kolumnie "review" na string
        review['review'] = review['review'].astype(str)

        review.head()

        # Podzial danych na probke
        #review = review.sample(frac = 0.3, replace = False, random_state=42)

        # obrabianie danych
        def get_wordnet_pos(pos_tag):
            if pos_tag.startswith('J'):
                return wordnet.ADJ
            elif pos_tag.startswith('V'):
                return wordnet.VERB
            elif pos_tag.startswith('N'):
                return wordnet.NOUN
            elif pos_tag.startswith('R'):
                return wordnet.ADV
            else:
                return wordnet.NOUN

        def clean_text(text):
            # male litery
            text = text.lower()
            # tokenizacja i usuwanie interpunkcji
            text = [word.strip(string.punctuation) for word in text.split(" ")]
            # usuwanie slow zawierajacych cyfry
            text = [word for word in text if not any(c.isdigit() for c in word)]
            # usuwanie "stop" slow ('the', 'a' ,'this')
            stop = stopwords.words('english')
            text = [x for x in text if x not in stop]
            # usuwanie pustych tokenow
            text = [t for t in text if len(t) > 0]
            # oznaczanie slow POS (rzeczownik,przymiotnik,itd)
            pos_tags = pos_tag(text)
            # lemmanizacja tekstu (odmieniona forma do bezokolicznika, jesli istnieje)
            text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
            # usuwanie slow jednoliterowych
            text = [t for t in text if len(t) > 1]
            # fuzja tekstu
            text = " ".join(text)
            return (text)

        review["review_clean"] = review["review"].apply(lambda x: clean_text(x))

        # uzycie Vader do sprawdzenia nastroju slow do odroznienia negatywnych od pozytywnych
        sid = SentimentIntensityAnalyzer()
        review["sentiments"] = review["review"].apply(lambda x: sid.polarity_scores(x))
        review = pd.concat([review.drop(['sentiments'], axis=1), review['sentiments'].apply(pd.Series)], axis=1)

        # liczba liter
        review["nb_chars"] = review["review"].apply(lambda x: len(x))

        # liczba slow
        review["nb_words"] = review["review"].apply(lambda x: len(x.split(" ")))

        # reprezentacja wektorowa kazdej recenzji

        documents = [TaggedDocument(doc, [i]) for i, doc in
                     enumerate(review["review_clean"].apply(lambda x: x.split(" ")))]

        # trening Doc2Vec
        model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)

        # przetwarzanie danych do danych wektorowych (Wymagane w Doc2Vec)
        doc2vec_df = review["review_clean"].apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)
        doc2vec_df.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_df.columns]
        review = pd.concat([review, doc2vec_df], axis=1)

        # dodawanie wartosci TF-IDF dla kazdego slowa
        tfidf = TfidfVectorizer(min_df=10)
        tfidf_result = tfidf.fit_transform(review["review_clean"]).toarray()
        tfidf_df = pd.DataFrame(tfidf_result, columns=tfidf.get_feature_names())
        tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
        tfidf_df.index = review.index
        review = pd.concat([review, tfidf_df], axis=1)

        # pokazanie dystrybucji procentowej dobrych do zlych recenzji
        f = open("wyniki recenzji.txt", "w")
        print('dystrybucja dobrych do zlych recenzji:')
        print(review["is_bad_review"].value_counts(normalize=True))
        f.write('dystrybucja dobrych do zlych recenzji:\n')
        f.write(str(review["is_bad_review"].value_counts(normalize=True)))
        f.close()
        def show_wordcloud(data, title=None):
            wordcloud = WordCloud(
                background_color='white',
                max_words=200,
                max_font_size=40,
                scale=3,
                random_state=42
            ).generate(str(data))

            fig = plt.figure(1, figsize=(20, 20))
            plt.axis('off')
            if title:
                fig.suptitle(title, fontsize=20)
                fig.subplots_adjust(top=2.3)
            plt.imshow(wordcloud)
            self.canvas.draw()



        show_wordcloud(review["review"])
        # wypisanie 10 najbardziej pozytywnych recenzji
        print('wypisanie 10 najbardziej pozytywnych recenzji:')

        print(review[review["nb_words"] >= 5].sort_values("pos", ascending=False)[["review", "pos"]].head(10))
        f = open("wyniki recenzji.txt", "a")
        f.write('\nwypisanie 10 najbardziej pozytywnych recenzji:\n')
        f.write(str(review[review["nb_words"] >= 5].sort_values("pos", ascending=False)[["review", "pos"]].head(10)))
        f.close()








class TestFrame(wx.Frame):
    def __init__(self, parent, title):
        wx.Frame.__init__(self, parent, title=title, size=(1200, 800), style=wx.MINIMIZE_BOX | wx.SYSTEM_MENU |
                                                                            wx.CAPTION | wx.CLOSE_BOX | wx.CLIP_CHILDREN)


        self.sp = wx.SplitterWindow(self)
        self.p1 = p1(self.sp)
        self.p2 = wx.Panel(self.sp, style=wx.SUNKEN_BORDER)

        self.sp.SplitHorizontally(self.p1, self.p2, 600)

        self.my_text = wx.TextCtrl(self.p2, style=wx.TE_MULTILINE, pos=(650, 5),size=(530, 120))
        self.btnt = wx.Button(self.p2, label='Wyniki porownania', pos=(490, 10))
        self.btnt.Bind(wx.EVT_BUTTON, self.otwarcieTt)

        self.porownanie = wx.Button(self.p2, label='ARIMA vs Regresja liniowa', pos=(330, 10))
        self.porownanie.Bind(wx.EVT_BUTTON, self.armreg)

        self.statusbar = self.CreateStatusBar()
        self.statusbar.SetStatusText("")

        self.plotbut = wx.Button(self.p2, -1, "ARIMA", size=(50, 30), pos=(160, 10))
        self.plotbut.Bind(wx.EVT_BUTTON, self.plot1)

        self.textm = wx.Button(self.p2, -1, "Analiza recenzji", size=(100, 30), pos=(330, 50))
        self.textm.Bind(wx.EVT_BUTTON, self.textMining)

        self.textwynik = wx.Button(self.p2, -1, "rozklad recenzji oraz 10 najlepszych", size=(200, 30), pos=(430, 50))
        self.textwynik.Bind(wx.EVT_BUTTON, self.otwarcieMn)

        self.sibut = wx.Button(self.p2, -1, "Zoom", size=(40, 20), pos=(60, 10))
        self.sibut.Bind(wx.EVT_BUTTON, self.zoom)

        self.hmbut = wx.Button(self.p2, -1, "Home", size=(40, 20), pos=(110, 10))
        self.hmbut.Bind(wx.EVT_BUTTON, self.home)

        self.hibut = wx.Button(self.p2, -1, "Pan", size=(40, 20), pos=(10, 10))
        self.hibut.Bind(wx.EVT_BUTTON, self.pan)

        self.radio = wx.RadioButton(self.p2, label="10 dni", style=wx.RB_GROUP, pos=(160, 45))
        self.radio2 = wx.RadioButton(self.p2, label="20 dni",pos=(160, 70))
        self.radio3 = wx.RadioButton(self.p2, label="60 dni", pos=(160, 95))
        self.Bind(wx.EVT_RADIOBUTTON, self.OnRadiogroup)




    def OnRadiogroup(self, event):
        rb = event.GetEventObject()
        if rb.GetLabel()=="10 dni":
            self.p1.plot1a(10)
        elif rb.GetLabel()=="20 dni":
            self.p1.plot1a(20)
        else:
            self.p1.plot1a(60)

    def zoom(self, event):
        self.statusbar.SetStatusText("Zoom")
        self.p1.toolbar.zoom()

    def home(self, event):
        self.statusbar.SetStatusText("Home")
        self.p1.toolbar.home()

    def pan(self, event):
        self.statusbar.SetStatusText("Pan")
        self.p1.toolbar.pan()

    def plot1(self, event):
        self.p1.plot1a(30)

    def textMining(self, event):
        self.p1.textMinin()

    def armreg(self, event):
        self.p1.porownanie()

    def otwarcieP(self, event):
        self.p1.onOpen()
    def otwarcieTt(self, event):
        self.my_text.Clear()
        path ="Porownanie_Algorytmow.txt"
        if os.path.exists(path):
            with open(path) as fobj:
                for line in fobj:
                    self.my_text.WriteText(line)

    def otwarcieMn(self, event):
        self.my_text.Clear()
        path = "wyniki recenzji.txt"
        if os.path.exists(path):
            with open(path) as fobj:
                for line in fobj:
                    self.my_text.WriteText(line)

app = wx.App(redirect=False)
frame = TestFrame(None, "Porownanie szeregow czasowych oraz analiza recenzji")
frame.Show()
app.MainLoop()