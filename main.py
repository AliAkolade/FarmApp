import pickle
import threading
import time

import numpy as np
import pandas as pd
from kivy.app import App
from kivy.clock import Clock, mainthread
from kivy.lang import Builder
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import ScreenManager, Screen


class SplashWindow(Screen):

    def change_screen(self):
        self.manager.current = "Predict"

    pass


class MainWindow(Screen):
    answers = []

    def predict(self):
        X_full = pd.read_csv("Resources/Dataset.csv", index_col="SOILID")

        conditions = [
            (X_full['Nitrogen'] <= 240),
            (X_full['Nitrogen'] > 240) & (X_full['Nitrogen'] <= 480),
            (X_full['Nitrogen'] > 480)]
        choices = ['low', 'medium', 'high']

        X_full['N_level'] = np.select(conditions, choices, default='null')

        conditions = [
            (X_full['Phosphorus'] <= 11),
            (X_full['Phosphorus'] > 11) & (X_full['Phosphorus'] <= 22),
            (X_full['Phosphorus'] > 22)]
        choices = ['low', 'medium', 'high']

        X_full['P_level'] = np.select(conditions, choices, default='null')

        conditions = [
            (X_full['Potassium'] <= 110),
            (X_full['Potassium'] > 110) & (X_full['Potassium'] <= 280),
            (X_full['Potassium'] > 280)]
        choices = ['low', 'medium', 'high']

        X_full['K_level'] = np.select(conditions, choices, default='null')

        conditions = [
            (X_full['N_level'] == 'low'),
            (X_full['N_level'] == 'medium'),
            (X_full['N_level'] == 'high')]
        choices = ['lots-of-nitrogen-fertilizer', 'small-amount-of-nitrogen-fertilizer', 'suitable-nitrogen']

        X_full['suggested_N_fertilizer'] = np.select(conditions, choices, default='null')

        conditions = [
            (X_full['P_level'] == 'low'),
            (X_full['P_level'] == 'medium'),
            (X_full['P_level'] == 'high')]
        choices = ['lots-of-phosphorus-fertilizer', 'small-amount-of-phosphorus-fertilizer', 'suitable-phosphorus']
        X_full['suggested_P_fertilizer'] = np.select(conditions, choices, default='null')

        conditions = [
            (X_full['K_level'] == 'low'),
            (X_full['K_level'] == 'medium'),
            (X_full['K_level'] == 'high')]
        choices = ['lots-of-potassium-fertilizer', 'small-amount-of-potassium-fertilizer', 'suitable-potassium']

        X_full['suggested_K_fertilizer'] = np.select(conditions, choices, default='null')

        X_full['suggestion'] = X_full[
            ['suggested_N_fertilizer', 'suggested_P_fertilizer', 'suggested_K_fertilizer']].agg(', '.join, axis=1)

        X_full.dropna(axis=0, subset=['suggested_N_fertilizer'], inplace=True)
        X_full.drop(['suggested_N_fertilizer'], axis=1, inplace=True)

        X_full.dropna(axis=0, subset=['suggested_P_fertilizer'], inplace=True)
        X_full.drop(['suggested_P_fertilizer'], axis=1, inplace=True)

        X_full.dropna(axis=0, subset=['suggested_K_fertilizer'], inplace=True)
        X_full.drop(['suggested_K_fertilizer'], axis=1, inplace=True)

        X_full.dropna(axis=0, subset=['suggestion'], inplace=True)
        y_new = X_full.suggestion
        X_full.drop(['suggestion'], axis=1, inplace=True)

        from sklearn.preprocessing import LabelEncoder
        lb = LabelEncoder()
        y = lb.fit_transform(y_new)

        filename = 'Resources/xgboost.sav'
        loaded_model = pickle.load(open(filename, 'rb'))

        Nitrogen = float(self.ids.input_n.text)
        Phosphorus = float(self.ids.input_p.text)
        Potassium = float(self.ids.input_k.text)

        new_input = pd.DataFrame({'Nitrogen': [Nitrogen],
                                  'Phosphorus': [Phosphorus],
                                  'Potassium': [Potassium]})

        conditions = [
            (new_input['Nitrogen'] <= 240),
            (new_input['Nitrogen'] > 240) & (new_input['Nitrogen'] <= 480),
            (new_input['Nitrogen'] > 480)]
        choices = ['low', 'medium', 'high']
        new_input['N_level'] = np.select(conditions, choices, default='null')

        conditions = [
            (new_input['Phosphorus'] <= 11),
            (new_input['Phosphorus'] > 11) & (new_input['Phosphorus'] <= 22),
            (new_input['Phosphorus'] > 22)]
        choices = ['low', 'medium', 'high']
        new_input['P_level'] = np.select(conditions, choices, default='null')

        conditions = [
            (new_input['Potassium'] <= 110),
            (new_input['Potassium'] > 110) & (new_input['Potassium'] <= 280),
            (new_input['Potassium'] > 280)]
        choices = ['low', 'medium', 'high']
        new_input['K_level'] = np.select(conditions, choices, default='null')

        categorical_cols2 = list(new_input.columns[new_input.dtypes == 'object'])
        my_cols2 = categorical_cols2
        new_train = new_input[my_cols2].copy

        numerical_cols2 = list(new_input.columns[new_input.dtypes == 'int64'])
        continous_cols2 = list(new_input.columns[new_input.dtypes == 'float64'])

        my_cols3 = numerical_cols2 + continous_cols2
        X_num = new_input[my_cols3].copy()

        new_train = pd.get_dummies(categorical_cols2)

        X_test = pd.concat([X_num, pd.DataFrame(new_train, index=new_input.index)], axis=1)

        new_output = loaded_model.predict(new_input)
        new_output = np.around(new_output)
        new_output = new_output.astype(int)

        new_output = lb.inverse_transform(new_output)
        self.answers = new_output
        self.showResult()

    def showResult(self):
        self.answers = self.answers[0].split(',')
        a1 = "Nitrogen = " + str(self.answers[0].strip().capitalize())
        a2 = "Phosphorous = " + str(self.answers[1].strip().capitalize())
        a3 = "Potassium = " + str(self.answers[2].strip().capitalize())
        if a1 == 'Nitrogen = Lots-of-nitrogen-fertilizer' or a1 == 'Nitrogen = Small-amount-of-nitrogen-fertilizer':
            suggest1 = "Ammonium fertilizer should be added"
        if a2 == 'Phosphorous = Lots-of-phosphorus-fertilizer' or a2 == 'Phosphorous = Small-amount-of-phosphorus-fertilizer':
            suggest2 = "Phosphate fertilizer should be added"
        if a3 == 'Potassium = Lots-of-potassium-fertilizer' or a3 == 'Potassium = Small-amount-of-potassium-fertilizer':
            suggest3 = "Potassh fertilizer should be added"
        ans = suggest1 + "\n\n" + suggest2 + "\n\n" + suggest3
        #ans = a1 + "\n\n" + a2 + "\n\n" + a3
        ans = ans.replace('-', ' ')
        result = Label(text=ans)
        popupWindow = Popup(title="What You Need?", title_align='center', content=result, size_hint=(0.5, 0.5))
        popupWindow.open()


class WindowManager(ScreenManager):
    pass


class Gui(App):
    def build(self):
        return Builder.load_file("gui.kv")


if __name__ == '__main__':
    Gui().run()
