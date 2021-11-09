import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class DataPrep():
    def __init__(self):
        self.trait_cat_dict = {
            'OPN': 'cOPN',
            'CON': 'cCON',
            'EXT': 'cEXT',
            'AGR': 'cAGR',
            'NEU': 'cNEU',
        }
        self.trait_score_dict = {
            'OPN': 'sOPN',
            'CON': 'sCON',
            'EXT': 'sEXT',
            'AGR': 'sAGR',
            'NEU': 'sNEU',
        }

    def prep_data(self, type, trait, regression=False, model_comparison=False):
        df_status = self.prep_status_data()

        tfidf = TfidfVectorizer(stop_words='english', strip_accents='ascii')

        if type == 'status':        
            X = df_status['STATUS']
            
            if regression:
                y_column = self.trait_score_dict[trait]
            else:
                y_column = self.trait_cat_dict[trait]
            y = df_status[y_column]

        return X, y

    def prep_status_data(self):
        df = pd.read_csv(
            'data/myPersonality/personality.csv', encoding="ISO-8859-1")
        df = self.convert_traits_to_boolean(df)
        return df


    def convert_traits_to_boolean(self, df):
        trait_columns = ['cOPN', 'cCON', 'cEXT', 'cAGR', 'cNEU']
        d = {'y': True, 'n': False}

        for trait in trait_columns:
            df[trait] = df[trait].map(d)

        return df
