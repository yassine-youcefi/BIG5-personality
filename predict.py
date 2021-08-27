import pickle
from model import Model
import matplotlib
matplotlib.use('agg')


class Predictor():
    def __init__(self):

        self.traits = ['OPN', 'CON', 'EXT', 'AGR', 'NEU']
        self.models = {}
        self.load_models()
        

    def load_models(self):
        M = Model()
        for trait in self.traits:
            with open('static/' + trait + '_model.pkl', 'rb') as f:
                self.models[trait] = pickle.load(f)

    def predict(self, X, traits='All', predictions='All'):
        predictions = {}
        if traits == 'All':
            for trait in self.traits:
                pkl_model = self.models[trait]

                trait_scores = pkl_model.predict(
                    X, regression=True).reshape(1, -1)
                
                predictions['pred_s'+trait] = trait_scores.flatten()[0]

                trait_categories = pkl_model.predict(X, regression=False)
                predictions['pred_c'+trait] = str(trait_categories[0])

                trait_categories_probs = pkl_model.predict_proba(X)
                predictions['pred_prob_c' +
                            trait] = trait_categories_probs[:, 1][0]

        return predictions

    def agg_avg_personality(self):
      

        df_mean_scores = self.df.groupby(['NAME'], as_index=False).agg(
            {'pred_sOPN': ['mean'], 'pred_sCON': ['mean'], 'pred_sEXT': ['mean'], 'pred_sAGR': ['mean'], 'pred_sNEU': ['mean']})

        df_mean_scores.columns = ['NAME', 'avg_pred_sOPN', 'avg_pred_sCON',
                                  'avg_pred_sEXT', 'avg_pred_sAGR', 'avg_pred_sNEU']

        df = self.df.merge(df_mean_scores, how='right', on='NAME')

       

        df_mean_probs = df.groupby(['NAME'], as_index=False).agg(
            {'pred_prob_cOPN': ['mean'], 'pred_prob_cCON': ['mean'], 'pred_prob_cEXT': ['mean'], 'pred_prob_cAGR': ['mean'], 'pred_prob_cNEU': ['mean']})
        df_mean_probs.columns = ['NAME', 'avg_pred_prob_cOPN', 'avg_pred_prob_cCON',
                                 'avg_pred_prob_cEXT', 'avg_pred_prob_cAGR', 'avg_pred_prob_cNEU']

        df = df.merge(df_mean_probs, how='right', on='NAME')

        return df

if __name__ == '__main__':
    P = Predictor()
    print('init p')
    P.agg_avg_personality()
