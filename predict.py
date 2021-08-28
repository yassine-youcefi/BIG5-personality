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

if __name__ == '__main__':
    P = Predictor()
    print('init p ------- ',P)
    
