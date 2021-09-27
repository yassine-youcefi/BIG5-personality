import pickle
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from data_prep import DataPrep
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
class Model():
    def __init__(self):
        self.rfr = RandomForestRegressor(bootstrap=True,
         max_features='sqrt',
         min_samples_leaf=1,
         min_samples_split=2,
         n_estimators= 200)
        self.rfc = RandomForestClassifier(max_features='sqrt', n_estimators=110)
        self.tfidf = TfidfVectorizer(stop_words='english', strip_accents='ascii')

    def fit(self, X, y, regression=True):
        X = self.tfidf.fit_transform(X)
        if regression:
            self.rfr = self.rfr.fit(X, y)
            
        else:
            self.rfc = self.rfc.fit(X, y)
            
    def predict(self, X, regression=True):
        X = self.tfidf.transform(X)
        if regression:
            print("regression accuracy ", self.rfr.score(X, y))
            return self.rfr.predict(X)

        else:
            print("classification accuracy ", self.rfc.score(X, y))
            return self.rfc.predict(X)

    def predict_proba(self, X, regression=False):
        X = self.tfidf.transform(X)
        if regression:
            raise ValueError('Cannot predict probabilites of a regression!')
        else:
            return self.rfc.predict_proba(X)

if __name__ == '__main__':
    traits = ['OPN', 'CON', 'EXT', 'AGR', 'NEU']
    model = Model()

    for trait in traits:
        dp = DataPrep()
        X_regression, y_regression = dp.prep_data('status', trait, regression=True, model_comparison=False)
        X_categorical, y_categorical = dp.prep_data('status', trait, regression=False, model_comparison=False)

        # Splitting the data into training set and test set
        X_regression_train, X_regression_test, y_regression_train, y_regression_test = train_test_split(X_regression, y_regression, test_size = 0.2, random_state = 0)
        X_categorical_train, X_categorical_test, y_categorical_train, y_categorical_test = train_test_split(X_categorical, y_categorical, test_size = 0.2, random_state = 0)


        # Fitting Random Forest Regression to the Training set
        print('Fitting trait ' + trait + ' regression model...')
        regression_model = model.fit(X_regression_train, y_regression_train, regression=True).predict(X_regression_test, regression=True)
        print('regression accuracy', model.score(X_regression_test, y_regression_test, regression=True))
        print('Done!')

        # Fitting Random Forest Classification to the Training set
        print('Fitting trait ' + trait + ' categorical model...')
        categorical_model = model.fit(X_categorical_train, y_categorical_train, regression=False).predict(X_categorical_test, regression=False)
        print('categorical accuracy', model.score(X_categorical_test, y_categorical_test, regression=False))

        print('Done!')

        # with open('static/' + trait + '_model.pkl', 'wb') as f:
        #     # Write the model to a file.
        #     pickle.dump(model, f)

      

