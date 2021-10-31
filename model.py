import pickle
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from data_prep import DataPrep
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

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
            return self.rfr.predict(X)

        else:
            return self.rfc.predict(X)

    def score(self, X, y, regression=True):
        X = self.tfidf.transform(X)
        if regression:
            return self.rfr.score(X, y)
        else:
            return self.rfc.score(X, y)

            


    def predict_proba(self, X, regression=False):
        X = self.tfidf.transform(X)
        if regression:
            raise ValueError('Cannot predict probabilites of a regression!')
        else:
            return self.rfc.predict_proba(X)

if __name__ == '__main__':
    traits = ['OPN', 'CON', 'EXT', 'AGR', 'NEU']
    model = Model()
    regression_accuracy = []
    classification_accuracy = []

    for trait in traits:
        dp = DataPrep()
        X_regression, y_regression = dp.prep_data('status', trait, regression=True, model_comparison=False)
        X_categorical, y_categorical = dp.prep_data('status', trait, regression=False, model_comparison=False)

        # Splitting the data into training set and test set
        X_regression_train, X_regression_test, y_regression_train, y_regression_test = train_test_split(X_regression, y_regression, test_size = 0.2, random_state = 0)
        X_categorical_train, X_categorical_test, y_categorical_train, y_categorical_test = train_test_split(X_categorical, y_categorical, test_size = 0.2, random_state = 0)


        # print(' test regression data >> ',len(X_regression_test),len(y_regression_test))
        # print('Y test regression data >> ',len(y_regression_train),len(X_regression_train))
        # print('X train categorical data >> ',len(X_categorical_train))
        # print('Y train categorical data >> ',len(y_categorical_train))

        print('Fitting trait ' + trait + ' regression model..........................................................')
       
        # # fit regression model 'train the model'
        model.fit(X_regression_train, y_regression_train, regression=True)

        # # predict the classification model with test dataset 
        y_pred = model.predict(X_regression_test, regression=True)

        y_test = y_regression_test
        # # get model accuracy score on test data set 
        score = model.score(X_regression_test, y_pred)
        regression_accuracy.append({'trait': trait, 'score': score})





        print('Fitting trait ' + trait + ' categorical model..........................................................')
        
        # # fit classification model 'train the model'
        model.fit(X_categorical, y_categorical, regression=False)
        
        # # predict the classification model with test dataset 
        y_pred = model.predict(X_categorical_test, regression=False)
        
        y_test = y_categorical_test
        # # get model accuracy score on test data set 
        score = accuracy_score(y_test, y_pred)
        classification_accuracy.append({'trait': trait, 'score': score})

        # with open('static/' + trait + '_model.pkl', 'wb') as f:
        #     # Write the model to a file.
        #     pickle.dump(model, f)
    print('Regression accuracy: ' + str(regression_accuracy))
    print('Classification accuracy: ' + str(classification_accuracy))


    # write the accuracy result into json file 
    with open('static/accuracy.txt', 'w') as f:
        f.write(str(regression_accuracy))
        f.write(str(classification_accuracy))
      

