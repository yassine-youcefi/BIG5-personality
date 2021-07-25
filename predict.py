import pickle
from model import Model
import matplotlib
matplotlib.use('agg')


class Predictor():
    def __init__(self):

        self.traits = ['OPN', 'CON', 'EXT', 'AGR', 'NEU']
        self.models = {}
        self.load_models()
        self.df = self.load_df()
        self.df = self.agg_avg_personality()

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
                # scaler = MinMaxScaler(feature_range=(0, 50))
                # print(scaler.fit_transform(trait_scores))
                # scaled_trait_scores = scaler.fit_transform(trait_scores)
                predictions['pred_s'+trait] = trait_scores.flatten()[0]
                # predictions['pred_s'+trait] = scaled_trait_scores.flatten()

                trait_categories = pkl_model.predict(X, regression=False)
                predictions['pred_c'+trait] = str(trait_categories[0])
                # predictions['pred_c'+trait] = trait_categories

                trait_categories_probs = pkl_model.predict_proba(X)
                predictions['pred_prob_c' +
                            trait] = trait_categories_probs[:, 1][0]
                # predictions['pred_prob_c'+trait] = trait_categories_probs[:, 1]

        return predictions

    def agg_avg_personality(self):
        # df_mean_scores = df.groupby('NAME')[[
        #     'pred_sOPN', 'pred_sCON', 'pred_sEXT', 'pred_sAGR', 'pred_sNEU',
        # ]].mean()

        df_mean_scores = self.df.groupby(['NAME'], as_index=False).agg(
            {'pred_sOPN': ['mean'], 'pred_sCON': ['mean'], 'pred_sEXT': ['mean'], 'pred_sAGR': ['mean'], 'pred_sNEU': ['mean']})

        df_mean_scores.columns = ['NAME', 'avg_pred_sOPN', 'avg_pred_sCON',
                                  'avg_pred_sEXT', 'avg_pred_sAGR', 'avg_pred_sNEU']

        df = self.df.merge(df_mean_scores, how='right', on='NAME')

        # df_mean_scores = df.groupby('NAME')[[
        #     'pred_prob_cOPN', 'pred_prob_cCON', 'pred_prob_cEXT', 'pred_prob_cAGR', 'pred_prob_cNEU'
        # ]].mean()

        df_mean_probs = df.groupby(['NAME'], as_index=False).agg(
            {'pred_prob_cOPN': ['mean'], 'pred_prob_cCON': ['mean'], 'pred_prob_cEXT': ['mean'], 'pred_prob_cAGR': ['mean'], 'pred_prob_cNEU': ['mean']})
        df_mean_probs.columns = ['NAME', 'avg_pred_prob_cOPN', 'avg_pred_prob_cCON',
                                 'avg_pred_prob_cEXT', 'avg_pred_prob_cAGR', 'avg_pred_prob_cNEU']

        df = df.merge(df_mean_probs, how='right', on='NAME')

        return df

    # def create_plot(self, values, name, compare=False):

    #     plt.cla()
    #     plt.clf()
    #     traits = [
    #         'Openness',
    #         'Conscientiousness',
    #         'Extraversion',
    #         'Agreeableness',
    #         'Neuroticism'
    #     ]

    #     N = len(traits)

    #     # We are going to plot the first line of the data frame.
    #     # But we need to repeat the first value to close the circular graph:
    #     # values=person[self.traits].values.flatten().tolist()
    #     values += values[:1]
    #     values

    #     # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    #     angles = [n / float(N) * 2 * pi for n in range(N)]
    #     angles += angles[:1]

    #     # Initialise the spider plot
    #     if compare:
    #         my_personality_data = self.fb_statuses.find_one({'my_personality': {'$exists': True}}, {
    #             'datetime': 1,
    #             'actual_personality_scores': 1,
    #             'radar_plot_url': 1,
    #             '_id': 0})

    #         ax = self.create_plot(list(
    #             my_personality_data['actual_personality_scores']['percentiles'].values()), 'My_Personality')
    #         filename = 'static/images/' + name + '_Compare.png'
    #     else:
    #         ax = plt.subplot(111, polar=True)
    #         filename = 'static/images/' + name + '.png'

    #     # Draw one axe per variable + add labels labels yet
    #     plt.xticks(angles[:-1], traits, color='grey', size=11)

    #     # Draw ylabels
    #     ax.set_rlabel_position(0)
    #     plt.yticks([10, 20, 30, 40, 50, 60, 70, 80, 90], [
    #                "10", "20", "30", '40', '50', '60', '70', '80', '90'], color="grey", size=8)
    #     plt.ylim(0, 100)

    #     # Plot data
    #     ax.plot(angles, values, linewidth=1, linestyle='solid')

    #     # Fill area
    #     ax.fill(angles, values, 'b', alpha=0.1)

    #     plt.savefig(filename)

    #     return ax


if __name__ == '__main__':
    P = Predictor()
    print('init p')
    P.add_profile_pic()
    print('add profile pic')
    P.predict_fb_statuses()
    print('predict fb statuses')
    P.agg_avg_personality()
    P.insert_avgs_into_db()
    P.add_percentiles()
    P.create_radar_plots()
