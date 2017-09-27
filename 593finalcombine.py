# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 17:35:33 2017

@author: manoj, shrey
"""


import pandas as pd
import gzip
from datetime import datetime
import math

from vaderSentiment import SentimentIntensityAnalyzer
import operator

def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
       yield eval(l)


def get_data_frame(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


def get_slices_of_data_frame():

    # Df's required for computation
    df = get_data_frame('meta_Digital_Music.json.gz')
    df2 = get_data_frame('reviews_Digital_Music.json.gz')

    df3 = pd.merge(df, df2, on='asin')
    df4 = df3[['asin', 'title', 'salesRank', 'related', 'overall']]

    # print("Computed DFs:" + str(datetime.now()))

    return {'meta_df': df, 'reviews_df': df2, 'meta_reviews_df': df3, 'meta_reviews_projection_df': df4}


def get_product_name_map(meta_df):
    product_name_map = {}
    for row in meta_df.itertuples():
        product_name_map[row.asin] = row.title
    return product_name_map


def do_cold_start(df3, temp_list, min_rank, max_rank):

    # Cold start
    w1 = 0.4
    w2 = 0.6
    cold = {}
    if len(temp_list) < 3000:
        for row in df3.itertuples():
            cold_id = row.asin
            if cold_id in cold:
                continue
            else:
                sales_rank_record = row.salesRank['Music']
                cold[cold_id] = w1 * row.overall + w2 * ((1 - sales_rank_record/(max_rank - min_rank))*5)

    # print(cold)


def get_recommendation_list_and_score(meta_df, meta_reviews_projection_df, meta_reviews_df, past_purchase_list):

    # Inputs
    category = "Music"
    w1 = 0.4
    w2 = 0.6

    # SalesRank min and max value computation
    sales_rank = []
    sales_rank_dictionary = {}

    for i in range(len(meta_df.index)):
        sales_rank_record = meta_df.salesRank[i]
        try:
            if math.isnan(sales_rank_record):
                sales_rank_dictionary[meta_df.asin[i]] = 0
                i += 1
                continue
        except:
            sales_rank.append(sales_rank_record[category])
            sales_rank_dictionary[meta_df.asin[i]] = sales_rank_record[category]
        i += 1

    min_rank = min(sales_rank)
    max_rank = max(sales_rank)

    # Populating the related items list
    related_items = {}
    for i in range(len(past_purchase_list)):
        past_purchase_id = past_purchase_list[i]
        for index, row in meta_df.iterrows():
            if row.asin == past_purchase_id:
                related_items[past_purchase_id] = row.related
                i = i + 1

    related_items_pruned = []
    for i in range(len(past_purchase_list)):
        if 'buy_after_viewing' in related_items[past_purchase_list[i]]:
            related_items_pruned.extend(related_items[past_purchase_list[i]]['buy_after_viewing'])
        if 'also_bought' in related_items[past_purchase_list[i]]:
            related_items_pruned.extend(related_items[past_purchase_list[i]]['also_bought'])
        if 'bought_together' in related_items[past_purchase_list[i]]:
            related_items_pruned.extend(related_items[past_purchase_list[i]]['bought_together'])

    # convert list to a set to remove duplicates
    related_items_set = list(set(related_items_pruned))

    # do_cold_start(meta_reviews_df, related_items_set, min_rank, max_rank)

    overall_rating_dictionary = {}

    for row in meta_reviews_projection_df.itertuples():
        product_id = row.asin
        if product_id in related_items_set:
            if product_id not in overall_rating_dictionary:
                overall_rating_dictionary[product_id] = [row.overall]
            else:
                overall_rating_dictionary[product_id].append(row.overall)

    for (key, value) in overall_rating_dictionary.items():
        overall_rating_dictionary[key] = pd.np.mean(overall_rating_dictionary[key])

    recommendation_dictionary = {}
    for rec_id in related_items_set:
        if rec_id in sales_rank_dictionary and rec_id in overall_rating_dictionary:
            recommendation_dictionary[rec_id] = \
                w1 * overall_rating_dictionary[rec_id] \
                + w2 * (1 - (sales_rank_dictionary[rec_id] / (max_rank - min_rank)) * 5)

    return {'recommendation_dictionary': recommendation_dictionary, 'related_asin_ids': related_items_set}


def get_overall_score_based_on_sentiment(df3, related_asin_ids):

    analyzer = SentimentIntensityAnalyzer()
    compound_sentiment_dictionary = {}
    compound_helpful_dictionary = {}

    # calculate sentiment for each review for an id and create a dictionary with
    # 'asin' -> {list of compound sentiments for the 'asin'}
    for row in df3.itertuples():
        asin_id = row.asin
        if asin_id in related_asin_ids:

            # Exception Handling: Divide by zero
            if row.helpful[1] != 0:
                help_score = row.helpful[0] / row.helpful[1]
            else:
                help_score = 0
            review_sentiment = analyzer.polarity_scores(row.reviewText)
            if asin_id in compound_sentiment_dictionary:
                compound_sentiment_dictionary[asin_id].append(review_sentiment['compound'])
                compound_helpful_dictionary[asin_id].append(help_score)
            else:
                compound_sentiment_dictionary[asin_id] = [review_sentiment['compound']]
                compound_helpful_dictionary[asin_id] = [help_score]

    # calculate the overall mean of all the compound sentiments for an 'asin'
    for key, list_values in compound_sentiment_dictionary.items():
        compound_sentiment_dictionary[key] = pd.np.mean(compound_sentiment_dictionary[key])
        compound_helpful_dictionary[key] = pd.np.mean(compound_helpful_dictionary[key])

    sentiment_helpful_overall_score = {}
    w1, w2 = 0.5, 0.5

    for i in range(len(related_asin_ids)):
        related_id = related_asin_ids[i]
        if related_id in compound_sentiment_dictionary:
            if related_id not in sentiment_helpful_overall_score:
                score = compound_sentiment_dictionary[related_id] * w1 + compound_helpful_dictionary[related_id] * w2
                sentiment_helpful_overall_score[related_id] = score
        else:
            sentiment_helpful_overall_score[related_id] = 0
        i += 1

    return sentiment_helpful_overall_score


def get_score_based_on_sentiment_and_recommendation_score(recommendation_dictionary, sentiment_helpful_overall_score):

    sentiment_and_sales_rank_overall_score = {}
    sales_rank_wgt = 0.65
    sentiment_wgt = 0.35

    for (key, score_value) in sentiment_helpful_overall_score.items():
        if key in recommendation_dictionary:
            sentiment_and_sales_rank_overall_score[key] = sales_rank_wgt * recommendation_dictionary[key]\
                                                          + sentiment_wgt * sentiment_helpful_overall_score[key]
        else:
            sentiment_and_sales_rank_overall_score[key] = 0

    return sentiment_and_sales_rank_overall_score


def main():
    print("Start of program:" + str(datetime.now()) + "\n")
    print("--------------------------------------------------------------------")

    past_purchase_list = ['7901622466', 'B0000000ZW', 'B00000016W', 'B0000047CX', 'B0000046WB']
    result_df = get_slices_of_data_frame()

    reco_params = get_recommendation_list_and_score(
        result_df['meta_df'], result_df['meta_reviews_projection_df'], result_df['meta_reviews_df'], past_purchase_list)

    # print("\nPrint related_asin_ids length:" + str(len(reco_params['related_asin_ids'])))
    # print("Print related_asin_ids :" + str(reco_params['related_asin_ids']))

    # print("\nRecommendation dictionary length:" + str(len(reco_params['recommendation_dictionary'].items())) + "\n")
    # print("Recommendation dictionary:" + str(sorted(reco_params['recommendation_dictionary'].items(),
    #                                                 key=operator.itemgetter(1), reverse=True)))

    sentiment_helpful_score = get_overall_score_based_on_sentiment(
        result_df['meta_reviews_df'], reco_params['related_asin_ids'])
    output_gen = sorted(sentiment_helpful_score.items(), key=operator.itemgetter(1), reverse=True)

    # print("\nSentiment_helpful_overall_score length:" + str(len(output_gen)))
    # print("Sentiment_helpful_overall_score dict:" + str(output_gen))

    overall_score = get_score_based_on_sentiment_and_recommendation_score(
        reco_params['recommendation_dictionary'], sentiment_helpful_score)
    final_output = sorted(overall_score.items(), key=operator.itemgetter(1), reverse=True)

    product_name_map = get_product_name_map(result_df['meta_df'])

    print("For Past Purchase List of:")
    for product in past_purchase_list:
        print(str(product) + ": " + str(product_name_map[product]))

    print("\nThe following are the recommended products with their recommendation score :")

    for item in final_output:
        # not printing out items with 0 recommendation score
        if item[1] > 0:
            print(str(item[0]) + ": " + str(product_name_map[item[0]]) + " --> " + str(item[1]))

    print("--------------------------------------------------------------------")
    print("\nEnd of program:" + str(datetime.now()))

if __name__ == "__main__":
    main()

# Sample Output:
# Start of program:2017-04-23 02:42:39.600588
#
# --------------------------------------------------------------------
# For Past Purchase List of:
# 7901622466: On Fire
# B0000000ZW: Changing Faces
# B00000016W: Pet Sounds
#
# The following are the recommended products with their recommendation score :
# B0000039Q5: Harry --> 1.24763161695
# B00000010Z: Cars --> 1.23355404088
# B0025KVLTM: Solar Heat --> 0.945973803155
# B000002OU3: Jazz Samba --> 0.935751621385
# --------------------------------------------------------------------
#
# End of program:2017-04-23 02:42:39.688574
