"""
This scrip contains a recommendation system model for Olist's returning and new customers. Returning
customers recieve a recommendation based on collaborative filtering (similar users), hot products,
and popular in their area. New users receive recommendations only based on hot products, and popular
in their area.
"""

import pandas as pd

from surprise import Reader, Dataset, SVDpp, dump
from surprise.model_selection import train_test_split

# Helper function to save any resulting dataframes
def save_csv(data, file_name):
    """Saves dataframe to csv.

    Args:
    data -- a pandas dataframe
    file_name -- name for storing csv file
    """
    data.to_csv(f'data/{file_name}.csv', index=False)

# Collaborative filtering model for repeat customers
def surprise_df(data):
    """Returns a user ratings matrix with users incorporated as the index and each product olist
    offers as a unique column.

    Args:
    data -- a dataframe containing user id, item id, and ratings columns in that order.
    """

    scale = (data.estimator.min(), data.estimator.max())
    reader = Reader(rating_scale=scale)

    # The columns must correspond to user id, item id and ratings (in that order).
    user_ratings_matrix = Dataset.load_from_df(data[['customer_unique_id',
                                                     'productId',
                                                     'estimator']], reader)

    return user_ratings_matrix

def final_model(data):
    """Pickles the collaborative filtering recommendation system model for repeat customers.

    Args:
    data -- a dataframe containing user id, item id, and ratings columns in that order.
    """
    # Creates a user ratings surprise matrix for fitting model
    user_ratings_matrix = surprise_df(data)

    # Splits dataset into train and test datasets to generate predictions
    train_set, test_set = train_test_split(user_ratings_matrix, test_size=0.2, random_state=19)

    # Best params determined using GridSearchCV
    params = {'n_factors': 10, 'n_epochs': 50, 'lr_all': 0.01, 'reg_all': 0.1}

    svdpp = SVDpp(n_factors=params['n_factors'],
                  n_epochs=params['n_epochs'],
                  lr_all=params['lr_all'],
                  reg_all=params['reg_all'])

    svdpp.fit(train_set)
    predictions = svdpp.test(test_set)

    # Use surprise wrapper to pickle model
    dump.dump('repeat_customer_model', predictions=predictions, algo=svdpp, verbose=0)

# Popular items and popular in your area for both types of customers

def find_popular_items(data, n_recs):
    """Identifies the most popular items in the dataset based on purchase count and saves them
    as a csv file.

    Args:
    data -- user-product pandas DataFrame
    n_recs -- number of items to be returned
    """

    top_n_items = data.product_id.value_counts().sort_values(ascending=False)[:n_recs].index
    top_n_items_df = pd.DataFrame(list(top_n_items))
    save_csv(top_n_items_df, 'hot_items')

def popular_in_your_area(data, n_recs):
    """Identifies the most popular items in the dataset for each state in Brazil. Saves results
    as a csv file.

    Args:
    data -- user-product pandas DataFrame
    n_recs -- number of items to be returned
    """

    all_states = list(data.customer_state.unique())
    top_in_area_dict = {}

    for state in all_states:
        location_df = data[data.customer_state == state]
        top_n_df = location_df.product_id.value_counts().sort_values(ascending=False)[:n_recs]
        top_n_items = top_n_df.index
        top_in_area_dict[state] = top_n_items

    top_n_items_df = pd.DataFrame(top_in_area_dict)

    save_csv(top_n_items_df, 'in_your_area')

def main():
    """Loads the repeater dataset and outputs a surprise collaborative filtering model"""

    repeater_user_ratings_data = pd.read_csv('data/repeat_user_ratings_data.csv')
    combined_df = pd.read_csv('data/combined_data.csv')
    final_model(repeater_user_ratings_data)
    find_popular_items(combined_df, 10)
    popular_in_your_area(combined_df, 10)
