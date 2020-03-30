"""
This script creates a connection to PostgreSQL in AWS and pulls data needed for
analysis and building a recommendation system. Data is then cleaned and the resulting
table is saved as a csv file.
"""

import pandas as pd
from connect_to_aws import access_postgres_in_aws

def save_csv(data, file_name):
    """Saves dataframe to csv.

    Args:
    data -- a pandas dataframe
    file_name -- name for storing csv file
    """
    data.to_csv(f'data/{file_name}.csv', index=False)

def query_aws(aws_engine):
    """returns the pandas dataframe outlined in the query

    Args:
    aws_engine -- the connection to AWS
    """
    query = """
            SELECT
                customers.customer_unique_id,
                customers.customer_zip_code_prefix,
                customers.customer_city,
                customers.customer_state,
                order_item.order_id,
                order_item.product_id,
                order_item.seller_id,
                order_item.price,
                orders.order_purchase_timestamp,
                orders.order_delivered_customer_date,
                orders.order_estimated_delivery_date,
                payments.payment_type,
                payments.payment_installments,
                payments.payment_value,
                reviews.review_score,
                products.product_weight_g,
                product_category.product_category_name_english,
                sellers.seller_zip_code_prefix,
                sellers.seller_state,
                sellers.seller_state
            FROM customers
                JOIN orders
                    on orders.customer_id=customers.customer_id
                JOIN reviews
                    on reviews.order_id=orders.order_id
                JOIN order_item
                    on order_item.order_id=orders.order_id
                JOIN payments
                    on payments.order_id=orders.order_id
                JOIN products
                    on products.product_id=order_item.product_id
                JOIN product_category
                    on product_category.product_category_name=products.product_category_name
                JOIN sellers
                    on sellers.seller_id=order_item.seller_id

            """

    return pd.read_sql(query, aws_engine)

# Cleaning data
def create_total_payment_value(data):
    """Returns the dataset with a new column 'total_payments' that accounts for payments
    made by vouchers, which are split amongst multiple lines. Then drops 'payment_value'
    column as it is redundent.

    Args:
    data -- pandas dataframe with total_value column that indicates each purchase value
    """
    data['total_payment'] = data['payment_value'].groupby(data['order_id']).transform('sum')
    data = data.drop('payment_value', axis=1)
    return data

def duplicates(data):
    """Returns the dataframe with the duplicate rows dropped."""

    data = data.drop_duplicates(keep='first')

    return data

def convert_to_datetime(data):
    """Returns the dataframe with time-stamped columns converted to datetime objects."""

    data.order_purchase_timestamp = pd.to_datetime(data.order_purchase_timestamp,
                                                   format="%Y/%m/%d %H:%M:%S")
    data.order_delivered_customer_date = pd.to_datetime(data.order_delivered_customer_date,
                                                        format="%Y/%m/%d %H:%M:%S")
    data.order_estimated_delivery_date = pd.to_datetime(data.order_estimated_delivery_date,
                                                        format="%Y/%m/%d %H:%M:%S")
    return data

# Feature engineering
def repeat_and_first_time(data):
    """Returns two datasets, one with only repeat customers and one with only
    first time customers. Saves both datasets to csv files.
    """
    repeaters = data.groupby('customer_unique_id').filter(lambda x: len(x) > 1)
    first_timers = data.groupby('customer_unique_id').filter(lambda x: len(x) == 1)

    # Save datasets to csv files
    save_csv(repeaters, 'repeater_data')
    save_csv(first_timers, 'first_timers_data')

    return repeaters, first_timers

def regenerate_dataset_with_indicators(repeaters, first_timers):
    """Saves a csv dataset that is a concatination of the two imput datasets and assigns
    a 1 to any repeat customers and a 0 to first time customers indicated in the 'repeater'
    column.

    Args:
    repeater_data -- dataset that receives the 1 indication of repeat customers
    first_timer_data -- dataset that receives the 1 indication of first time customers
    """
    repeaters['repeater'] = 1
    first_timers['repeater'] = 0

    combined_df = pd.concat((repeaters, first_timers), axis=0)
    combined_df = combined_df.drop('Unnamed: 0', axis=1).reset_index()

    save_csv(combined_df, 'combined_data')

def create_user_ratings_df(data):
    """Saves a csv dataset containing a users column, product column, and ratings column
    for repeat customers.
    """

    user_prod_reviewscore_data = data.groupby(['customer_unique_id', 'product_id']
                                             )['review_score'].agg(['mean']).reset_index()

    user_prod_reviewscore_data = user_prod_reviewscore_data.rename({'mean':'estimator',
                                                                    'product_id':'productId'},
                                                                   axis=1)

    save_csv(user_prod_reviewscore_data, 'repeat_user_ratings_data')


def main():
    """Calls internal and external functions to the script to pull data from aws, clean data,
    create additional dataframes, and engineer features. All dataframes are saved as csv
    files.
    """

    # Calls external function to connect to aws using information stored externally
    aws_engine = access_postgres_in_aws()

    # Call internal functions to this script
    olist_data = query_aws(aws_engine)
    olist_data = create_total_payment_value(olist_data)
    olist_data = duplicates(olist_data)
    olist_data = convert_to_datetime(olist_data)
    repeater_data, first_timer_data = repeat_and_first_time(olist_data)
    regenerate_dataset_with_indicators(repeater_data, first_timer_data)
    create_user_ratings_df(repeater_data)

main()
