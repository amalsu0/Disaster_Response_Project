# import libraries
import pandas as pd
from sqlalchemy import create_engine
import sys


def load_data(messages_filepath, categories_filepath):
    """
    reads in data from two CSV files located at filepaths 
    then merges them into a single dataframe.

    Parameters:
        messages_filepath: filepath for the messages CSV file.
        categories_filepath: filepath for the categories CSV file.

    Returns:
        df: A pandas dataframe that contains the merged data from both files.
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, on = "id")
    return df
    
def clean_data(df):
    """
    performs data cleaning operations on it.

    Parameters:
        df: A pandas DataFrame that contains the data to be cleaned.

    Returns:
        out_df: cleaned pandas DataFrame.
    """
    # create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = [i.split("-")[0] for i in row.tolist()]
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.split("-")[-1])
        
        # convert column from string to numeric
        categories[column] = categories[column].apply(lambda x: int(x))

    # drop the original categories column from `df`
    df.drop(columns=["categories"], inplace = True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    out_df = pd.concat([df,categories], axis=1)
    
    # clean related column which has value (2) that is out of distribution 
    sub = out_df.loc[out_df["related"] == 2]
    out_df.drop(sub.index, inplace = True)
    out_df.reset_index(inplace=True, drop =True)

    # drop duplicates
    out_df.drop_duplicates(inplace=True)
    
    return out_df
    
def save_data(df, database_filename):
    """
    save the clean dataset into a SQLite database.

    Parameters:
        df: cleaned DataFrame to be saved.
        database_filename:  path to create engine
    """
    engine = create_engine(f"sqlite:///{database_filename}")
    df.to_sql("disaster_response" , engine, index=False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()