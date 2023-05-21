# import libraries
import pandas as pd
from sqlalchemy import create_engine
import sys
import re

import nltk
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.pipeline import Pipeline 
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import pickle

def load_data(database_filepath):
    """ 
    load the cleaned dataframe from the database.

    Parameters:
       database_filepath:  path of database to read from

    Returns:
       X: Input features
       Y: Tagret variable
       category_names: 36 categories in the dataset. 
    """

    # load data from database
    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql_table("disaster_response", engine)
    
    X = df["message"]
    Y = df[df.columns[4:]]
    category_names = df.columns[4:]
    
    return X, Y, category_names
    
def tokenize(text):
    """
    Perform a set of text processing steps to the text
    
    Parameters:
       text: row message
      
    Returns:
       text: processed message
    """
    # Remove links
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls =re.findall(url_regex, text)
    for url in urls:
        text = text.replace(url, "urlplaceholder")
        
    # Normalization:
    # 1- Capitalization Removal
    text = text.lower()
    # 2- Punctation Removal
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) 
    
    # Tokenization
    text = word_tokenize(text)
    
    # Stop Words Removal
    text = [word for word in text if word not in stopwords.words("english")]
    
    # Lemmatization
    text = [WordNetLemmatizer().lemmatize(w) for w in text]
    
    # Stemming 
    text = [PorterStemmer().stem(word) for word in text] 
    
    return text

def build_model():
    """
     build sklearn model
     
     Returns:
       model: ML pipeline model that sequentially apply a list of transforms and a final estimator.
 
     """

    # Instantiate transformers :
    # 1- Convert a collection of text documents to a matrix of token counts.
    vect = CountVectorizer(tokenizer=tokenize) # Override the tokenization step
    # 2- Transform a count matrix to a normalized tf-idf representation.
    tfidf = TfidfTransformer()

    # Instantiate classifier
    clf = MultiOutputClassifier(estimator = RandomForestClassifier()) 

    # Create a pipeline
    pipeline = Pipeline([('vect', vect ), ('tfidf', TfidfTransformer()), ('clf', clf)])

    # Use grid search to find better parameters.
    parameters = {
            'clf__estimator__n_estimators': [50, 100, 200], 
            'clf__estimator__min_samples_split': [2, 3, 4]
        }

    model = GridSearchCV(pipeline, param_grid=parameters)
        
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model and report the f1 score, precision and recall 

    Parameters: 
       model: a sklearn pipeline model
       X_test: messages
       Y_test: messages' categories
       category_names: list of category types 
    """
    Y_predicted = model.predict(X_test[:10])
    report = classification_report(Y_test[:10], Y_predicted, target_names=category_names)
    print(report)

def save_model(model, model_filepath):
    """
    save the model in the specified path

    Parameters:
        model: the skearn pipeline model to be saved
        model_filepath: path to save the model in 
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)
        
        
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train[:50], Y_train[:50])
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()