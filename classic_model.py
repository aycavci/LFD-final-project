from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
import spacy
import argparse
import scipy.sparse as sp
import pickle
import warnings

warnings.filterwarnings("ignore")

# Loading Spacy Library
nlp = spacy.load("en_core_web_sm")

def create_arg_parser():
    parser = argparse.ArgumentParser()
    # -cr flag to run the algorithm with custom features
    parser.add_argument("-cf", "--create_custom_features", action="store_true",
                        help="Create custom feature matrix and train the SVM model")
    
    parser.add_argument("-ct", "--custom_test_set", action="store_true",
                        help="Use custom test set to test model")

    parser.add_argument("-val", "--val_set", action="store_true",
                        help="Use val set to test model instead of processed test set")

    parser.add_argument("-t", "--tfidf", action="store_true",
                        help="Use the TF-IDF vectorizer instead of CountVectorizer")

    parser.add_argument("-nb", "--naive_bayes", action="store_true",
                        help="Use Naive Bayes for classification")

    parser.add_argument("-rf", "--random_forest", action="store_true",
                        help="Use Random Forest for classification")

    parser.add_argument("-dt", "--decision_tree", action="store_true",
                        help="Use Decision Tree for classification")

    parser.add_argument("-svm", "--svm", action="store_true",
                        help="Use SVM for classification")

    parser.add_argument("-knn", "--knn", action="store_true",
                        help="Use K-nearest Neighbors for classification")

    parser.add_argument("-en", "--ensemble", action="store_true",
                        help="Use Ensemble for classification")

    parser.add_argument("-s", "--seed", default=42, type=int,
                        help="Set the seed for model trainings (default 42)")

    parser.add_argument("-svm_pretrained", "--svm_pretrained", action="store_true",
                        help="Use pretrained SVM for classification")
                        
    parser.add_argument("-o", "--output_file", type=str,
                    help="Output file to which we write predictions for test set")

    args = parser.parse_args()
    return args


def identity(x):
    '''Dummy function that just returns the input'''
    return x


def write_to_file(labels, output_file):
    '''Write list to file'''
    with open(output_file, "w") as out_f:
        for line in labels:
            out_f.write(line.strip() + '\n')
    out_f.close()


def tokenizer(body):
    '''Perform tokenization'''
    doc = nlp(body)
    tokens = [word.text for word in doc]
    return tokens


def get_score(classifier, X_test, Y_test, output_file):
    # Given a trained model, predict the label of a new set of data.
    Y_pred = classifier.predict(X_test)
    # Calculate the accuracy score of the trained model by comparing predicted labels with actual labels.
    acc = accuracy_score(Y_test, Y_pred)
    if output_file:
        write_to_file(Y_pred, output_file)
    print(classification_report(Y_test, Y_pred))

    return acc


def base_model(vec, X_train, Y_train):
    '''Train Naive Bayes Model'''
    print("Navie Bayes Classification")
    model = MultinomialNB()

    classifier = Pipeline([('vec', vec), ('cls', model)])
    classifier.fit(X_train, Y_train)

    return classifier


def optimize_rf(vec, X_train, Y_train, seed):
    '''Train Random Forest Model'''
    print("Random Forest Classification")
    model = RandomForestClassifier(criterion='gini', n_estimators=233, max_depth=10, max_features=0.064,
                                   random_state=seed)

    classifier = Pipeline([('vec', vec), ('cls', model)])
    classifier.fit(X_train, Y_train)

    return classifier


def optimize_knn(vec, X_train, Y_train):
    '''Train KNN Model'''
    print("KNN Classification")
    model = KNeighborsClassifier(n_neighbors=118, weights='uniform', n_jobs=-1)

    classifier = Pipeline([('vec', vec), ('cls', model)])
    classifier.fit(X_train, Y_train)

    return classifier


def optimize_dt(vec, X_train, Y_train, seed):
    '''Train Decision Tree Model'''
    print("Decision Tree Classification")
    model = DecisionTreeClassifier(
        splitter='best',
        max_depth=14,
        max_features=0.81,
        criterion='entropy',
        random_state=seed
    )

    classifier = Pipeline([('vec', vec), ('cls', model)])
    classifier.fit(X_train, Y_train)

    return classifier


def optimize_svm(vec, X_train, Y_train, seed):
    '''Train SVM Model'''
    print("SVM classification")
    if vec is None:
        svm_ = svm.SVC(kernel='linear', C=1.14, random_state=seed)
    else:
        svm_ = Pipeline([('vec', vec), ('cls', svm.SVC(kernel='linear', C=1.14, random_state=seed))])

    svm_.fit(X_train, Y_train)

    return svm_


def custom_feature(row):
    '''Use custom feature'''
    dic = {}
    dic['org_count'] = row['org_count']
    dic['sentence_count'] = row['sentence_count']
    dic['gpe_count'] = row['gpe_count']
    return dic


def ensemble(vec, X_train, Y_train, seed):
    '''Train Ensemble Model'''
    print("Ensemble of Naive Bayes, Random Forest and SVM")

    nb = Pipeline([('vec_cn', vec), ('cls', MultinomialNB())])
    rf = Pipeline([('vec_tf', vec), ('cls', RandomForestClassifier(criterion='gini', n_estimators=233, max_depth=10,
                                                                   max_features=0.064, n_jobs=-1, random_state=seed))])
    svm_ = Pipeline([('vec_tf', vec), ('cls', svm.SVC(kernel='linear', C=1.14, random_state=seed))])

    estimators = [('nb', nb), ('rf', rf), ('svm', svm_)]

    ensemble_classifier = VotingClassifier(estimators, voting='hard')
    classifier = ensemble_classifier.fit(X_train, Y_train)

    return classifier


if __name__ == "__main__":
    args = create_arg_parser()

    """ Below code refactored for the format python LFD_assignment2.py -i <trainset> -ts <testset>.
        Normally, it is used with split_data function to experiment with different classifiers. """
    
    # Read from CSV
    train = pd.read_csv('./processed_data/processed_train.csv')

    if args.custom_test_set:
        test = pd.read_csv('./processed_data/processed_custom_test.csv')
    else:
        if args.val_set:
            test = pd.read_csv('./processed_data/processed_val.csv')
        else:
            test = pd.read_csv('./processed_data/processed_test.csv')

    X_train, Y_train = train['clean'], train['newspaper_name']

    X_test, Y_test = test['clean'], test['newspaper_name']

    
    if args.create_custom_features:
        # Create custom features dictionary
        train_dic = [custom_feature(row) for index, row in train.iterrows()]
        test_dic = [custom_feature(row) for index, row in test.iterrows()]

        dic_train_matr = DictVectorizer().fit_transform(train_dic)
        dic_test_matr = DictVectorizer().fit_transform(test_dic)

        # Applying TF-IDF on text
        vec = TfidfVectorizer().fit(train['clean'])

        train_word_matr = vec.transform(train['clean'])
        test_word_matr = vec.transform(test['clean'])

        train_matr = sp.hstack((train_word_matr, dic_train_matr), format='csr')
        test_matr = sp.hstack((test_word_matr, dic_test_matr), format='csr')

        classifier = optimize_svm(None, train_matr, Y_train, args.seed)

        acc = get_score(classifier, test_matr, Y_test, args.output_file)
        print("\n Accuracy: {}".format(acc))

    else:
        if args.tfidf:
            vec = TfidfVectorizer(tokenizer=tokenizer, preprocessor=identity)
        else:
            """ Bag of Words vectorizer """
            vec = CountVectorizer(tokenizer=tokenizer, ngram_range=(1, 1), min_df=3)
        if args.naive_bayes:
            classifier = base_model(vec, X_train, Y_train)
        elif args.random_forest:
            classifier = optimize_rf(vec, X_train, Y_train, args.seed)
        elif args.decision_tree:
            classifier = optimize_dt(vec, X_train, Y_train, args.seed)
        elif args.knn:
            classifier = optimize_knn(vec, X_train, Y_train)
        elif args.ensemble:
            classifier = ensemble(vec, X_train, Y_train, args.seed)
        else:
            if args.svm_pretrained:
                with open('./model/svm.pkl', 'rb') as file:
                    classifier = pickle.load(file)
            else:
                classifier = optimize_svm(vec, X_train, Y_train, args.seed)

                filename = "./model/svm.pkl"
                with open(filename, 'wb') as file:
                    pickle.dump(classifier, file)

        acc = get_score(classifier, X_test, Y_test, args.output_file)
        print("\n Accuracy: {}".format(acc))
