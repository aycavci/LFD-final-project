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

nlp = spacy.load("en_core_web_sm")
import argparse
import scipy.sparse as sp
import pickle
import warnings

warnings.filterwarnings("ignore")


def create_arg_parser():
    parser = argparse.ArgumentParser()
    # -cr flag to run the algorithm with custom features
    parser.add_argument("-cf", "--create_custom_features", action="store_true",
                        help="Create custom feature matrix and train the svm model")

    parser.add_argument("-t", "--tfidf", action="store_true",
                        help="Use the TF-IDF vectorizer instead of CountVectorizer")

    parser.add_argument("-nb", "--naive_bayes", action="store_true",
                        help="Use Naive Bayes For Classification")

    parser.add_argument("-rf", "--random_forest", action="store_true",
                        help="Use Random Forest For Classification")

    parser.add_argument("-dt", "--decision_tree", action="store_true",
                        help="Use Decision Tree For Classification")

    parser.add_argument("-svm", "--svm", action="store_true",
                        help="Use Naive Bayes For Classification")

    parser.add_argument("-knn", "--knn", action="store_true",
                        help="Use Naive Bayes For Classification")

    parser.add_argument("-en", "--ensemble", action="store_true",
                        help="Use Ensemble For Classification")

    parser.add_argument("-s", "--seed", default=42, type=int,
                        help="Seed for model trainings (default 42)")

    parser.add_argument("-svm_pretrained", "--svm_pretrained", action="store_true",
                        help="Use pretrained SVM for classification")

    args = parser.parse_args()
    return args


def identity(x):
    """Dummy function that just returns the input"""
    return x


def tokenizer(body):
    doc = nlp(body)
    tokens = [word.text for word in doc]
    return tokens


def get_score(classifier, X_val, Y_val):
    # Given a trained model, predict the label of a new set of data.
    Y_pred = classifier.predict(X_val)
    # Calculates the accuracy score of the trained model by comparing predicted labels with actual labels.
    acc = accuracy_score(Y_val, Y_pred)
    print(classification_report(Y_val, Y_pred))

    return acc


def base_model(vec, X_train, Y_train):
    print("Navie Bayes Classification")
    model = MultinomialNB()

    classifier = Pipeline([('vec', vec), ('cls', model)])
    classifier.fit(X_train, Y_train)

    return classifier


def optimize_rf(vec, X_train, Y_train, seed):
    print("Random Forest Classification")
    model = RandomForestClassifier(criterion='gini', n_estimators=233, max_depth=10, max_features=0.064,
                                   random_state=seed)

    classifier = Pipeline([('vec', vec), ('cls', model)])
    classifier.fit(X_train, Y_train)

    return classifier


def optimize_knn(vec, X_train, Y_train):
    print("KNN Classification")
    model = KNeighborsClassifier(n_neighbors=118, weights='uniform', n_jobs=-1)

    classifier = Pipeline([('vec', vec), ('cls', model)])
    classifier.fit(X_train, Y_train)

    return classifier


def optimize_dt(vec, X_train, Y_train, seed):
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
    print("SVM classification")
    if vec is None:
        svm_ = svm.SVC(kernel='linear', C=1.14, random_state=seed)
    else:
        svm_ = Pipeline([('vec', vec), ('cls', svm.SVC(kernel='linear', C=1.14, random_state=seed))])

    svm_.fit(X_train, Y_train)

    return svm_


def custom_feature(row):
    dic = {}
    dic['org_count'] = row['org_count']
    dic['sentence_count'] = row['sentence_count']
    dic['gpe_count'] = row['gpe_count']
    return dic


def ensemble(vec, X_train, Y_train, seed):
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

    train = pd.read_csv('./processed_data/processed_train.csv', nrows=100)
    val = pd.read_csv('./processed_data/processed_val.csv', nrows=10)
    # test = pd.read_csv('./processed_data/processed_test.csv')
    test = pd.read_csv('./processed_data/processed_custom_test.csv', nrows=10)

    X_train, Y_train = train['clean'], train['newspaper_name']

    X_val, Y_val = val['clean'], val['newspaper_name']

    if args.create_custom_features:
        # Create custom features dictionary
        train_dic = [custom_feature(row) for index, row in train.iterrows()]
        val_dic = [custom_feature(row) for index, row in val.iterrows()]
        test_dic = [custom_feature(row) for index, row in test.iterrows()]

        dic_train_matr = DictVectorizer().fit_transform(train_dic)
        dic_val_matr = DictVectorizer().fit_transform(val_dic)
        dic_test_matr = DictVectorizer().fit_transform(test_dic)

        # Applying Bag-of-words on text
        vec = TfidfVectorizer().fit(train['clean'])

        train_word_matr = vec.transform(train['clean'])
        val_word_matr = vec.transform(val['clean'])
        test_word_matr = vec.transform(test['clean'])

        train_matr = sp.hstack((train_word_matr, dic_train_matr), format='csr')
        val_matr = sp.hstack((val_word_matr, dic_val_matr), format='csr')
        test_matr = sp.hstack((test_word_matr, dic_test_matr), format='csr')

        classifier = optimize_svm(None, train_matr, Y_train, args.seed)

        filename = "./model/svm.pkl"
        with open(filename, 'wb') as file:
            pickle.dump(classifier, file)

        acc = get_score(classifier, val_matr, Y_val)
        print("\n Accuracy: {}".format(acc))

    else:
        if args.tfidf:
            vec = TfidfVectorizer(tokenizer=tokenizer, preprocess=identity)
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
                    model = pickle.load(file)

                acc = get_score(model, X_val, Y_val)
                print("\n Accuracy: {}".format(acc))

            else:
                classifier = optimize_svm(vec, X_train, Y_train, args.seed)

                filename = "./model/svm.pkl"
                with open(filename, 'wb') as file:
                    pickle.dump(classifier, file)

                acc = get_score(classifier, X_val, Y_val)
                print("\n Accuracy: {}".format(acc))
