# Automated Newspaper Prediction from News Articles

An automated system that takes articles as input and predicts the names of the corresponding newspapers using various ML and DL based approaches.

## To setup the repository:

1) Clone the Github repository.

2) Go inside the repository folder, and setup the virtual environment.
    - pip install virtualenv
    - virtualenv venv
    - source virtualenv/Scripts/activate

3) Then install the requirements.
    - pip install -r requirements.txt

4) Go to this [Link](https://drive.google.com/drive/folders/1tzYDBq-MXYu7Bz9OmRywLFAjpcQGzl0N?usp=sharing) to download and copy files inside the folders to the corresponding folders in the Github repository. For example, Glove pre-trained word-embedding, pre-trained models, pre-processed data.

5) To COP_filt3_sub in the repository, copy all the COP files in the format of that is downloaded from the link that you provided, including COP25 file named as "COP25.filt3.sub" to test models.


## Running the models:

### To run the model using processed data by provided us:

#### Running classic_model.py:

classic_model.py takes the following arguments:

    -cf -> Create custom feature matrix and train the SVM model
    -ct -> Use custom test set to test model (COP25)
    -val -> Use val set to test model instead of processed test set
    -t -> Use the TF-IDF vectorizer instead of CountVectorizer
    -nb -> Use Naive Bayes for classification
    -rf -> Use Random Forest for classification
    -dt -> Use Decision Tree for classification
    -svm -> Use SVM for classification
    -knn -> Use K-nearest Neighbors for classification
    -en -> Use Ensemble for classification (combines Naive Bayes, Random Forest and SVM)
    -s -> Set the seed for model trainings (default 42)
    -svm_pretrained -> Use pretrained SVM for classification
    -o -> Output file to which we write predictions for test set
    

Example for running pretrained SVM using TF-IDF on processed test set with seed=36, and output results to a file: python classic_model.py -svm_pretrained -t - o -s 36

#### Running deep_learning.py:

deep_learning.py takes the following arguments:

    -ct -> Use custom test set to test model (COP25)
    -val -> Use val set to test model instead of processed test set
    -lstm -> Use the LSTM for classification
    -epoch -> Number of epochs to train
    -batch -> Batch size to train
    -bert_pretrained -> Use pretrained BERT for classification
    -lstm_pretrained -> Use pretrained LSTM for classification
    -s -> Set seed for model trainings (default 42)
    -o -> Output file to which we write predictions for test set
   
Example for running pretrained BERT on val set with epoch=100 and batch_size=32: python deep_learning.py -bert_pretrained -val -epoch 100 -batch 32

### To run the model using custom test set (COP25):

1) python read_custom_test.py
2) python custom_test_preprocessing.py

Then, you can run either classic_model.py or deep_learning.py by passing the -ct argument. See the ""Running classic_model.py" and "Running deep_learning.py" for other argumnets that you might want to pass.
