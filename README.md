# LFD-final-project

## To setup the repository:

1) Clone the github repository

2) Go inside the folder and setup the virtual environment
    - pip install virtualenv
    - virtualenv venv
    - source virtualenv/Scripts/activate

3) Then install the requirements
    - pip install -r requirements.txt

4) Go to the Drive folder that is shared and copy files inside the folders to the corresponding folders in the github repository
5) To COP_filt3_sub in the repository, copy all the COP files in the format of that is downloaded from the link that you provided, including COP25 to test the model


## Running the models:

### To run the model using processed data by provided us:

#### Running classic_model.py:

classic_model.py takes the following arguments:

    -cf -> Create custom feature matrix and train the SVM model
    -ct -> Use custom test set to test model (COP25)
    -val -> Use val set to test model instead of test set
    -t -> Use the TF-IDF vectorizer instead of CountVectorizer
    -nb -> Use Naive Bayes for classification
    -rf -> Use Random Forest for classification
    -dt -> Use Decision Tree for classification
    -svm -> Use SVM for classification
    -knn -> Use K-nearest Neighbors for classification
    -en -> Use Ensemble for classification (combines Naive Bayes, Random Forest and SVM)
    -s -> Set the seed for model trainings (default 42)
    -svm_pretrained -> Use pretrained SVM for classification
    -o -> Output a file to which we write predictions for test set
    

Example for running pretrained SVM using TF-IDF on custom test set (COP25) with seed=36, and output results to a file: python classic_model.py -svm_pretrained -t -ct - o -s 36 
