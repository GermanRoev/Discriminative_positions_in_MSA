from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import RFE
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Search for discriminative positions between subtypes in MSA of Hepatitis B virus S gene')
parser.add_argument('--input', type=str, help='Input file with MSA in fasta format')
parser.add_argument('--output', type=str, help='Output file')
args = parser.parse_args()

def nucl_to_vec(char):
    '''
    Transforms each nucleotide in alignment into a vector
    '''
    if char=='a':
        rez=[1.0, 0, 0, 0]
    elif char=='c':
        rez=[0, 1.0, 0, 0]
    elif char=='t':
        rez=[0, 0, 1.0, 0]
    elif char=='g':
        rez=[0, 0, 0, 1.0]
    elif char=='r':
        rez=[1/2, 0, 0, 1/2]
    elif char=='y':
        rez=[0, 1/2, 1/2, 0]
    elif char=='s':
        rez=[0, 1/2 , 0, 1/2]
    elif char=='w':
        rez=[1/2,0,1/2,0]
    elif char=='k':
        rez=[0, 0, 1/2, 1/2]
    elif char=='m':
        rez=[1/2, 1/2, 0, 0]
    elif char=='b':
        rez=[0, 1/3, 1/3, 1/3]
    elif char=='d':
        rez=[1/3, 0, 1/3, 1/3]
    elif char=='h':
        rez=[1/3, 1/3, 1/3, 0]
    elif char=='v':
        rez=[1/3, 1/3, 0, 1/3]
    elif char=='-':
        rez=[0, 0, 0, 0]
    elif char=='n':
        rez=[1/4, 1/4, 1/4, 1/4]
    else:
        raise Error
    return rez

with open(args.input) as f:
    file_msa=f.read().strip().split('>')[1:]
number_of_seqs=len(file_msa)  # Number of sequences in alignment
len_of_seqs=len(file_msa[0].strip().split('\n')[1])  # Length of alignment

X, y = [], []
for i in file_msa:
    sequence=i.strip().split('\n')[1]
    name=i.strip().split('\n')[0]
    vectorized_seq=np.array(list(map(nucl_to_vec, sequence)))  # Encode nucleotides to vectors
    X.append(vectorized_seq.reshape(1, len_of_seqs*4))  # Flattening
    label=str(name.split('-')[1])
    y.append(label)
X = np.array(X).reshape(number_of_seqs, len_of_seqs*4)  # Each row - sample, columns - encoded nucleotides
y = np.array(y)  # Labels of subtypes

train_size = 0.7
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=4)
print('Shape of X_train = ', X_train.shape)
print('Shape of X_test = ', X_test.shape)
print('Lenght of y_train = ', len(y_train))
print('Lenght of y_test = ', len(y_test))

forest=RandomForestClassifier(criterion='entropy', max_depth=5, 
                          max_features='sqrt', min_samples_leaf=3,
                          n_estimators=50, class_weight='balanced_subsample',
                           n_jobs=1, random_state=1
                          )
rfe=RFE(estimator=forest, n_features_to_select=10, step=0.005, verbose=0)  # Performing feature selection
rfe.fit(X_train,y_train)

important_features=[i for i,j in enumerate(rfe.support_) if j==True]  # Features (positions in MSA) that determine genotype
print('Raw positions of important features are = ', important_features)  # Raw positions don't correspond to real, because their amount is 4 times bigger
print('Important positions in alignment are = ', [i//4 + 1 for i in important_features])  # Real positions in MSA

X__train=[]  # Select only important positions from previous step
for i in range(X_train.shape[0]):
    d=[]
    for j in range(X_train.shape[1]):
        if j in important_features:
            d.append(X_train[i][j])
    X__train.append(d)
X__train=np.array(X__train)

X__test=[]
for i in range(X_test.shape[0]):
    d=[]
    for j in range(X_test.shape[1]):
        if j in important_features:
            d.append(X_test[i][j])
    X__test.append(d)
X__test=np.array(X__test)

forest_test=RandomForestClassifier(criterion='entropy', max_depth=5, 
                          max_features='sqrt', min_samples_leaf=3,
                          n_estimators=50, class_weight='balanced_subsample',
                           n_jobs=1, random_state=1
                          )
forest_test.fit(X__train,y_train) # Fit classifier only on important positions in MSA to test

preds = forest_test.predict(X__test)
print('\n', '-' * 80, '\n')
print('f1_weighted\t\t',f'{f1_score(y_test, preds, average="weighted")}')
print('accuracy\t\t',f'{accuracy_score(y_test, preds)}')
print('balanced_accuracy\t',f'{balanced_accuracy_score(y_test, preds)}')
print('recall_weighted\t\t',f'{recall_score(y_test, preds, average="weighted")}')
print('precision_weighted\t',f'{precision_score(y_test, preds, average="weighted")}')
print('confusion_matrix\t')
print(f'{confusion_matrix(y_test, preds)}')
print('\n', '-' * 80, '\n')


with open(args.output, 'w') as f:
    f.write('Raw positions of important features are = '+', '.join([str(x) for x in important_features])+'\n')
    f.write('Important positions in alignment are = '+ ', '.join([str(i//4 + 1) for i in important_features])+'\n')
    f.write('f1_weighted\t\t'+f'{f1_score(y_test, preds, average="weighted")}'+'\n')
    f.write('accuracy\t\t'+f'{accuracy_score(y_test, preds)}'+'\n')
    f.write('balanced_accuracy\t'+f'{balanced_accuracy_score(y_test, preds)}'+'\n')
    f.write('recall_weighted\t\t'+f'{recall_score(y_test, preds, average="weighted")}'+'\n')
    f.write('precision_weighted\t'+f'{precision_score(y_test, preds, average="weighted")}'+'\n')
    f.write('confusion_matrix\t'+'\n')
    f.write(f'{confusion_matrix(y_test, preds)}')
