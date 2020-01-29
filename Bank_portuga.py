import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import pickle

sns.set()


def caching_dataframe():
    """
    Function to cache csv data in pickle to quick
    load and replacing unknown value with NAN
    :return: None
    """
    data = pd.read_excel('Book1.xlsx')
    # Converting Unknown to None
    data.replace('unknown', math.nan, inplace=True)
    data.to_pickle(r'cached_data.pkl')


# caching_dataframe()
pd.set_option('display.max_columns', 46)
df = pd.read_pickle('cached_data.pkl')


# Build a function to show categorical values distribution


def plot_bar(column):
    # temp df
    temp_1 = pd.DataFrame()
    # count categorical values
    temp_1['No_deposit'] = df[df['y'] == 'no'][column].value_counts()
    temp_1['Yes_deposit'] = df[df['y'] == 'yes'][column].value_counts()
    temp_1.plot(kind='bar')
    plt.xlabel(f'{column}')
    plt.ylabel('Number of clients')
    plt.title('Distribution of {} and deposit'.format(column))


plot_bar('job'), plot_bar('marital'),
plot_bar('education'), plot_bar('contact'),
plot_bar('loan'), plot_bar('housing')


def cleaning_data(dataframe):
    """
    To clean data of NAN values
    :param dataframe: data input which
    require to be cleaned
    :return: cleaned dataframe
    """
    dataframe['default'].replace(math.nan, 'no', inplace=True)
    dataframe['pdays'].replace(999, 0, inplace=True)
    dataframe['education'].replace(math.nan, 'illiterate', inplace=True)
    dataframe.dropna(inplace=True)
    return dataframe


df = cleaning_data(df)


def features_manipulation(data):
    """
    creating extra features to how accurate prediction
    :return: dataset with extra feature
    """
    # Binning age min age is 17 and max is 98
    data['age_bin'] = (data['age'] // 10) * 10
    # Binning duration
    data['duration'] = (data['duration'] // 100)
    # Replacing values with binary ()
    df.y = df.y.map({'no': 0, 'yes': 1}).astype('uint8')
    data.contact = data.contact.map({'cellular': 1, 'telephone': 0}).astype('uint8')
    data.loan = data.loan.map({'yes': 1, 'no': 0}).astype('uint8')
    data.housing = data.housing.map({'yes': 1, 'no': 0}).astype('uint8')
    data.default = data.default.map({'no': 1, 'yes': 0}).astype('uint8')
    data.poutcome = data.poutcome.map({'nonexistent': 0, 'failure': 0, 'success': 1}).astype('uint8')
    # change the range of Var Rate
    data['emp.var.rate'] = data['emp.var.rate'].apply(lambda x: x * -0.0001 if x > 0 else x * 1)
    data['emp.var.rate'] = data['emp.var.rate'] * -1
    data['emp.var.rate'] = data['emp.var.rate'].apply(lambda x: -np.log(x) if x < 1 else np.log(x)).astype('uint8')
    # Multiply consumer index
    data['cons.price.idx'] = (data['cons.price.idx'] * 10).astype('uint8')
    # change the sign (we want all be positive values)
    data['cons.conf.idx'] = data['cons.conf.idx'] * -1
    data.previous = data.previous.apply(lambda x: 1 if x > 0 else 0).astype('uint8')
    # re-scale variables
    data['nr.employed'] = np.log2(data['nr.employed']).astype('uint8')
    data['cons.price.idx'] = np.log2(data['cons.price.idx']).astype('uint8')
    data['cons.conf.idx'] = np.log2(data['cons.conf.idx']).astype('uint8')
    data.age = np.log(data.age)
    # less space
    data.euribor3m = data.euribor3m.astype('uint8')
    data.campaign = data.campaign.astype('uint8')
    data.pdays = data.pdays.astype('uint8')
    return data


features_manipulation(df)


# function to One Hot Encoding
def encode(data, col):
    return pd.concat([data, pd.get_dummies(col, prefix=col.name)], axis=1)


# One Hot encoding of 3 variable
df = encode(df, df.job)
df = encode(df, df.month)
df = encode(df, df.day_of_week)

# Drop transformed features
df.drop(['job', 'month', 'day_of_week'], axis=1, inplace=True)

# converting education and marital, education, one hot
y = df['y'].copy()
enc = ce.OneHotEncoder(cols=['marital', 'education'], drop_invariant=True)
df_num = enc.fit_transform(df, y)
corr = df_num.corr()
print(corr['y'].sort_values(ascending=False))
df_num.drop('y', axis=1, inplace=True)

print(df_num.info())

# Decreasing number of features using PCA
pca = PCA(n_components=0.95)
# Standard scaling
sc = StandardScaler()
# pipeline
# Fit pipelines
pipe = Pipeline(steps=[('pca', pca), ('sc', sc)])
pipe.fit_transform(df_num, y)
print(df.shape, y.shape)
# splitting train and test data
x_train, x_test, y_train, y_test = train_test_split(df_num, y, test_size=0.3,
                                                    random_state=42)

# Initializing SVC
svc = SVC(random_state=42, probability=True)
# Initializing Cv
cv = StratifiedKFold(shuffle=True, n_splits=5, random_state=42)
print(cv)


def train_best_model(classifier, cross_val, xTrain, xTest, yTrain, yTest):
    """

    :param classifier: classifier to train
    :param cross_val: cross validation value
    :param xTrain: training labesls
    :param xTest: training target
    :param yTrain: testing labels
    :param yTest: testing target
    :return: best model saved in model.pkl
    """
    # Grid search implementation
    grid_param = {'C': [140, 141, 142]}
    grid_search = GridSearchCV(classifier, grid_param, scoring='accuracy', cv=cross_val, n_jobs=-1)
    grid_search.fit(xTrain, yTrain)
    # Printing best estimator, param and best score
    print('Best param', grid_search.best_params_)
    print('Best score', grid_search.best_score_)

    # Training best model
    model = grid_search.best_estimator_
    model.fit(xTrain, yTrain)
    predict = model.predict(xTest)
    prediction_proba = model.predict_proba(xTest)[:, 1]
    print('Accuracy', accuracy_score(yTest, predict))
    print('ROC_AUC score', roc_auc_score(yTest, prediction_proba))
    # Saving best model
    pickle.dump(model, open('model.pkl', 'wb'))


# Calling train model
# train_best_model(svc, cv, x_train, x_test, y_train, y_test)
# Loading from model
pkl_model = pickle.load(open('model.pkl', 'rb'))
prediction = pkl_model.predict(x_test)
predict_proba = pkl_model.predict_proba(x_test)[:, 1]
print('Accuracy', accuracy_score(y_test, prediction))
auc_roc = roc_auc_score(y_test, predict_proba)
print('ROC_AUC score', auc_roc)

''' Build graph for ROC_AUC '''

fpr, tpr, threshold = roc_curve(y_test, pkl_model.predict_proba(x_test)[:, 1])


def plot_roc_curve(fpr1, tpr1):
    plt.figure()
    sns.lineplot(fpr, tpr, color='orange', label='ROC')
    sns.lineplot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()


plot_roc_curve(fpr, tpr)
plt.show()
