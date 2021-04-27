import pandas as pd
import numpy as np
import re
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype,is_float_dtype
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from pandas.tseries.offsets import MonthEnd
from sklearn.pipeline import make_pipeline, make_union
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import pylab
from bayes_opt import BayesianOptimization
import xgboost as xgb
import seaborn as sns
from matplotlib import rcParams



def display_all(df):
    """
    Docstring: make dataframe display all rows/columns (within 1000).
    
    Parameters
    ----------
    df: pandas dataframe to be display.
    """

    with pd.option_context('display.max_rows',1000,'display.max_columns',1000):
        display(df)
        
        
def addValue_withoutHue(plot, feature):
    """
    Docstring: display the number of counts and percentage
    
    Parameters
    ----------
    plot: the figure instance
    feature: which feature to plot
    """

    total = len(feature)
    for p in plot.patches:
        cnt = p.get_height()
        percentage = '({:.1f}%)'.format(100 * p.get_height()/total)
        x_cnt = p.get_x() + p.get_width() / 2 - 0.1
        x_pct = p.get_x() + p.get_width() / 2 + 0.05
        y = p.get_y() + p.get_height()+1
        plot.annotate(cnt, (x_cnt, y), size = 12)
        plot.annotate(percentage, (x_pct, y), size = 12)
    plt.show()
    
def plot_missing_values(df):
    """
    Docstring: For each column with missing values plot proportion that is missing.
    
    Parameters
    ----------
    df: target dataframe
    """
    data = [(col, df[col].isnull().sum() / len(df)) 
            for col in df.columns if df[col].isnull().sum() > 0]
    col_names = ['column', 'percent_missing']
    missing_df = pd.DataFrame(data, columns=col_names).sort_values('percent_missing')
    pylab.rcParams['figure.figsize'] = (15, 8)
    missing_df.plot(kind='barh', x='column', y='percent_missing'); 
    plt.title('Percent of missing values in colummns');
    
def fraud_rate(df,col,col_name):
    """
    Docstring: count fraud rate by different categories
    
    Parameters
    ----------
    df: pandas dataframe
    col: categories from which column
    col_name: fraud rate column name
    
    Return
    ----------
    the fraud rate of each categories from the "col" field
    """
    return df.groupby(col).apply(lambda x:pd.Series({col_name:x.isFraud.mean()}))


def ttest(df,ind_col,target,equal_var=True):
    """
    Docstring: Calculate the T-test for the means of two independent samples of scores
    
    Parameters
    ----------
    df: pandas dataframe
    ind_col: the column that used to separate two independent samples with value True and False
    target: the column that used to count mean
    equal_var: if the two group have equal variance, default is True
    
    Return
    ----------
    Will print t statistic, pvalue and significant test
    """
    test = ttest_ind(df.loc[df[ind_col]==True,target],df.loc[df[ind_col]==False,target],equal_var=equal_var)
    print("t statistic is {} with p value {}".format(test.statistic,test.pvalue))
    print("---------------------------------")
    if_sig = "significant" if test.pvalue<0.05 else "not significant"
    print("The difference is",if_sig)
    
def to_part_of_day(x):
    if x < 5:
        return "midnight"
    elif x < 12:
        return "morning"
    elif x<18:
        return "afternoon"
    else:
        return "night"
    
def plot_ROC(y_pred, y_train,y_pred_val, y_val, y_pred_te, y_test):
    """
    Docstring: plot the roc curve
    """  

    fpr_0, tpr_0, thresholds_0 = roc_curve(y_train, y_pred)
    fpr_1, tpr_1, thresholds_1 = roc_curve(y_val, y_pred_val)
    fpr_2, tpr_2, thresholds_2 = roc_curve(y_test, y_pred_te)
    roc_auc_0 = auc(fpr_0, tpr_0)
    roc_auc_1 = auc(fpr_1, tpr_1)
    roc_auc_2 = auc(fpr_2, tpr_2)
    print("Area under the ROC curve - train: %f" % roc_auc_0)
    print("Area under the ROC curve - validation: %f" % roc_auc_1)
    print("Area under the ROC curve - test: %f" % roc_auc_2)
    # Plot ROC curve
    plt.figure(figsize=(8,8))
    plt.plot(fpr_0, tpr_0, label='ROC curve - train(AUC = %0.2f)' % roc_auc_0, color='g')
    plt.plot(fpr_1, tpr_1, label='ROC curve - validation (AUC = %0.2f)' % roc_auc_1, color='b')
    plt.plot(fpr_2, tpr_2, label='ROC curve - test (AUC = %0.2f)' % roc_auc_2, color='r')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for lead score model')
    plt.legend(loc="lower right")
    plt.show()
    
    
def plot_metrics(model, X_tr, y_train,X_val,y_val, X_te, y_test):
    """
    Docstring: plot the recall-precision report, roc curve and confusion matrix for train and test data
    """
    y_pred = model.predict(X_tr)#.argmax(axis=1)
    y_pred_val = model.predict(X_val)#.argmax(axis=1)
    y_pred_te = model.predict(X_te)#.argmax(axis=1)
    y_pred = [1 if y>=0.5 else 0 for y in y_pred]   
    y_pred_val = [1 if y>=0.5 else 0 for y in y_pred_val]
    y_pred_te = [1 if y>=0.5 else 0 for y in y_pred_te]
    
    confusion1 = metrics.confusion_matrix(y_train, y_pred)
    confusion3 = metrics.confusion_matrix(y_test, y_pred_te)
    confusion2 = metrics.confusion_matrix(y_val, y_pred_val)
    print("   -----------------------------")
    print("   classification report TRAIN")    
    print("   -----------------------------")
    print(metrics.classification_report(y_train,y_pred))
    print("   -----------------------------")
    print("   classification report VALID")   
    print("   -----------------------------")
    print(metrics.classification_report(y_val,y_pred_val))
    print("   -----------------------------")
    print("   classification report TEST")   
    print("   -----------------------------")
    print(metrics.classification_report(y_test,y_pred_te))
    print("\n\n")
    print("   -----------------------------")
    print("    ROC")
    print("   -----------------------------")
#     plt.title('credit card model ROC')
#     plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
#     plt.legend(loc = 'lower right')
#     plt.plot([0, 1], [0, 1],'r--')
#     plt.xlim([0, 1])
#     plt.ylim([0, 1])
#     plt.ylabel('True Positive Rate')
#     plt.xlabel('False Positive Rate')
#     plt.show()
    plot_ROC(y_pred, y_train,y_pred_val, y_val, y_pred_te, y_test)
    print("\n\n")
    print("   -----------------------------")
    print("    confusion matrix TRAIN")
    print("   -----------------------------")
    sns.heatmap(confusion1, annot=True, fmt='d', cmap='Blues')
    plt.show()
    print("   -----------------------------")
    print("    confusion matrix VALID")
    print("   -----------------------------")
    sns.heatmap(confusion2, annot=True, fmt='d', cmap='Blues')
    plt.show()
    print("   -----------------------------")
    print("    confusion matrix TEST")
    print("   -----------------------------")
    sns.heatmap(confusion3, annot=True, fmt='d', cmap='Blues')  

    
    
def balanceSampling(X_tr, y_train, up_ratio=1,dn_ratio=1):
    """
    Docstring: up and under sampling data
    
    Parameters
    ----------
    up_ratio: upsampling ratio
    dn_ratio: downsampling ratio

    """
    # Ratio argument is the percentage of the upsampled minority class in relation to the majority class. Default is 1.0
    over = SMOTE(sampling_strategy = up_ratio)
    under = RandomUnderSampler(sampling_strategy = dn_ratio)
    steps = [('over', over), ('under', under)]
    pipeline = Pipeline(steps=steps)
    X_train_sm, y_train_sm = pipeline.fit_resample(X_tr, y_train)
    
    print(X_train_sm.shape, y_train_sm.shape)
    return X_train_sm, y_train_sm
    
    
def metric(self,prob,cl):
    self.fpr, self.tpr, threshold = roc_curve(self.y_valid, prob)
    auc = roc_auc_score(self.y_valid,prob)
    self.plot_roc_curve(self.fpr,self.tpr)
    acc = (self.y_valid==cl).mean()
    f1 = f1_score(self.y_valid,cl)
    print(acc,f1,auc)    
    

"""""""""""""""""""""""""""""""""""""""""
XGBOOST + BAYESIAN OPTIMIZATION

"""""""""""""""""""""""""""""""""""""""""
    
def xgbCv(dtrain,dvalid, eta, gamma, maxDepth, minChildWeight, subsample, colSample):
    # prepare xgb parameters 
    params = {
        "objective": "binary:logistic",
        "booster" : "gbtree",
        "eval_metric": "auc",
        "tree_method": 'auto',
        "silent": 1,
        "eta": eta,
        "max_depth": int(maxDepth),
        "min_child_weight" : minChildWeight,
        "subsample": subsample,
        "colsample_bytree": colSample,
        "gamma": gamma
    }
    
    #dtrain = xgb.DMatrix(X_train, y_train) 
    #dvalid = xgb.DMatrix(X_valid, y_valid)
    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
   # print("numr",numRounds)
    gbm = xgb.train(params, dtrain, 200, evals = watchlist, early_stopping_rounds = 100)
    score = gbm.best_score
    
    # return the best score
    return -1.0*score # invert the cv score to let bayopt maximize
    
def bayesOpt(dtrain,dvalid):
    ranges = {
        #'numRounds': (100, 500),
        'eta': (0.001, 0.3),
        'gamma': (0, 25),
        'maxDepth': (1, 10),
        'minChildWeight': (0, 10),
        'subsample': (0, 1),
        'colSample': (0, 1)
    }
    # proxy through a lambda to be able to pass train and features
    optFunc = lambda eta, gamma, maxDepth, minChildWeight, subsample, colSample: xgbCv(dtrain,dvalid, eta, gamma, maxDepth, minChildWeight, subsample, colSample)
    bo = BayesianOptimization(optFunc, ranges)
    bo.maximize(init_points = 2, n_iter = 1, kappa = 2, acq = "ei", xi = 0.0)
    
    #bestAUC = round((-1.0 * bo.res['max']['max_val']), 6)
    #print("\n Best AUC found: %f" % bestAUC)
    #print("\n Parameters: %s" % bo.res['max']['max_params'])
    return bo
    
    
    
    
"""""""""""""""""""""""""""""""""""""""""
sklearn custom fucntions and transformers

"""""""""""""""""""""""""""""""""""""""""
class FeatureSelector(BaseEstimator, TransformerMixin):
    #Class Constructor 
    def __init__( self,feature_names ):
        self._feature_names = feature_names 
    
    #Return self nothing else to do here    
    def fit( self, X,y = None ):
        return self 
    
    #Method that describes what we need this transformer to do
    def transform( self, X,y = None ):
        return X[self._feature_names] 
    
    def get_feature_names(self):
        return self._feature_names
# #Numerical features to pass down the numerical pipeline 
# numerical_features = ['availableMoney', 'transactionAmount', 'currentBalance','posOnPremises','cardPresent']


class NumericalTransformer(BaseEstimator, TransformerMixin):
    #Class Constructor
    def __init__( self):
        #pass
        self.features = []
        
    #Return self, nothing else to do here
    def fit( self, X, y = None ):
        return self 
    
    #Custom transform method we wrote that creates aformentioned features and drops redundant ones 
    def transform(self, X, y = None):
        #Check if needed 
        df = X.copy()
        #create new column
        df.loc[:,"blnc_avl_rt"] = df['currentBalance']/df['availableMoney']
        #drop redundant column
        df.drop('currentBalance',axis=1)
        
        #identify if the current tranasction amount is using up more than 80% of the available money
        df.loc[:,"usingUp_avl"] = np.where((df['availableMoney']<0) | (df['transactionAmount']/df['availableMoney'] >0.8),1,0)
        
                
        #Converting any infinity values in the dataset to Nan
        df = df.replace( [ np.inf, -np.inf ], np.nan )
        
        self.features += df.columns.tolist()
        #print(self.features)
        return df
    
    def get_feature_names(self):
        return self.features

class DropSomeColumns(BaseEstimator, TransformerMixin):
    """
    Docstring: remove specified columns
    
    Parameters
    ----------
    cols: columns to be removed
    
    Return
    ----------
    trans: new dataset after dropping "cols"
    """
    def __init__(self, cols):
        self.features = []
        if not isinstance(cols, list):
            self.cols = [cols]
        else:
            self.cols = cols

    def fit(self, X, y = None):
        # there is nothing to fit
        self.cols_ = self.cols
        return self

    def transform(self, X, y = None):
        trans = X.drop(self.cols_, axis=1).copy()
        self.features = trans.columns.tolist()

        return trans
    
    def get_feature_names(self):
        return self.features
    
# #Categrical features to pass down the categorical pipeline 
# cateforical_features = ['creditCard',cardCVV', 'enteredCVV',"merchantCategoryCode","merchantName","creditLimit"]

# #dummy features to get one-hot encoded to pass down categorical pipeline
# dummy_features = ["acqCountry","merchantCountryCode","posEntryMode","posConditionCode","transactionType"]
    
class CategoricalTransformer(BaseEstimator, TransformerMixin):
    """
    Docstring: covert defined columns to category datatype
    
    Parameters
    ----------
    cols: columns to be used
    
    Return
    ----------
    df: new dataset after datatype converting"
    """
    def __init__(self):
        self.order = {}
        self.features = []

    def fit(self, X, y=None):
        # there is nothing to fit
        df = X.copy()
        for n,c in df.items():
#             if #(n=="creditCard") or (n=="reg_merchantName"):
#                 continue
            if n=="creditLimit":
                df[n] = c.astype(str)
                df[n] = df[n].astype('category')
                df[n].cat.set_categories(['250.0','500.0','1000.0','2500.0','5000.0','7500.0','10000.0','15000.0','20000.0','50000.0'],
                                         ordered=True,inplace=True)
                self.order[n]=['250.0','500.0','1000.0','2500.0','5000.0','7500.0','10000.0','15000.0','20000.0','50000.0']
                
            if c.nunique()>11:
                df[n] = c.astype("category").cat.as_ordered()
                self.order[n]=df[n].cat.categories
        return self

    def transform(self, X,y=None):
        df = X.copy()
        # normalize merchantName by removing the trailing number identifier "#XXXXX"
        #df["reg_merchantName"] = df["merchantName"].apply(lambda x:re.sub(r"\ \#.+","",x))
        df["notSameCVV"] = df['cardCVV']!=df['enteredCVV']
        df.drop(columns=['cardCVV','enteredCVV'],inplace=True)
        for n,c in df.items():
            if n in self.order:
                df[n] = c.astype("category").cat.as_ordered()
                df[n].cat.set_categories(self.order[n],ordered=True,inplace=True)
                df[n] = df[n].cat.codes+1
        df.fillna("missing",inplace=True)
        
        self.features = df.columns.tolist()
        return df
    
    def get_feature_names(self):
        return self.features
    
    
    # #datetime features to pass down the datetime pipeline 
# datetime_features = ["transactionDateTime","accountOpenDate","dateOfLastAddressChange",'currentExpDate']



class DateTransformer(BaseEstimator, TransformerMixin):
    """
    Docstring: covert defined columns to datetime datatype
    
    Parameters
    ----------
    cols: columns to be used
    add_day_col: the columns that need to add day to the date
    
    Return
    ----------
    df: new dataset after datatype converting"
    """
    def __init__(self,add_day_col=None):
        self.add_day_col = add_day_col
        self.features = []
        
    def fit(self, X,y=None):
        # there is nothing to fit
        return self

    def transform(self, X,y=None):
        df = X.copy()
        for c in df.columns:
            df[c] = pd.to_datetime(df[c])

        # convert and complete (add the last day of month) the currentExpDate column [1]
        if self.add_day_col:
            df[self.add_day_col] = pd.to_datetime(df[self.add_day_col]) + MonthEnd(1)
            
        self.features = df.columns.tolist()
        return df
    
    def get_feature_names(self):
        return self.features
    
class ExtractDatePart(BaseEstimator, TransformerMixin):
    """
    Docstring: covert defined columns to datetime datatype
    
    Parameters
    ----------
    cols: columns to be used
    add_day_col: the columns that need to add day to the date
    
    Return
    ----------
    df: new dataset after datatype converting"
    """
    def __init__(self,cols):
        self.features = []
        if not isinstance(cols, list):
            self.cols = [cols]
        else:
            self.cols = cols
        
    def fit(self, X,y=None):
        # there is nothing to fit
        self.cols_ = self.cols
        return self

    def transform(self, X,y=None):
        df = X.copy()
        for date_col in self.cols:
            fld = df[date_col]
            if not np.issubdtype(fld.dtype, np.datetime64):
                df[date_col] = fld = pd.to_datetime(fld,
                                             infer_datetime_format=True)
            targ_pre = re.sub('[Dd]ate$', '', date_col)
            for n in ('Month', 'Week', 'Day', #'Dayofweek',
                    'Dayofyear', 'Is_month_end', 'Is_month_start',#'hour',
                    'Is_quarter_end', 'Is_quarter_start'):
                #look inside of "dt" and finds an attribute with that name
                df[targ_pre+n] = getattr(fld.dt,n.lower())
        self.features = df.columns.tolist()
        return df
    
    def get_feature_names(self):
        return self.features
    
class GetDayDiff(BaseEstimator, TransformerMixin):
    """
    Docstring: calculate how many different days between some datetime columns
    
    Parameters
    ----------
    cols: columns to be used
    
    Return
    ----------
    df: new dataset
    """
    def __init__(self):
        self.features = []

    def fit(self, X,y=None):
        # there is nothing to fit
        return self

    def transform(self, X,y=None):
        df = X.copy()
        df['days_Open_trans'] = (df['transactionDateTime'] - df['accountOpenDate']).dt.days
        # 'dateOfLastAddressChange' converts to How many days away from the transaction time?
        df['days_AddressChange_trans'] = (df['transactionDateTime'] - df['dateOfLastAddressChange']).dt.days
        # 'currentExpDate' converts to: How many days away from the transaction time?
        df['days_Exp_trans'] = (df['currentExpDate'] - df['transactionDateTime']).dt.days

        df.drop(columns=['accountOpenDate','dateOfLastAddressChange','currentExpDate','transactionDateTime'],inplace=True)
        self.features = df.columns.tolist()
        return df
    
    def get_feature_names(self):
        return self.features
    
    
class AnomalyTransformer(BaseEstimator, TransformerMixin):
    """
    Docstring: covert defined columns to category datatype
    
    Parameters
    ----------
    cols: columns to be used
    
    Return
    ----------
    df: new dataset after datatype converting"
    """
    def __init__(self,cols:list =None):
        self.normal = defaultdict(set)
        self.cols = cols
        self.features=[]

    def fit(self, X, y=None):
        # transform the dataset inplace
        #df = X.copy()
        X['isFraud']=y.values
        #X.sort_values(by=["customerId","transactionDateTime"],inplace=True)
        for c in self.cols:
            is_bad = X[["customerId","transactionDateTime",'isFraud',c]].groupby(
                'customerId',as_index=False).apply(self.anmly,col_name=c)
            X["ab_"+c]=is_bad.reset_index(level=0, drop=True)
            if c=="reg_merchantName":
                del X[c]
        del X['isFraud']
        X = X.drop(['transactionDateTime','customerId'], axis=1, errors='ignore')        
        return self
    
    def anmly(self,df,col_name=None):
        """
        Docstring: check if current value in col_name is deviated from the previous values in time order; save the non-fraud value list for                    future transformation on new data
        """
        cur_val=set()
        out = []
        frd = df['isFraud'].values
        #print(frd)
        for i,v in enumerate(df[col_name]):
            if not frd[i]: 
                cur_val.add(v)
            else:
                if not v in cur_val:
                    out.append(1)
                    continue
            out.append(0)
        self.normal[df.name].update(cur_val)
        return pd.Series(out,index=df.index,name="is_n")#,index=range(len(df)))
    
    def ifin(self,df,col_name=None):
        """
        Docstring: check if current value in col_name is deviated from the previous values self.normal list
        """
        #print(df["merchantName"])
        idx = df['customerId']
        if idx in self.normal:
            if df[col_name] not in self.normal[idx]:
                return 1
        return 0
    
    def transform(self, X,y=None):
        """
        Docstring: check if current value in col_name is deviated from the previous values self.normal list
        """
        df = X.copy()
        for c in self.cols:
            # if c in df, it means that the df was not fitted
            if c in df.columns:
                df["ab_"+c]=df.apply(self.ifin,col_name=c,axis=1)
                if c=="reg_merchantName":
                    del df[c]
        df = df.drop(['transactionDateTime','customerId'], axis=1, errors='ignore')        
        self.features += df.columns.tolist()
        return df
    
    def get_feature_names(self):
        return self.features
    
    
def convert_to_cat(df):
    """
    Docstring: covert defined columns to category datatype
    
    Parameters
    ----------
    cols: columns to be used
    
    Return
    ----------
    trans: new dataset after datatype converting"
    """
    global to_cat_column
    df = df.copy()
    if not isinstance(to_cat_colum, list):
        cols = [to_cat_colum]
    else:
        cols = to_cat_colum
    for c in cols:
        df[c] = df[c].astype('category')
 
    return df

def convert_to_date(df):
    """
    Docstring: covert defined columns to datetime datatype
    
    Parameters
    ----------
    cols: columns to be used
    
    Return
    ----------
    trans: new dataset after datatype converting"
    """
    global to_date_column
    global exp_date_col
    df = df.copy()
    if not isinstance(to_date_column, list):
        cols = [to_date_column]
    else:
        cols = to_date_column
    for c in cols:
        # convert time-related columns to datetime format
        df[c] = pd.to_datetime(df[c])
 
    # convert and complete (add the last day of month) the currentExpDate column [1]
    df[exp_date_col] = pd.to_datetime(card[exp_date_col]) + MonthEnd(1)
    return df






"""""""""""""""""""""""""""""
self-modified fastai packages
"""""""""""""""""""""""""""""

def add_datepart(df):
    """
    Docstring: extracts particular date fields from a complete datetime for the purpose of constructing categoricals
    
    Parameters
    ----------
    df: pandas dataframe
    col: datetime columns needs to be processed
    drop: if drop col
    """
    global date_col
    df = df.copy()
    fld = df[date_col]
    if not np.issubdtype(fld.dtype, np.datetime64):
        df[date_col] = fld = pd.to_datetime(fld, 
                                     infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', date_col)
    for n in ('Month', 'Week', 'Day', 'Dayofweek', 
            'Dayofyear', 'hour','Is_month_end', 'Is_month_start', 
            'Is_quarter_end', 'Is_quarter_start'):
        #look inside of "dt" and finds an attribute with that name
        df[targ_pre+n] = getattr(fld.dt,n.lower())
    #df[targ_pre+'Elapsed'] = fld.astype(np.int64) // 10**9
    return df

def day_diff(df):
    # 'accountOpenDate' converts to How many days away from the transaction time?
    df = df.copy()
    df['days_Open_trans'] = (df['transactionDateTime'] - df['accountOpenDate']).dt.days
    # 'dateOfLastAddressChange' converts to How many days away from the transaction time?
    df['days_AddressChange_trans'] = (df['transactionDateTime'] - df['dateOfLastAddressChange']).dt.days
    # 'currentExpDate' converts to: How many days away from the transaction time?
    df['days_Exp_trans'] = (df['currentExpDate'] - df['transactionDateTime']).dt.days

    df.drop(columns=['accountOpenDate','dateOfLastAddressChange','currentExpDate'],inplace=True)
    return df



# Cell
def train_cats(df):
    """Change any columns of strings in a panda's dataframe to a column of
    categorical values. This applies the changes inplace.
    """
    for n,c in df.items():
        if is_string_dtype(c): df[n] = c.astype('category').cat.as_ordered()

# Cell
def apply_cats(df, trn):
    """Changes any columns of strings in df into categorical variables using trn as
    a template for the category codes.
    """
    for n,c in df.items():
        if (n in trn.columns) and (trn[n].dtype.name=='category'):
            df[n] = c.astype('category').cat.as_ordered()
            df[n].cat.set_categories(trn[n].cat.categories, ordered=True, inplace=True)
            
# Cell
def fix_missing(df, col, name, na_dict):
    """ Fill missing data in a column of df with the median, and add a {name}_na column
    which specifies if the data was missing.
    """
    if is_numeric_dtype(col):
        if pd.isnull(col).sum() or (name in na_dict):
            df[name+'_na'] = pd.isnull(col)
            filler = na_dict[name] if name in na_dict else col.median()
            df[name] = col.fillna(filler)
            na_dict[name] = filler
    return na_dict


# Cell
def numericalize(df, col, name, max_n_cat):
    """ Changes the column col from a categorical type to it's integer codes.
    """
    if not is_numeric_dtype(col) and ( max_n_cat is None or len(col.cat.categories)>max_n_cat):
        df[name] = pd.Categorical(col).codes+1
        
# Cell
def proc_df(df, y_fld=None, skip_flds=None, ignore_flds=None, do_scale=False, na_dict=None,
            preproc_fn=None, max_n_cat=None, subset=None, mapper=None):
    """ proc_df takes a data frame df and splits off the response variable, and
    changes the df into an entirely numeric dataframe. For each column of df
    which is not in skip_flds nor in ignore_flds, na values are replaced by the
    median value of the column.
    """
    if not ignore_flds: ignore_flds=[]
    if not skip_flds: skip_flds=[]
    if subset: df = get_sample(df,subset)
    else: df = df.copy()
    ignored_flds = df.loc[:, ignore_flds]
    df.drop(ignore_flds, axis=1, inplace=True)
    if preproc_fn: preproc_fn(df)
    if y_fld is None: y = None
    else:
        if not is_numeric_dtype(df[y_fld]): df[y_fld] = pd.Categorical(df[y_fld]).codes
        y = df[y_fld].values
        skip_flds += [y_fld]
    df.drop(skip_flds, axis=1, inplace=True)

    if na_dict is None: na_dict = {}
    else: na_dict = na_dict.copy()
    na_dict_initial = na_dict.copy()
    for n,c in df.items(): na_dict = fix_missing(df, c, n, na_dict)
    if len(na_dict_initial.keys()) > 0:
        df.drop([a + '_na' for a in list(set(na_dict.keys()) - set(na_dict_initial.keys()))], axis=1, inplace=True)
    if do_scale: mapper = scale_vars(df, mapper)
    for n,c in df.items(): numericalize(df, c, n, max_n_cat)
    df = pd.get_dummies(df, dummy_na=True)
    df = pd.concat([ignored_flds, df], axis=1)
    res = [df, y, na_dict]
    if do_scale: res = res + [mapper]
    return res