import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from pandas.tseries.offsets import MonthEnd
from sklearn.pipeline import make_pipeline, make_union
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer


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
    
    
def metric(self,prob,cl):
    self.fpr, self.tpr, threshold = roc_curve(self.y_valid, prob)
    auc = roc_auc_score(self.y_valid,prob)
    self.plot_roc_curve(self.fpr,self.tpr)
    acc = (self.y_valid==cl).mean()
    f1 = f1_score(self.y_valid,cl)
    print(acc,f1,auc)    
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
# #Numerical features to pass down the numerical pipeline 
# numerical_features = ['availableMoney', 'transactionAmount', 'currentBalance','posOnPremises','cardPresent']


class NumericalTransformer(BaseEstimator, TransformerMixin):
    #Class Constructor
    def __init__( self):
        pass
        
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
        return df

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
        return trans
    
    
# #Categrical features to pass down the categorical pipeline 
# cateforical_features = ['cardCVV', 'enteredCVV',"merchantCategoryCode","merchantName","creditLimit"]

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

    def fit(self, X, y=None):
        # there is nothing to fit
        df = X.copy()
        for n,c in df.items():
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

    def transform(self, X:pd.DataFrame):
        df = X.copy()
        # normalize merchantName by removing the trailing number identifier "#XXXXX"
        df["reg_merchantName"] = df["merchantName"].apply(lambda x:re.sub(r"\ \#.+","",x))
        df["notSameCVV"] = df['cardCVV']!=df['enteredCVV']
        df.drop(columns=['cardCVV','enteredCVV'],inplace=True)
        for n,c in df.items():
            if n in self.order:
                df[n] = c.astype("category").cat.as_ordered()
                df[n].cat.set_categories(self.order[n],ordered=True,inplace=True)
                df[n] = df[n].cat.codes+1
        df.fillna("missing",inplace=True)
        return df
    
    
    
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
        
        return df
    
    
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
            for n in ('Month', 'Week', 'Day', 'Dayofweek',
                    'Dayofyear', 'hour','Is_month_end', 'Is_month_start',
                    'Is_quarter_end', 'Is_quarter_start'):
                #look inside of "dt" and finds an attribute with that name
                df[targ_pre+n] = getattr(fld.dt,n.lower())
        return df

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
        pass

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

        df.drop(columns=['accountOpenDate','dateOfLastAddressChange','currentExpDate'],inplace=True)
        return df
    
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