
# coding: utf-8

# In[1]:

# Import statements

import MySQLdb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score 

from HTMLParser import HTMLParser
import htmlentitydefs
import itertools
import re
import datetime

import sys

get_ipython().magic(u'matplotlib inline')


# # Importing Database

# In[2]:

# Create database object

db = MySQLdb.connect("localhost", "root", "", "junglescene")


# In[3]:

# Create database cursor object

cursor = db.cursor()


# In[4]:

cursor.execute("SHOW TABLES")


# In[5]:

cursor.fetchall()


# In[6]:

# Create forumsdata DataFrame. forumsdata is all the collected posts from several of the Junglescene database forums.
# Data comes from all forums with over 10000 posts. These forums are: Jungle Scene, Clubs & Parties, Photos, Producing,
# Neptune For Poetry, Hip Hop, Party Reviews, General Discussion, Ragga, Dubstep, Sets & Tracklistings, and 
# Dancer's Corner. 

# Create DataFrame column names for the forumsdata DataFrame

table_cols = ['tid', 'pid', 'rid', 'children', 'subject', 'body', 'username', 
              'user_addr', 'mailreply', 'rank', 'status', 'modified', 'created', 
              'lastip', 'userstatus', 'closed', 'forum_id']

# Create list of sql forum table names
forumtablenames = ['forumstopics1', 'forumstopics2', 'forumstopics122', 'forumstopics9', 'forumstopics40', 
                   'forumstopics30', 'forumstopics63', 'forumstopics137', 'forumstopics25', 'forumstopics130', 
                  'forumstopics125','forumstopics44']

# Create list of forum id from sql. These will mark the posts in the forumsdata DataFrame so their origin is known.
forumid = [1,2,122,9,40,30,63,137,25,130,125,44]

# Create empty DataFrame
forumsdata = pd.DataFrame(columns=table_cols)


# Create forumsdata
for tn,fid in zip(forumtablenames, forumid):
    cursor.execute("select * from {}".format(tn))
    data = cursor.fetchall()

    temp_forum = pd.DataFrame(list(data))
    temp_forum['forum_id'] = fid
    temp_forum.columns = table_cols
    
    forumsdata = forumsdata.append(temp_forum)
    
forumsdata.reset_index(inplace=True)


# # Cleaning, Analysis, Feature Engineering

# In[7]:

# Separating time information

forumsdata['modifiedyear'] = forumsdata['modified'].dt.year
forumsdata['modifiedmonth'] = forumsdata['modified'].dt.month
forumsdata['modifiedday'] = forumsdata['modified'].dt.day
forumsdata['modifiedhour'] = forumsdata['modified'].dt.hour
forumsdata['modifiedminute'] = forumsdata['modified'].dt.minute
forumsdata['modifiedsecond'] = forumsdata['modified'].dt.second

forumsdata['createdyear'] = forumsdata['created'].dt.year
forumsdata['createdmonth'] = forumsdata['created'].dt.month
forumsdata['createdday'] = forumsdata['created'].dt.day
forumsdata['createdhour'] = forumsdata['created'].dt.hour
forumsdata['createdminute'] = forumsdata['created'].dt.minute
forumsdata['createdminute'] = forumsdata['created'].dt.minute
forumsdata['createdsecond'] = forumsdata['created'].dt.second

forumsdata.head()


# In[112]:

forumsdata.shape


# In[8]:

forumsdata['subject'] = forumsdata['subject'].str.replace('Re: ', '')
forumsdata.head()


# In[9]:

# Number of threads closed

len(forumsdata[forumsdata['closed']==1]['subject'].unique())


# In[10]:

# Number of threads open

len(forumsdata[forumsdata['closed']==0]['subject'].unique())


# In[11]:

# There is a discrepency between the closed and not closed topics. I think there are some threads with the same subject
# that are marked as closed, and as open. I.E. some posts in the same subject were marked closed and others as open.

len(forumsdata['subject'].unique())


# In[12]:

89059+215


# In[13]:

# It looks like there are 18 of these threads

89274-89256


# In[14]:

# Set of closed subjects
closed_subjects = set(forumsdata[forumsdata['closed']==1]['subject'].unique())

# Set of open subjects
open_subjects = set(forumsdata[forumsdata['closed']==0]['subject'].unique())

# List of subjects where thread is marked as both closed and open
duplicate_closedopen = list(set.intersection(closed_subjects, open_subjects))


# In[15]:

# Subject titles for threads with both closed and open designations

duplicate_closedopen


# In[16]:

# Deleting threads marked both closed and open

sizeb4 = forumsdata.size
print 'Number of posts Before: ', sizeb4

for x in duplicate_closedopen:
    dup_idx = forumsdata[forumsdata['subject']==x].index
    forumsdata = forumsdata.drop(dup_idx)
    
sizeafter = forumsdata.size
print 'Number of posts after: ', sizeafter
print 'Number of posts deleted: ', sizeb4-sizeafter


# In[17]:

# Verify threads marked both closed/open are gone

# Set of closed subjects
closed_subjects = set(forumsdata[forumsdata['closed']==1]['subject'].unique())

# Set of open subjects
open_subjects = set(forumsdata[forumsdata['closed']==0]['subject'].unique())

# List of subjects where thread is marked as both closed and open
duplicate_closedopen = list(set.intersection(closed_subjects, open_subjects))
duplicate_closedopen


# In[18]:

forumsdata.describe()


# In[19]:

# Look at example of body text. We can see it's got all kind of special characters we need to remove.

forumsdata['body'][12]


# In[20]:

# Clean up body text

# Class to help parse html within a body of text

class HTMLTextExtractor(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.result = [ ]

    def handle_data(self, d):
        self.result.append(d)

    def handle_charref(self, number):
        codepoint = int(number[1:], 16) if number[0] in (u'x', u'X') else int(number)
        self.result.append(unichr(codepoint))

    def handle_entityref(self, name):
        codepoint = htmlentitydefs.name2codepoint[name]
        self.result.append(unichr(codepoint))

    def get_text(self):
        return u''.join(self.result)

# Function to call to use HTMLTextExtractor Class

def html_to_text(html):
    # Set up HTMLTextExtractor object and send for processing
    s = HTMLTextExtractor()
    s.feed(html)
    return s.get_text()
    
def clean_string(string):
    
    # Dictionary used to help remove extra characters HTMLTextExtractor did not get
    dict = {'\\r':' ','\\n':' ','\n':' ','\r':' ','\u':' ','\\':' ','/':' ', '@':' ', '.':' ',
            '?':' ','-':' ','!':' ',',':' ',':':' ','[':' ',']':' ', '(':' ', ')':' ','\"':' ',
            '\'':'', '#':' ','~':' ', '&':''}
    
    # Remove characters from dict
    for i,j in dict.iteritems():
         string = string.replace(i,j)
            
    #Remove html using html_to_text
    string = html_to_text(string)
            
    # Remove some Unicode punctuation
    string = re.sub(r"\u2019",'',string)

    # Remove URLs
    string = re.sub(r"http\S+", "link", string)
    
    # Remove edit tags
    string = string.split('*$$ ')[0]
    
    # Return the super clean string
    return string.encode('ascii', 'ignore')


# In[21]:

clean_post=[]
for post in forumsdata['body']:
    try:
        clean_post.append(clean_string(post.decode('utf-8')))
    except:
        e = sys.exc_info()[0]
        clean_post.append(str(e))

forumsdata['body'] = clean_post


# In[22]:

forumsdata.head()


# In[23]:

# Clean body text!

forumsdata['body'][13840]


# In[24]:

# Getting rid of features that don't make sense as predictors

del_cols = ['tid', 'pid', 'rid', 'children', 'user_addr', 'mailreply', 'lastip']

forumsdata = forumsdata.drop(del_cols, axis = 1)
forumsdata.head()


# In[25]:

# Eliminate posts with empty bodys. Most likely these were posts made up entirely of html. There were only ~7000 posts
# like this, of which only 65 were from closed threads.

forumsdata = forumsdata.drop(forumsdata[forumsdata['body']==''].index)
forumsdata = forumsdata.drop(forumsdata[forumsdata['subject']==''].index)


# In[26]:

forumsdata[forumsdata['subject']=='']


# In[27]:

# Top posters in closed threads

forumsdata[forumsdata['closed']==1]['username'].value_counts()


# In[28]:

# Top posters in open threads
forumsdata[forumsdata['closed']==0]['username'].value_counts()


# In[29]:

# Top posters in all threads
forumsdata['username'].value_counts()


# # Create aggregate DataFrame threaddf 
# 
# ### Used to collect whole thread stats

# In[30]:

# Create aggregate Dataframe to collect whole thread stats

threaddf = pd.DataFrame(forumsdata.groupby('subject').first().forum_id).astype(int)
threaddf = threaddf.reset_index()


# In[31]:

threaddf.head()


# In[32]:

# Creating some features for threaddf

threaddf['timelength'] = list(forumsdata.groupby('subject')['created'].max()-forumsdata.groupby('subject')['created'].min())
threaddf['threadlength'] = list(forumsdata.groupby('subject').size())
threaddf['timelength'] = threaddf['timelength']+datetime.timedelta(0,1)
threaddf['post_per_sec'] = threaddf['threadlength']/threaddf['timelength'].apply(lambda x: x.total_seconds())
threaddf['most_common_user'] = list(forumsdata.groupby('subject')['username'].agg(lambda x: x.value_counts().index[0]))


# In[33]:

threaddf[(threaddf['post_per_sec']<0)]


# In[34]:

threaddf.head()


# In[35]:

# Removes reply text from posts. This ensures that another user's sentiments aren't a) repeated in the thread, 
# and b) influencing the sentiment of another user's post

for i in list(forumsdata['body'].index):
    if '**********' in forumsdata['body'][i]:
        head, sep, tail = forumsdata['body'][i].partition('**********')
        forumsdata.set_value(i, 'body', head)


# In[36]:

# Make dependent variable type integer.

threaddf['closed'] = list(forumsdata.groupby('subject')['closed'].first().astype(int))


# In[37]:

threaddf.describe()


# # Create Sentiment Ratings For Each Post

# In[38]:

# Lexicon of negative words
neg = pd.read_csv('opinion-lexicon-English/negative-words.csv')


# In[39]:

# Lexicon of positive words
pos = pd.read_csv('opinion-lexicon-English/positive-words.csv')


# In[40]:

# Both of the lists above were provided by the paper: 
# Minqing Hu and Bing Liu. "Mining and Summarizing Customer Reviews." 
# Proceedings of the ACM SIGKDD International Conference on Knowledge 
# Discovery and Data Mining (KDD-2004), Aug 22-25, 2004, Seattle, Washington, USA 


# In[41]:

neg.head()


# In[42]:

pos.head()


# In[43]:

# Creates a sentiment rating for all posts in forumsdata. Averages all positive and negative words 
# in the post that are shared with two lists

pos_set = set(pd.read_csv('opinion-lexicon-English/positive-words.csv')['words'])
neg_set = set(pd.read_csv('opinion-lexicon-English/negative-words.csv')['words'])
forumsdata['sent_rating'] =np.nan

for index in forumsdata.index:
    post = forumsdata['body'][index]
    post_set = set(post.split())
    forumsdata.set_value(index, 'sent_rating', np.mean([len(set.intersection(post_set, pos_set)), 
                                              -1*len(set.intersection(post_set, neg_set))]))


# In[44]:

forumsdata[['body', 'sent_rating']].head(20) 


# In[45]:

# A telling example of the algorithm not picking up on slang. The lexicon classifies 'dope' as a negative word
# but here it is used in a positive context. Interestingly enough, there is a little negative sentiment in here
# as the user refers to a party that was poorly attended. So, the fact that it has a negative rating works in this case.
# Maybe the negative rating shouldn't be so strong though. 

print forumsdata['body'][185]
print forumsdata['sent_rating'][185]


# ### Sentiment rating may be an important feature for classifier models down the road.

# In[46]:

# Percent of posts in closed threads with negative sentiments
len(forumsdata[(forumsdata['closed']==1)&(forumsdata['sent_rating']<0)])/float(len(forumsdata[forumsdata['closed']==1]))*100


# In[47]:

# Percent of posts in open threads with negative sentiments
len(forumsdata[(forumsdata['closed']==0)&(forumsdata['sent_rating']<0)])/float(len(forumsdata[forumsdata['closed']==0]))*100


# In[48]:

# Creating a sentiment feature for our aggregate DataFrame

threaddf['thread_sent'] = list(forumsdata.groupby('subject').mean()['sent_rating'])


# In[49]:

threaddf.head()


# In[50]:

threaddf.groupby('closed').describe()


# # Data Visualizations for Separate Features of threaddf vs. Closed

# In[51]:

plt.scatter(threaddf['thread_sent'],threaddf['closed'])
plt.xlabel('Thread Sentiment')
plt.ylabel('0=Open, 1=Closed')
plt.title('Thread Sentiment vs. Closed/Open')


# In[52]:

plt.hist(threaddf[threaddf['closed']==1]['threadlength'], 
         bins = 20, weights= np.zeros_like(threaddf[threaddf['closed']==1]['threadlength'])
         +100.0/threaddf[threaddf['closed']==1]['threadlength'].size)

plt.xlabel('Threadlength (Closed Threads)')
plt.ylabel('Percent')
plt.ylim((0,100))
plt.title('Percent Distribution of Threadlength For Closed Threads')


# In[53]:


plt.hist(threaddf[threaddf['closed']==0]['threadlength'], range = (0,300), 
         bins = 20, weights= np.zeros_like(threaddf[threaddf['closed']==0]['threadlength'])
         +100.0/threaddf[threaddf['closed']==0]['threadlength'].size)

plt.xlabel('Threadlength (Open Threads)')
plt.ylabel('Percent')
plt.ylim((0,100))
plt.title('Percent Distribution of Threadlength For Open Threads')


# In[54]:

threadtime = threaddf[threaddf['closed']==1]['timelength'].apply(lambda x: x.total_seconds())

plt.hist(threadtime, range = (0,500000), weights= np.zeros_like(threadtime)+100.0/threadtime.size)

plt.xlabel('Seconds (Closed Threads)')
plt.ylabel('Percent')
plt.ylim((0,100))
plt.title('Percent Distribution of Timelength For Closed Threads')


# In[55]:

threadtime = threaddf[threaddf['closed']==0]['timelength'].apply(lambda x: x.total_seconds())

plt.hist(threadtime, range = (0,500000), weights= np.zeros_like(threadtime)+100.0/threadtime.size)

plt.xlabel('Seconds (Open Threads)')
plt.ylabel('Percent')
plt.ylim((0,100))
plt.title('Percent Distribution of Timelength For Open Threads')


# In[56]:

forum_id_closed = threaddf[threaddf['closed']==1].sort('forum_id')['forum_id']
forum_id_closed_prct = forum_id_closed.value_counts()/forum_id_closed.value_counts().sum()*100

plt.bar(forum_id_closed.unique(), forum_id_closed_prct)

plt.xlabel('Forum ID')
plt.xlim((0,137))
plt.ylabel('Percent')
plt.ylim((0,100))
plt.title('Distribution of Forum IDs For Closed Threads')


# In[57]:

forum_id_open = threaddf[threaddf['closed']==0].sort('forum_id')['forum_id']
forum_id_open_prct = forum_id_open.value_counts()/forum_id_open.value_counts().sum()*100

plt.bar(forum_id_open.unique(), forum_id_open_prct)

plt.xlabel('Forum ID')
plt.xlim((0,137))
plt.ylabel('Percent')
plt.ylim((0,100))
plt.title('Distribution of Forum IDs For Open Threads')


# In[58]:

# Percent of threads that were closed with a negative sentiment rating

len(threaddf[(threaddf['closed']==1)&(threaddf['thread_sent']<0)])/float(len(threaddf[threaddf['closed']==1]))*100


# In[59]:

# Percent of threads that were open with a negative sentiment rating

len(threaddf[(threaddf['closed']==0)&(threaddf['thread_sent']<0)])/float(len(threaddf[threaddf['closed']==0]))*100


# # Modeling

# In[60]:

# Closed thread counts

threaddf[threaddf['closed']==1].count()


# In[61]:

# Open thread counts

threaddf[threaddf['closed']==0].count()


# In[62]:

# Percent distribution of posts in forums among Open threads. (forum_id is on the left)

forum_id_open_prct


# In[64]:

# We have a huge class imbalance, so we need to sample from the Open threads so we 
# at least balance our classes out. Not only that, but so that our sample matches its population,
# we need to take a probability sample of the open threads, so that our sample has the same 
# post distribution among the forums as the population. 

fid_sample_n = {'1.0':64, '2.0':56, '40.0':16, '30.0':12, '9.0':10, '125.0':8, '122.0':8, '63.0':6, '130.0':6, 
                '25.0':6, '137.0':4, '44.0':4}
open_thread_sample = pd.DataFrame()

for fid, num in fid_sample_n.iteritems():
    open_thread_sample = open_thread_sample.append(threaddf[threaddf['forum_id']==float(fid)].sample(n=num, random_state = 9))

open_thread_sample[open_thread_sample['forum_id']==40.0].count()


# In[65]:

# Combine the open thread prob sample with all the closed thread data. We should have 200 if all worked well.

threaddf_sample = threaddf.drop(threaddf[threaddf['closed']==0].index).append(open_thread_sample)
threaddf_sample[threaddf_sample['closed']==0].count()


# In[66]:

threaddf.head()


# In[67]:

#Convert timelength to integer seconds

threaddf_sample['timelength'] = threaddf_sample['timelength'].astype('timedelta64[s]').astype(int)
threaddf_sample.head()


# In[71]:

# Create independent, dependent variables

X = threaddf_sample.drop(['closed', 'most_common_user', 'subject', 'forum_id'], axis = 1).reset_index()
y = threaddf_sample.closed.reset_index()


# In[72]:

# Cleaning up X

X = X.drop('index', axis = 1)
X.head()


# In[73]:

# Cleaning up y

y = y.drop('index', axis = 1)
y.head()


# In[74]:

# Train-test splits

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 9)


# In[75]:

# Some features have large ranges, so we create another train-test split, but with a log() function applied to them.
# This may help our modeling down the road.

threaddf_sample2 = threaddf_sample
threaddf_sample2.head()


# In[76]:

X_train.head()


# In[77]:

threaddf_sample2['timelength'] = np.log(threaddf_sample['timelength'])
threaddf_sample2['post_per_sec'] = np.log(threaddf_sample['post_per_sec'])


# In[78]:

# Second log() set of data

X2 = threaddf_sample2.drop(['closed', 'most_common_user', 'subject'], axis = 1).reset_index()
X2 = X2.drop('index', axis = 1)
y2 = threaddf_sample2.closed.reset_index()
y2 = y2.drop('index', axis = 1)


# In[79]:

X2.head()


# In[80]:

# Logged data train-test split

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, random_state = 9)


# In[81]:

X_train.head()


# In[82]:

X_train2.head()


# In[83]:

X_test.head()


# In[84]:

X_test2.head()


# In[85]:

y_train.head()


# In[86]:

y_train2.head()


# ## Naive Bayes (GaussianNB)

# In[87]:

# testing accuracy of Gaussian Naive Bayes on first test set

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_class = gnb.predict(X_test)
print metrics.accuracy_score(y_test, y_pred_class)


# In[88]:

# Confusion Matrix

print metrics.confusion_matrix(y_test, y_pred_class)


# In[89]:

# Accuracy

print metrics.roc_auc_score(y_test, y_pred_class)


# In[90]:

# testing accuracy of Gaussian Naive Bayes on logged dataset

gnb2 = GaussianNB()
gnb2.fit(X_train2, y_train2)
y_pred_class2 = gnb2.predict(X_test2)
print metrics.accuracy_score(y_test2, y_pred_class2)


# In[91]:

# Confusion Matrix

print metrics.confusion_matrix(y_test2, y_pred_class2)


# In[92]:

# Accuracy

print metrics.roc_auc_score(y_test2, y_pred_class2)


# ## Logistic Regression Model

# In[93]:

# Logistic Regression Model

lrm = LogisticRegression()

lrm_cv = GridSearchCV(estimator=lrm, param_grid={'C': [10**-i for i in range(-10, 10)]}, scoring = 'roc_auc', cv = 5)

lrm_cv.fit(X_train.values, y_train['closed'].values)

print "Best Params: ",lrm_cv.best_params_

print "Best estimator: ",lrm_cv.best_estimator_

print "Best Roc_Auc Score: ",lrm_cv.best_score_


# ## Support Vector Machines w/ GridSearchCV

# In[95]:

# Support Vector Machines Model

from sklearn import preprocessing

svc = SVC()

#param_grid = [
# {'C': [1, 10, 100], 'kernel': ['linear']},
# {'C': [1, 10, 100], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}]

svc_cv = GridSearchCV(estimator = svc, param_grid = {'C':10, 'Kernel':'linear'}, scoring = 'roc_auc', cv = 5)

svc_cv.fit(X_train.values, y_train['closed'].values)

print "Best Params: ",svc.best_params_ 

print "Best estimator: ",svc.best_estimator_

print "Best Roc_Auc Score: ",svc.best_score_


# ## Random Forest w/ GridSearchCV

# In[101]:

# Random Forest with GridSearchCV

rfm = RandomForestClassifier()

param_grid = {'n_estimators':[20,200], 'max_features':['auto', 'sqrt','log2']}

cv_rfm = GridSearchCV(estimator = rfm, param_grid = param_grid, cv = 6, scoring = 'roc_auc')
cv_rfm.fit(X_train.values, y_train['closed'].values)

print "Best Params: ",cv_rfm.best_params_

print "Best estimator: ",cv_rfm.best_estimator_

print "Best Roc_Auc Score: ",cv_rfm.best_score_




# In[102]:

# Best estimator

best_rfm = cv_rfm.best_estimator_


# In[103]:

# Feature Importances

feature_importances = best_rfm.feature_importances_
feature_names = X_train.columns.values
feature_imp_df = pd.DataFrame({'Features':feature_names, 'Importance':feature_importances})
feature_imp_df


# In[104]:

# Create prediction values for Random Forest 

rfm_predict = best_rfm.predict(X_test)


# In[105]:

# Calculate Random Forest model accuracy

rfm_accuracy = metrics.accuracy_score(y_test, rfm_predict)
rfm_accuracy


# In[106]:

# ROC Curve for Random Forest Model

rfm_probas = best_rfm.predict_proba(X_test)
plt.plot(roc_curve(y_test, rfm_probas[:,1])[0], roc_curve(y_test, rfm_probas[:,1])[1])
plt.xlabel('False Positve Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve: GridSearchCV Random Forest Model')


# In[107]:

# Classification Report for Random Forest model

classification_report(y_test, rfm_predict)


# In[133]:

# Can we beat the Random Forest with the logged data?

rfm2 = RandomForestClassifier()

param_grid2 = {'n_estimators':[20,200], 'max_features':['auto', 'sqrt','log2']}

cv_rfm2 = GridSearchCV(estimator = rfm2, param_grid = param_grid2, cv = 6, scoring = 'roc_auc')
cv_rfm2.fit(X_train2.values, y_train2['closed'].values)

print "Best Params: ",cv_rfm2.best_params_

print "Best estimator: ",cv_rfm2.best_estimator_

print "Best Roc_Auc Score: ",cv_rfm2.best_score_


# ### Random Forest model fit to the logged data has about the same performance

# In[134]:

# Best estimator

best_rfm2 = cv_rfm2.best_estimator_


# In[135]:

# Feature Importances

feature_importances2 = best_rfm2.feature_importances_
feature_names = X_train.columns.values
feature_imp_df2 = pd.DataFrame({'Features':feature_names, 'Importance':feature_importances2})
feature_imp_df2


# In[136]:

# Create prediction values for Random Forest 

rfm_predict2 = best_rfm2.predict(X_test2)


# In[137]:

# Calculate Random Forest model accuracy

rfm_accuracy2 = metrics.accuracy_score(y_test2, rfm_predict2)
rfm_accuracy2


# In[ ]:



