#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system(' pip install mysql-connector-python-rf')
get_ipython().system(' pip install pyyaml')
get_ipython().system('conda install -c conda-forge wordcloud')
get_ipython().system('conda install wordcloud')


# In[18]:


conda install -c anaconda nltk


# In[3]:


import mysql.connector
import yaml
with open("credentials.yaml", "r") as f:
    params = yaml.load(f, Loader=yaml.FullLoader)


# In[4]:


cnx = mysql.connector.connect(
    host=params['host'],
    port=params['port'],
    user=params['user'],
    password=params['password'],
    database=params['database'],
    auth_plugin='mysql_native_password'
)


# In[5]:


cursor = cnx.cursor()


# In[6]:


cursor.execute("SELECT COUNT(*) QTY, cast(Created as Date) Date FROM tbl_Hashtag GROUP BY cast(Created as Date)")
hashtag = cursor.fetchall()
cursor.close()


# In[23]:


#cursor.close()
type(hashtag)


# In[24]:


import pandas as pd


# In[25]:


df = pd.DataFrame(hashtag)
df


# In[26]:


import pandas as pd
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# In[27]:


# set width of bar
barWidth = 1

# set height of bar
bars1 = df[0].values

# Set position of bar on X axis
#r1 = np.arange(len(bars1))

r1 = df[1].values

# Make the plot
plt.bar(r1, bars1, color='#000001', width=barWidth, edgecolor='white', label='hashtags')

# Add xticks on the middle of the group bars

plt.xlabel('Date', fontweight='bold')
#plt.xticks([r + barWidth for r in range(len(bars1))], df(1).unique())
plt.xticks(rotation='vertical')

# Create legend & Show graphic
plt.legend()
plt.xlabel("Date")
plt.ylabel("Count")
plt.show()


# # We identify the amount of daily hashtags related to spanish marihuana / marijuana.

# In[28]:


cursor = cnx.cursor()


# In[29]:


cursor.execute("SELECT Word, count(*) FROM db_BeduProject.tbl_Words WHERE ACTIVE = 1 and ProcessWord= 1 group by Word order by count(*) Desc")
word_spanish = cursor.fetchall()
cursor.close()


# In[30]:


word_spanish


# In[31]:


import pandas as pd


# In[32]:


word_spanish = pd.DataFrame(word_spanish)


# In[34]:


word_spanish.columns=['word', 'count']


# In[35]:


word_spanish


# In[37]:


text = word_spanish['word']
text


# In[38]:


get_ipython().system('pip install WordCloud')


# In[39]:


from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# In[46]:


wordcloud = WordCloud().generate(str(text[0:19,]))

plt.figure(figsize = (10, 15), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 

plt.show() 


# In[47]:


cursor = cnx.cursor()


# In[48]:


cursor.execute("Select Word, count(*) FROM tbl_Words WHERE ACTIVE = 1 and ProcessWord= 1 Group by Word ORDER BY count(*) desc") 
word_count = cursor.fetchall()
cursor.close()


# In[49]:


word_pareto = pd.DataFrame(word_count)


# In[50]:


word_pareto


# In[51]:


cursor = cnx.cursor()


# In[52]:


cursor.execute("SELECT * FROM tbl_userstwitter")
User = cursor.fetchall()
cursor.close()
User


# In[53]:


Tweet_users = pd.DataFrame(User)


# In[54]:


Tweet_users.columns=['Id','runner','user','datetime']


# In[55]:


Tweet_users


# In[56]:


Tweet_users.dtypes


# In[57]:


#convert datetime column to just date
Tweet_users['datetime'] = pd.to_datetime(Tweet_users['datetime']).dt.normalize()


# In[58]:


Tweet_users


# In[59]:


Tweet_users.groupby(['user']).user.value_counts().nlargest(20)


# In[70]:


#plot data
fig, ax = plt.subplots(figsize=(8,8))
Tweet_users.groupby(['user']).user.value_counts().nlargest(15).plot(ax=ax)
ax.set_ylabel('Count')
ax.set_xlabel('tweeter user')
for tick in ax.get_xticklabels():
        tick.set_rotation(25)
        
        


# In[1]:


Tweet_users.groupby(['user']).datetime.count().plot(kind='bar')


# In[ ]:


import seaborn as sns

ax = sns.countplot(x="datetime",data=Tweet_users)
for tick in ax.get_xticklabels():
    tick.set_rotation(90)


# In[ ]:


cursor = cnx.cursor()


# In[ ]:


cursor.execute("SELECT Seintiment FROM tbl_Text")
Seintiment = cursor.fetchall()
cursor.close()
Seintiment


# In[ ]:


Sentiment_analysis = pd.DataFrame(Seintiment,columns=['Value'])


# In[ ]:


Sentiment_analysis


# In[ ]:


Sentiment_analysis.loc[Sentiment_analysis['Value'] <.02 , 'Result'] = 'Negative' 
Sentiment_analysis.loc[Sentiment_analysis['Value']   >= 0.5, 'Result'] = 'Positive' 
print (Sentiment_analysis)


# In[ ]:


Sentiment_analysis['count'] = 1


# In[ ]:


value_counts = Sentiment_analysis.groupby(['Result'])['count'].sum()


# In[ ]:


value_counts


# In[109]:


plt.style.use('seaborn')

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot()

plt1 = ax.bar(value_counts.index, value_counts['N'], label='N',
              color=["#7788AA","#4E638E","#2E4372","#152A55"])

ax.set_ylabel('count')
ax.set_title('Sentiment tweeter users', fontsize=13, pad=15);
plt.legend((plt1[0], plt2[0]), ('Negative', 'Positive'));
ax.set_ylim(0, 4500);


# In[ ]:


plot = value_counts.plot.pie(y='results', figsize=(5, 5))


# In[7]:


cursor = cnx.cursor()


# In[8]:


cursor.execute("SELECT * FROM tbl_RetweetsFav")
RetweetsFav = cursor.fetchall()
cursor.close()
RetweetsFav


# In[9]:


RetweetsFav = pd.DataFrame(RetweetsFav)


# In[ ]:


RetweetsFav.columns=['Id','IdTweet','Retweet','Favs']


# In[ ]:


RetweetsFav


# In[58]:


sns.scatterplot(x=RetweetsFav['Favs'],y= RetweetsFav['Retweet'])


# In[59]:


RetweetsFav['Favs'].corr(RetweetsFav['Retweet'])


# In[ ]:





# In[10]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[11]:


cursor = cnx.cursor()


# In[12]:


cursor.execute("SELECT * FROM db_BeduProject.tbl_Text t left join tbl_AWSSentiment a on t.idText = a.idText WHERE SeintimentDate < '2021-07-05'") 
sentiment_analysis = cursor.fetchall()
cursor.close()
sentiment_analysis


# In[13]:


sentiment_analysis=pd.DataFrame(sentiment_analysis)


# In[14]:


sentiment_analysis.columns=['idtext','Idlog','tweet text','seintiment','sent dat','processed', 'AWSSentiment', 'id aws sentiment', 'idtext', 'sentiment', 'positive','negtive','neutrl','mix']


# In[15]:


sentiment_analysis


# In[16]:


sentiment_analysis.dropna()


# In[17]:


filter2 = sentiment_analysis.columns=["Idlog","sent dat","sentiment"]]
print(filter2)


# In[18]:


# import libraries
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# load dataset
sentyment = sns.load_dataset("tips")

# set plot style: grey grid in the background:
sns.set(style="darkgrid")

# set the figure size
plt.figure(figsize=(14, 14))

# top bar -> sum all values(smoker=No and smoker=Yes) to find y position of the bars
total = tips.groupby('day')['total_bill'].sum().reset_index()

# bar chart 1 -> top bars (group of 'smoker=No')
bar1 = sns.barplot(x="day",  y="total_bill", data=total, color='darkblue')

# bottom bar ->  take only smoker=Yes values from the data
smoker = tips[tips.smoker=='Yes']

# bar chart 2 -> bottom bars (group of 'smoker=Yes')
bar2 = sns.barplot(x="day", y="total_bill", data=smoker, estimator=sum, ci=None,  color='lightblue')

# add legend
top_bar = mpatches.Patch(color='darkblue', label='smoker = No')
bottom_bar = mpatches.Patch(color='lightblue', label='smoker = Yes')
plt.legend(handles=[top_bar, bottom_bar])

# show the graph
plt.show()


# In[19]:


df_filtered = sentiment_analysis.drop(columns=['idtext','Idlog','tweet text','seintiment','sent dat','processed', 'AWSSentiment', 'id aws sentiment', 'idtext', 'sentiment'])


# In[20]:


df_filtered.corr()


# In[21]:


plt.figure(figsize=(8, 6))
ax = sns.heatmap(df_filtered.corr(), vmin=-1, vmax=1, annot=True, cmap="YlGnBu", linewidths=.5);


# In[ ]:





# In[22]:


series = sentiment_analysis["seintiment"].to_numpy()


# In[23]:


sentiment_analysis.shape


# In[24]:


split_time = 1000

x_train = series[:split_time]
x_test = series[split_time:]


# In[25]:


plt.figure(figsize=(10,6))
plt.plot(x_test)
plt.xlabel("Time")
plt.ylabel("Value")
plt.grid(True)


# In[26]:


cursor = cnx.cursor()


# In[27]:


cursor.execute("SELECT* From db_BeduProject.tbl_TweetDetails")
tweets = cursor.fetchall()
cursor.close()


# In[28]:


tweets=pd.DataFrame(tweets)


# In[29]:


tweets.head()


# In[30]:


tweets.columns=['Id','IDTweet','Tuser','TText','AWSSentiment']


# In[31]:


tweets.head()


# In[40]:


tweets['count']=1


# In[ ]:





# In[41]:


tweets.head()


# In[42]:


agg_tweets = tweets.groupby(['Tuser', 'AWSSentiment'])['count'].sum().unstack().fillna(0)
agg_tweets


# In[63]:


col_names = list(agg_tweets.keys())
                 
agg_tweets['TOTAL'] = agg_tweets[col_names].sum(axis=1)
print(agg_tweets)


# In[62]:





# In[65]:


agg_tweets.sort_values(by=['TOTAL'], ascending=False, inplace=True)
print(agg_tweets)


# In[66]:


top20 = agg_tweets[0:19]
print(top20)


# In[72]:


from matplotlib import pyplot as plt

# Very simple one-liner using our agg_tips DataFrame.
agg_tweets[0:10].plot(kind='bar', stacked=True)

# Just add a title and rotate the x-axis labels to be horizontal.
plt.title('sentiment by top users')
plt.xticks(rotation=70, ha='center')


# In[ ]:


from matplotlib import pyplot as plt

fig, ax = plt.subplots()
# First plot the 'Male' bars for every day.
ax.bar(agg_tweets.index, agg_tweets['MIXED'], label='MIXED')
# Then plot the 'Female' bars on top, starting at the top of the 'Male'
# bars.
ax.bar(agg_tweets.index, agg_tweets['NEGATIVE'], bottom=agg_tweets['MIXED'],
       label='NEGATIVE')
ax.set_title('Tips by Day and Gender')
ax.legend()


# In[ ]:


cursor = cnx.cursor()


# In[ ]:


cursor.execute("SELECT db_BeduProject.tbl_TweetDetails.TUser, db_BeduProject.tbl_TweetDetails.IDTweet, db_BeduProject.tbl_RetweetsFav.Favs, db_BeduProject.tbl_RetweetsFav.Retweet FROM db_BeduProject.tbl_TweetDetails Right JOIN db_BeduProject.tbl_RetweetsFav  ON db_BeduProject.tbl_TweetDetails.IDTweet = db_BeduProject.tbl_RetweetsFav.idTweet")
usersfavrep = cursor.fetchall()
cursor.close()


# In[ ]:


usersfavrep=pd.DataFrame(usersfavrep)


# In[ ]:


usersfavrep.columns=['User','IDTweet','Favs','Retweet']


# In[ ]:


agg_usersfavrep = usersfavrep.groupby(['User', 'Favs'])['Retweet'].sum().unstack().fillna(0)
agg_usersfavrep

