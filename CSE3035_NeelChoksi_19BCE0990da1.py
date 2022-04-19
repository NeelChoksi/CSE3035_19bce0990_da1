#!/usr/bin/env python
# coding: utf-8

# # Getting data from facebook Graph API

# In[47]:


get_ipython().system('pip install facebook-sdk --quiet')


# In[62]:


import json
import facebook

def main():
    token = {"EAAOpoXPN6PIBAGyF8LO9nftNjvXQiwe6zbj99RFUcWFBJhD7SwrMQkvZBzVKSQHsqYT59TbrKWJlECP3dWgrr1j1u4jOay7SYHQWmWxMAg6xNWUknOpydVvcuj1kqySqscZCb8ZCZB7ZAWIhXmve8V4ZB3DgyACMkNwMZA9QJia7QhAxBdwZCOdfrPz3ygqZAHlxUgmQAatm8FAdYgJ5PaIrBev50hUZChogrSow93ldF1EiIj3FLcrKE8ZARAH5ht5ShQZD"}
    graph = facebook.GraphAPI(token)
    
    fields = ['id','name','posts']
    profile = graph.get_object("me",fields = fields)
    print(type(profile))
    all_data = json.dumps(profile,indent=4)
    print(json.dumps(profile,indent=4))
    return profile
    


# In[63]:


all_data = main()


# In[64]:


type(all_data)


# In[74]:


json_data = json.dumps(all_data,indent=4)
print(type(json_data))
# print(json_data)
json_data_parsable = json.loads(json_data)
all_posts = json_data_parsable["posts"]["data"]
print(type(all_posts[0]))
print(all_posts)
posts_fetched = []
for i in all_posts:
    posts_fetched.append(i["message"])


# # Sentiment analysis on my posts using Feed Forward Neural Network

# In[29]:


import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm,tqdm_notebook


# In[11]:


# DATA_SET = './datasets/aclimdb.csv'
DATA_SET = './datasets/IMDBDataset.csv'
test =pd.read_csv(DATA_SET).sample(5)
test


# In[12]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
sentiment_label = le.fit_transform(test["sentiment"])
print(le.classes_)
test_new =test.drop("sentiment",axis="columns")
test_new["sentiment"] = sentiment_label
test_new
# negative - 0 , positive - 1


# In[13]:


test_vectorizer = CountVectorizer(stop_words='english', max_df = 0.99 , min_df = 0.005)
test_seq = test_vectorizer.fit_transform(test_new['review'].tolist())
#print(test_seq)


# In[14]:


class Sequences(Dataset):
    def __init__(self,path):
        df = pd.read_csv(path)
        le = preprocessing.LabelEncoder()
        sentiment_labels = le.fit_transform(df["sentiment"])
        #print(le.classes_)
        df =df.drop("sentiment",axis="columns")
        df["sentiment"] = sentiment_labels

        self.vectorizer = CountVectorizer(stop_words='english', max_df = 0.99 , min_df = 0.005)
        self.sequences = self.vectorizer.fit_transform(df.review.tolist())
        self.labels = df.sentiment.tolist()
        self.token2idx = self.vectorizer.vocabulary_
        self.idx2token = {idx: token for token , idx in self.token2idx.items()}
    def __getitem__(self,i):
        return self.sequences[i, :].toarray(), self.labels[i]
    def __len__(self):
        return self.sequences.shape[0]        


# In[15]:


trial_db = Sequences(DATA_SET)
train_loader = DataLoader(trial_db, batch_size = 4096)
#print(trial_db.vectorizer)
# print(trial_db.sequences)
#print(trial_db.token2idx)


# In[16]:


print(trial_db.sequences)


# In[17]:


print(trial_db.sequences[0].shape)


# In[18]:


print(len(trial_db.labels))
print(trial_db.labels)


# In[19]:


print(len(trial_db.token2idx))
print(trial_db.token2idx)


# In[20]:


print(len(trial_db.idx2token))
print(trial_db.idx2token)


# In[21]:


print(trial_db[5][0].shape)


# In[22]:


# feed forward 1 : bow neural network 
class BoWFF1(nn.Module):
    def __init__(self,vocab_size,hidden1,hidden2):
        super(BoWFF1,self).__init__()
        self.fc1 = nn.Linear(vocab_size,hidden1)
        self.fc2 = nn.Linear(hidden1,hidden2)
        self.fc3 = nn.Linear(hidden2,1)
        
    def forward(self,inputs):
        x = F.relu(self.fc1(inputs.squeeze(1).float()))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# In[23]:


model_ff_bow = BoWFF1(len(trial_db.idx2token),128,64)
model_ff_bow


# In[24]:


criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam([p for p in model_ff_bow.parameters() if p.requires_grad],lr=0.001)


# In[30]:


model_ff_bow.train()
train_losses = [] 
for epoch in range(10):
    progress_bar = tqdm_notebook(train_loader,leave=False)
    losses = []
    total = 0
    for inputs,target in progress_bar:
        model_ff_bow.zero_grad()
        
        output = model_ff_bow(inputs)
        loss = criterion(output.squeeze(), target.float())
        
        loss.backward()
              
        nn.utils.clip_grad_norm_(model_ff_bow.parameters(), 3)

        optimizer.step()
        
        progress_bar.set_description(f'Loss: {loss.item():.3f}')
        
        losses.append(loss.item())
        total += 1
    
    epoch_loss = sum(losses) / total
    train_losses.append(epoch_loss)
        
    tqdm.write(f'Epoch #{epoch + 1}\tTrain Loss: {epoch_loss:.3f}')


# In[42]:


def predict_sentiment(text):
    model_ff_bow.eval()
    with torch.no_grad():
        test_vector = torch.LongTensor(trial_db.vectorizer.transform([text]).toarray())

        output = model_ff_bow(test_vector)
        prediction = torch.sigmoid(output).item()

        if prediction > 0.5:
            return [f'{prediction:0.3}','Positive sentiment']
        else:
            return [f'{prediction:0.3}','Negative sentiment']
            


# In[75]:


posts_fetched


# In[76]:


results=[]
for i in posts_fetched :
    results.append([i,predict_sentiment(i)[0],predict_sentiment(i)[1]])


# In[77]:


results


# In[78]:


results_df = pd.DataFrame(results,columns=['Post','Prediction Confidence','Sentiment'])


# In[79]:


results_df


# In[ ]:




