# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np
from googletrans import Translator
import swifter
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import re

translator = Translator(service_urls=['translate.googleapis.com'])


# Load the Random Forest CLassifier model

app = Flask(__name__)

import pandas as pd

from numpy import dot
from numpy.linalg import norm
from transformers import AutoTokenizer, AutoModel
from time import perf_counter

# =============================================================================
# nlp = spacy.load('en_core_web_lg')
# 
# =============================================================================

sci_bert_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
sci_bert_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')


@app.route('/')
def home():
    dd = pd.read_csv(r"C:/Users/I568791/OneDrive - SAP SE/Desktop/Hackathons/Bodyparts.csv")
   
    calculations=dd['Body Part'].values.tolist()
    calculations = list(set(calculations))
    return render_template('main.html',calculations=calculations)


@app.route('/predict', methods=['GET','POST'])

def predict():
    if request.method == 'POST':

    
            # body =  request.form.getlist("calculations")     
            sym1,sym2,sym3,age,bod,loc = request.form.get('Symptom1'),request.form.get('Symptom2'),request.form.get('Symptom3'),int(request.form.get('age')),request.form.get('calculations'),request.form.get('Location')      
         
            symm = [sym1,sym2,sym3]
          
            dd = pd.read_csv(r"C:/Users/I568791/OneDrive - SAP SE/Desktop/Hackathons/womenhealth1.csv",encoding='latin')
            
            
            dd['LBAge'] = dd['LBAge'].astype(int)
            dd['UBAge'] = dd['UBAge'].astype(int)
            dd1 = dd[(dd['LBAge']<=age)&(dd['UBAge']>=age)] 
            
            t1 = perf_counter()
            
            pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
            symm = [pattern.sub('', i) for i in symm]
            t2 = perf_counter()
            print(t2-t1)
                        
           
# =============================================================================
#             def mod(col):
#                 p =[i.strip() for i in col]
#                 return p
#             
#             dd1['Body Part'] = dd1['Body Part'].swifter.apply(mod)
#             
# =============================================================================
                                  
            # flaglist = [i   for i in range(len(dd1))   if bod in dd1['Body Part'].iloc[i]]
            dd1["Body Part"] = dd1["Body Part"].astype(str)
            dd2 = dd1[dd1["Body Part"].str.contains(bod, regex=True)]
            
            
            dd2['Symptoms1'] = dd2['Symptoms1'].str.strip('[]').str.split(',')
            
            dd2['Link']=dd2['Link'].astype(str) + loc
            t15 = perf_counter()    
            
            
            
            def get_bert_based_similarity(sentence_pairs, model, tokenizer):
                """
                computes the embeddings of each sentence and its similarity with its corresponding pair
                Args:
                    sentence_pairs(dict): dictionary of lists with the similarity type as key and a list of two sentences as value
                    model: the language model
                    tokenizer: the tokenizer to consider for the computation
                
                Returns:
                    similarities(dict): dictionary with similarity type as key and the similarity measure as value
                """
                similarities = dict()
                for sim_type, sent_pair in sentence_pairs.items():
                    inputs_1 = tokenizer(sent_pair[0], return_tensors='pt')
                    inputs_2 = tokenizer(sent_pair[1], return_tensors='pt')
                    sent_1_embed = np.mean(model(**inputs_1).last_hidden_state[0].detach().numpy(), axis=0)
                    sent_2_embed = np.mean(model(**inputs_2).last_hidden_state[0].detach().numpy(), axis=0)
                    similarities[sim_type] = dot(sent_1_embed, sent_2_embed)/(norm(sent_1_embed)* norm(sent_2_embed))
                return list(similarities.values())[0]
                                    
            
            
            
            def sigmoid(x,y):
              return get_bert_based_similarity({'similar':[x,y]}, sci_bert_model, sci_bert_tokenizer)
            
            # define vectorized sigmoid
            sigmoid_v = np.vectorize(sigmoid)
            
            
            symm = [i for i in symm if len(i)>0]
            d2 = dd2.copy()
                    
            mk = [[sigmoid_v(np.array(symm),i) for i in it] for it in d2['Symptoms1'].values.tolist()]
            wts =np.array([0.8,0.15,0.05])
            
            k = [[np.multiply(i,wts[:len(symm)]) for i in s] for s in mk]
            k1 = [[np.sum(i)/np.sum(wts[:len(symm)]) for i in s] for s in k]
            sk = [np.max(i) for i in k1]
            d2['Score'] = sk   
            
                   

            d2.sort_values(by='Score',ascending=False,inplace=True)
               
            dk = d2.iloc[:5,:]

           
           

            dk['Doctors'] ='Doctors Near You'
            def make_clickable(url, name):
                return '<a href="{}" rel="noopener noreferrer" target="_blank">{}</a>'.format(url,name)
            
           
            dk['Link'] = dk.swifter.apply(lambda x: make_clickable(x['Link'], x['Doctors']), axis=1)
      
            
            dk.drop_duplicates(subset='Conditions',keep='first',inplace=True)
            dk.drop(['Doctors','Symptoms1'],axis=1,inplace=True)
            
            def conditions(x):
                if x >= dk['Score'].quantile(0.75):
                    return "Moderate"
                elif x < dk['Score'].quantile(0.75) and x >= dk['Score'].quantile(0.25) :
                    return "Fair"
                else:
                    return "Low"
                
                       
            dk['Chances']=dk['Score'].swifter.apply(conditions)                       
            
            dk.drop('Score',axis=1,inplace=True)            
                 
            dk.rename(columns={'Link':'Doctors'},inplace=True)
            dk = dk[['Conditions','Symptoms','Chances','Doctors']]
                                  
            return render_template('result.html', tables=[dk.to_html(classes='table table-striped table-hover text-center', header=True, justify='center',index=False,escape=False).replace('<th>','<th style = "background-color: #191970;color: white;text-align:center !important">')], titles=[''])
        
        

if __name__ == '__main__':
	app.run()