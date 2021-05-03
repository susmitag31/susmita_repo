# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 12:24:28 2020

@author: ravikiran.sm
"""
import spacy
import sys
import os
nlp=spacy.load("en_core_web_lg")



def case(text):
    f = open("/home/ubuntu/FocusSeq2Seq-master/squad_test/dev.txt.shuffle.test.case", "w")
    temp=[]
    count=0
    for word in text.split():
      if count>0:  
        if word.isupper():
            temp.append('UP')
        else:
            temp.append('LOW')
      else:
         temp.append('UP')
      count=count+1    
    print("CASE: ", len(temp))  
    f.write(' '.join(temp) )  
    f.close
       


     

def pos(text):
    f = open("/home/ubuntu/FocusSeq2Seq-master/squad_test/dev.txt.shuffle.test.pos", "w")
    temp=[]
    doc=nlp(text)
    for t in doc:
        if t.pos_=='PUNCT':
            temp.append(t.text.upper())
        else:
            temp.append(t.tag_)
    print("POS: ",len(temp))        
    f.write(' '.join(temp) )        
    f.close



named_entities=['DATE',
 'DURATION',
 'LOCATION',
 'MISC',
 'MONEY',
 'NUMBER',
 'ORDINAL',
 'ORGANIZATION',
 'PERCENT',
 'PERSON',
 'TIME',
 'GPE'
 'CARDINAL',
 'ORG']

def ner(text):
    f = open("/home/ubuntu/FocusSeq2Seq-master/squad_test/dev.txt.shuffle.test.ner", "w")
    temp=[]
    doc=nlp(text)
    for t in doc.ents:
        temp=[]
        for i in range(len((t.text.split()))): 
            if t.label_=='ORG':
                temp.append('ORGANIZATION')
            elif t.label_=='CARDINAL':
                temp.append('NUMBER')
            else:
                if t.label_ in named_entities:
                    temp.append(t.label_)  
                else:
                    temp.append('O')

        text=text.replace(t.text,' '.join(temp))
    temp1=[]
    for word in text.split():
        if word in named_entities:
            temp1.append(word)
        else:
            temp1.append('O')
    print("NER: ",len(temp1))        
    f.write(' '.join(temp1))        
    f.close        
      
        
def source(text):
    f = open("/home/ubuntu/FocusSeq2Seq-master/squad_test/dev.txt.shuffle.test.source.txt", "w")
    print("SOURCE: ",len(text.split()))
    f.write(text)
    f.close()
    
def target(text):
    f = open("/home/ubuntu/FocusSeq2Seq-master/squad_test/dev.txt.shuffle.test.target.txt", "w")
    f.write(text)
    f.close()    



def BIO(text,bio):
    temp=[]
    f = open("/home/ubuntu/FocusSeq2Seq-master/squad_test/dev.txt.shuffle.test.bio", "w")
    text=text.replace(bio,'B',1)
    for i in text.split():
        if i!='B':
            temp.append('O')
        else:
            temp.append('B')
    print("BIO: ",len(temp))        
    f.write(' '.join(temp))
    f.close            
            
    


if __name__ == '__main__':
    text=sys.argv[1]
    text=text.replace("'s"," 's")
    text=text.replace("."," .")
    text=text.replace(","," ,")
    Bio=sys.argv[2]
    print(text)
   # print("NER: ",len(ner(text).split()))
   # print("POS: ",len(pos(text).split()))
   # print("CASE: ",len(case(text).split()))
   # print("SOURCE: ",len(source(text).split()))
   # print("TARGET: ",len(target(text).split()))
    ner(text)
    pos(text)
    case(text)
    source(text)
    target(text)
    BIO(text,Bio)
    os.system('python QG_data_loader1.py')
    os.system('python evaluate1.py --task=QG --model=NQG --load_glove=True --feature_rich --data=squad \
   --rnn=GRU --dec_hidden_size=512 --dropout=0.5 \
    --batch_size=64 --eval_batch_size=64 \
    --use_focus=True --n_mixture=3 --decoding=greedy \
    --load_ckpt=5 --eval_only')
  
    
    
    
    
    




