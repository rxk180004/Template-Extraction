# -*- coding: utf-8 -*-
"""
Created on Sun May  3 12:42:14 2020

@author: arams
"""
import sys
import glob
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize 
from nltk.stem import WordNetLemmatizer
import spacy.cli
import os
import string
from nltk.stem import PorterStemmer
import networkx as nx
import json
import codecs
from nltk.parse import CoreNLPParser
from nltk.parse.corenlp import CoreNLPDependencyParser
from nltk.wsd import lesk

nlp = spacy.load("en_core_web_sm")

# Extract the features from the given input file
def extractFeatures():
    stop_words = stopwords.words('english') + list(string.punctuation)
    file_loc='wikiTest/'
    os.chdir('/Users/ranjithreddykommidi/NLP/Project/wikiTest')
    file_names = glob.glob('*.txt')
    
    #Read every wikipedia articles given in the input fileList
    for file in file_names:
        readfile = open(file, 'r')
        text = readfile.read()
        corpus = {}
        sent_text = nltk.sent_tokenize(text)
        dep_parser = CoreNLPDependencyParser(url='http://localhost:9010')
        ner_tagger = CoreNLPParser(url='http://localhost:9010', tagtype='ner')
        count = 0
        for sentence in sent_text:
            tokenized_text = [i for i in nltk.word_tokenize(sentence.lower()) if i not in stop_words]  
            lemma = [WordNetLemmatizer().lemmatize(word) for word in tokenized_text]
            stemmed = [PorterStemmer().stem(word) for word in tokenized_text]
            tagged = nltk.pos_tag(tokenized_text)
            parse, = dep_parser.raw_parse(sentence)
            dependency_parse = list(parse.triples())
            tokenized_text_ner = nltk.word_tokenize(sentence) 
            try:
                ner_tag = ner_tagger.tag(tokenized_text_ner)
            except:
                ner_tag = ner_tagger.tag(tokenized_text)
            
            Synonym = []
            Hypernym = []
            Hyponym = []
            Meronym = []
            Holonym = []
            Heads = []
        
            for t in tokenized_text:
                Nyms = lesk(sentence, t)
                if Nyms is not None:
                    this_synonym = t
                    if Nyms.lemmas()[0].name() != t:this_synonym = Nyms.lemmas()[0].name()
                    Synonym.append(this_synonym)
                    if Nyms.hypernyms() != []:Hypernym.append(Nyms.hypernyms()[0].lemmas()[0].name())
                    if Nyms.hyponyms() != []:Hyponym.append(Nyms.hyponyms()[0].lemmas()[0].name())
                    if Nyms.part_meronyms() != []:Meronym.append(Nyms.part_meronyms()[0].lemmas()[0].name())
                    if Nyms.part_holonyms() != []:Holonym.append(Nyms.part_holonyms()[0].lemmas()[0].name())
                else:
                    Synonym.append(t)
        
            striped_sentence = sentence.strip(" '\"")
            if striped_sentence != "":
                dependency_parser = dep_parser.raw_parse(striped_sentence)
                parsetree = list(dependency_parser)[0]
                head_word = ""
                head_word = [k["word"]
                         for k in parsetree.nodes.values() if k["head"] == 0][0]
                if head_word != "":
                    Heads.append([head_word])
                else:
                    for i, pp in enumerate(tagged):
                        if pp.startswith("VB"):
                            Heads.append([tokenized_text[i]])
                            break
                    if head_word == "":
                        for i, pp in enumerate(tagged):
                            if pp.startswith("NN"):
                                Heads.append([tokenized_text[i]])
                                break
            else:
                Heads.append([""])

            count = count + 1
            corpus[count] = {}
            corpus[count]["sentence"] = {}
            corpus[count]["sentence"] = sentence
            corpus[count]["tokenized_text"] = {}
            corpus[count]["tokenized_text"] = tokenized_text
            corpus[count]["lemma"] = {}
            corpus[count]["lemma"] = lemma
            corpus[count]["stem"] = {}
            corpus[count]["stem"] = stemmed
            corpus[count]["tag"] = {}   
            corpus[count]["tag"] = tagged
            corpus[count]["dependency_parse"] = {}
            corpus[count]["dependency_parse"] = dependency_parse
            corpus[count]["synonyms"] = {}
            corpus[count]["synonyms"] = Synonym
            corpus[count]["hypernyms"] = {}
            corpus[count]["hypernyms"] = Hypernym
            corpus[count]["hyponyms"] = {}
            corpus[count]["hyponyms"] = Hyponym
            corpus[count]["meronyms"] = {}
            corpus[count]["meronyms"] = Meronym
            corpus[count]["holonyms"] = {}
            corpus[count]["holonyms"] = Holonym
            corpus[count]["ner_tag"] = {}
            corpus[count]["ner_tag"] = str(dict(ner_tag))
            corpus[count]["head_word"] = {}
            corpus[count]["head_word"] = Heads[0]
            corpus[count]["file_name"] = {}
            corpus[count]["file_name"] = file[len(file_loc):]

        outputName = file[len(file_loc)]        
        json_object = json.dumps(corpus, indent = 4) 
        with open(outputName, "w") as f:
            f.write(json_object)
            
# Retokenizing the doc 
def retokenizeEntities(doc):
    #Merge entities and noun chunks into one token
    '''ents = list(doc.ents) 
    nounChunks= list(doc.noun_chunks)
    spans = ents + nounChunks
    spans = filter_spans(spans)
    with doc.retokenize() as retokenizer:
        for span in spans:
            retokenizer.merge(span)'''
            
    doc=nlp(doc)        
    spans = list(doc.ents) + list(doc.noun_chunks)
    spans = filterSpans(spans)
    with doc.retokenize() as retokenizer:
        for span in spans:
            retokenizer.merge(span)
    return doc

# Filter a sequence of spans so they don't contain overlaps
# For spaCy 2.1.4+: this function is available as spacy.util.filter_spans()
def filterSpans(spans):
    get_sort_key = lambda span: (span.end - span.start, -span.start)
    sorted_spans = sorted(spans, key=get_sort_key, reverse=True)
    result = []
    seen_tokens = set()
    for span in sorted_spans:
        # Check for end - 1 here because boundaries are inclusive
        if span.start not in seen_tokens and span.end - 1 not in seen_tokens:
            result.append(span)
        seen_tokens.update(range(span.start, span.end))
    result = sorted(result, key=lambda span: span.start)
    return result

def conjuncts(pos,pos_final):
    for curr_pos in pos.conjuncts:
        pos_final.append(curr_pos) 
    return pos_final

def extract_workTemplate(sentence):
    original_sentence = sentence
    doc = retokenizeEntities(sentence)
    work_list = []     
    for per in filter(lambda w: w.ent_type_ == 'PERSON', doc):
        work = ()
        pos_final, org_final, loc_final = [],[],[]
        if per.dep_ == 'nsubj' and per.head.dep_ == 'root':
            pos = [w for w in per.head.lefts if w.dep_ == 'nsubj']
            pos_final.append(pos)
        elif per.dep_ == 'ROOT':
            if len(list(doc.sents)) > 1:
                root = list(doc.sents)[1].root
                pos = [w for w in root.rights if w.dep_ == 'attr'] 
                if pos:
                    pos = pos[0]
                    pos_final.append(pos)
                    pos_final = conjuncts(pos,pos_final)
        elif per.dep_ in ('nsubj'):
            pos = [w for w in per.head.rights if w.dep_ == 'attr']
            if pos:
                pos = pos[0]
                pos_final.append(pos)
                pos_final=conjuncts(pos,pos_final)
        for who in filter(lambda w: w.text.lower() == 'who'.lower(), doc):
            if who.dep_ == 'nsubj':
                who_prep = [w for w in who.head.rights if w.dep_ == 'prep' and w.text == 'as']
                if who_prep:
                    who_prep = who_prep[0]
                    pos = [w for w in who_prep.rights if w.dep_ == 'pobj']
                    if pos:
                        pos = pos[0]
                        pos_final.append(pos)
        for org in filter(lambda w: w.ent_type_ == 'ORG', doc):org_final.append(org)
        for loc in filter(lambda w: w.ent_type_ == 'GPE', doc):loc_final.append(loc)
        work = (per, list(set(org_final)), list(set(pos_final)), list(set(loc_final)))
        work_list.append(work)
    return populateWorkTemplate(work_list, original_sentence)  
'''
def display_dependency_parsing(sentence):
    en_nlp = spacy.load('en_core_web_sm')
    doc = en_nlp(sentence)

    def to_nltk_tree(node):
        if node.n_lefts + node.n_rights > 0:
            return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
        else:
            return node.orth_

    [to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]
    displacy.render(doc, style='dep')'''

def populateWorkTemplate(work_list, original_sentence):
    work_template = []
    for work in work_list:
        workTemp = {}
        for person, pos_list, org_list, loc_list in work_list:
            workTemp["1"] = person.text
            pos_str = ''
            for pos in pos_list:
                pos_str = pos_str + pos.text + ','
            workTemp["2"] = pos_str
            org_str = ''
            for org in org_list:
                org_str = org_str + org.text + ','
            workTemp["3"] = org_str
            loc_str = ''
            for loc in loc_list:
                loc_str = loc_str + loc.text + ','
            workTemp["4"] = loc_str
            
            if(workTemp != {}):
                work_template.append({"template":"WORK", "sentences": original_sentence, "arguments": workTemp})
    return work_template

def extractPartTemplate(sentence):
        part=[]
        doc = nlp(sentence)
        edges = []
        e_list={}
        graphNodes=[]
        tokens=[]
        outputTemplate = []
        
        #formning the edges for maintaing the relation between token and children
        for each_token in doc:
            for child in each_token.children:
                if(each_token.text not in tokens):        
                    tokens=tokens+[each_token.text]
                edges.append(('{0}'.format(each_token.lower_),
                      '{0}'.format(child.lower_)))
        
        #Forming the graphNodes with locations in the text and establishing the relations
        for ent in doc.ents:
            if(ent.text=="Richardson"):
                e_list[ent.text]='GPE'
                graphNodes = graphNodes+[ent.text]
            if(ent.label_=='GPE' and e_list.get(ent.text) is None):
                e_list[ent.text]=ent.label_
                graphNodes = graphNodes+[ent.text]
            
        # Forming the graph and digraph with the edges found
        graph = nx.Graph(edges)
        digraph = nx.DiGraph(edges)
        
        # For each nodes we iterate over all the nodes and find if their exists any relation between source and target
        for i in range(len(graphNodes)):
            for j in range(i+1,len(graphNodes)):
                node1 =graphNodes[i].lower()
                node2 =graphNodes[j].lower()
                
                if('in' in tokens):
                    if (nx.has_path(digraph, source='in', target = node1)):
                        if (nx.has_path(digraph, source='in', target = node2)):
                            rel = (node1,node2)
                            if rel not in part:
                                part = part+[rel]
                
                if('is' in tokens):
                    if (nx.has_path(digraph, source='is', target = node1)):
                        if (nx.has_path(digraph, source='is', target = node2)):
                            rel = (node1,node2)
                            if rel not in part:
                                part = part+[rel]
                                
                if('are' in tokens):
                    if(nx.has_path(digraph, source='are', target = node2)):
                        if (nx.has_path(digraph, source='are', target = node2)):
                            rel = (node1,node2)
                            if rel not in part:
                                part=part+[rel]
                                
                if((nx.has_path(graph, source = node1, target = node2))):
                    nodeInPath = nx.shortest_path(graph, source = node1, target = node2)
                    if('is' in nodeInPath and 'in' in nodeInPath):
                        rel = (node1,node2)
                        if rel not in part:
                            part = part + [rel]
                    if('is' in nodeInPath and 'of' in nodeInPath):
                        s=(node1,node2)
                        if s not in part:
                            part = part + [rel]
                    if('in' in nodeInPath and 'of' in nodeInPath):
                        s=(node1,node2)
                        if s not in part:
                            part = part + [rel]
                    if('in' in nodeInPath and 'of' not in nodeInPath):
                        s=(node1,node2)
                        if s not in part:
                            part = part + [rel]
        if(len(part) > 0):
            for i in range(len(part)):
                templateDict = {"template": "PART", "sentences": [], "arguments": {"1": "", "2": ""}}
                templateDict["sentences"].append(sentence)
                templateDict["arguments"]["1"] = part[i][0]
                templateDict["arguments"]["2"] = part[i][1]
                outputTemplate.append(templateDict)
        return(outputTemplate)
        
def extract_buyTemplate(doc):
    template = []
    modifiedDoc = retokenizeEntities(doc)
    def mergeTokenDependencies(item):
        for curr_item in item.conjuncts:
            Item_final.append(curr_item)
    for buy in filter(lambda w:w.dep_=='ROOT',doc):
        record = ()
        buyer_final=[]
        Item_final=[]
        Price_final=[]
        Quantity_final=[]
        Source_final=[]
        
        for token in modifiedDoc:
            if token.head.dep_ == "ROOT" and token.dep_ == "nsubj":
                buyer = token.text
                buyer_final.append(buyer)
            elif token.head.dep_ == "dobj" or token.dep_ == "dobj":
                item = token.text
                Item_final.append(item)
                mergeTokenDependencies(item)  
            elif token.ent_type_ == "MONEY":
                price = token.text
                Price_final.append(price)
            elif token.pos_ == "NUM" and token.ent_type_ != "DATE":
                quantity = token.text
                Quantity_final.append(quantity)
                
            elif token.dep_ == "pobj" and token.head.text == "from" and token.pos_ == "PROPN":
                form_prep = token.head
                if (form_prep.head == buy):
                    source = token.text
                    Source_final.append(source)
    record = [buyer, item, price, quantity, source]
    template.append(record)
    return template
        
def extract_buyTemplate2(doc):
    sp = spacy.load("en_core_web_md")
    word_doc = sp("buy")
    modifiedDoc = retokenizeEntities(doc)
    verbList = []
    for token in modifiedDoc:
        if(token.pos_ == "NOUN") and (token.ent_type_ == ""):
            sim = word_doc[0].similarity(sp(token.lemma_)[0])
            if sim >= 0.45:
                verbList.append(token)

        elif(token.pos_ == "VERB") and (token.ent_type_ == ""):
            sim = word_doc[0].similarity(sp(token.lemma_)[0])
            if sim >= 0.45:
                verbList.append(token)
    buyTemplate = []
    for work in verbList:
        buyer = None
        item = None
        price = None
        quantity = None
        source = None
        for token in modifiedDoc:
            if token.head == work and token.dep_ == "nsubj":
                buyer = token.text
            elif token.head == work and token.dep_ == "dobj":
                item = token.text
            elif token.ent_type_ == "MONEY":
                price = token.text
            elif token.pos_ == "NUM" and token.ent_type_ != "DATE":
                quantity = token.text
            elif token.dep_ == "pobj" and token.head.text == "from" and token.pos_ == "PROPN":
                form_prep = token.head
                if (form_prep.head == work):
                    source = token.text
        temp = [buyer, item, price, quantity, source]
        if(temp != {}):
            buyTemplate.append({"template":"BUY", "sentences": doc, "arguments": temp})
    return buyTemplate  
              
#Read Data from the Wikipedia Articles one by one and store in the fileData list

def getDataFromFiles(folderPath):
    fileData = []
    try:
        for file in os.listdir(folderPath):
            filePath = folderPath + '/' + file
            try:
                with open(filePath, 'r') as sourceFile:
                    rawData = sourceFile.read()
                    fileData.append({"sourceFileName": file, "inputData": rawData})
            except UnicodeDecodeError:
                sourceFile.close()
                with codecs.open(filePath, 'r', encoding='latin1', errors='ignore') as sourceFile:
                    raw_input = sourceFile.read().replace('\n', '')
                    fileData.append({"sourceFileName": file, "inputData": raw_input})
    except FileNotFoundError:
        return False
    else:
        return fileData


def templateExtraction(sourceFileName, doc):
    sents = sent_tokenize(doc)
    # split in to sentences
    templateExtractions = []
    
    for originalSentence in  sents:
        buyTemplate = extract_buyTemplate2(originalSentence)
        workTemplate = extract_workTemplate(originalSentence)
        partTemplate = extractPartTemplate(originalSentence)
        
        if len(buyTemplate) > 0:
           templateExtractions.extend(buyTemplate)
        if len(workTemplate) > 0:
            templateExtractions.extend(workTemplate)
        if len(partTemplate) > 0:
            templateExtractions.extend(partTemplate)
            
    template = []
    template.append({"document":sourceFileName, "extractions":templateExtractions})        
    return template

arg_list = sys.argv
folderPath = str(arg_list[1])
fileData = getDataFromFiles(folderPath)

#Process each file data
for file_data in fileData:
    inputData = file_data["inputData"]
    # For each file we extract the templates("BUY", "WORK", "PART")
    template = templateExtraction(file_data["sourceFileName"], inputData)
    if template != None:
        print(template)
        outputFileName = file_data["sourceFileName"].split(".")[0] + ".json"
        json_object = json.dumps(template, indent = 4) 
        with open(outputFileName, "w") as f:
            f.write(json_object)
            
            
