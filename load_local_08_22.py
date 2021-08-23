# Import , # load norm, voc_50k

#import matplotlib
#matplotlib.use('Agg')
#import pylab
#import warnings
#warnings.simplefilter('ignore')
#from adjustText import adjust_text

from sklearn import preprocessing;import heapq;
import gzip,re,time,pickle,sys,os,collections,string,itertools
from scipy import sparse; import pandas as pd; import numpy as np; 
#from nltk.corpus import stopwords;from nltk.stem.wordnet import WordNetLemmatizer
#cachedStopWords = stopwords.words('english');
from collections import defaultdict;from string import punctuation;from matplotlib import pyplot as plt
import community
import networkx as nx;import scipy.stats
import numpy, scipy.sparse
#from sparsesvd import sparsesvd
import sqlite3 as lite
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
from sklearn.metrics.pairwise import cosine_similarity
import json
import itertools

global dirName
dirName = os.path.dirname(os.path.realpath(__file__))
global adjust_path
adjust_path = '/../'
save_path = '/../D3_html_code/SampleData/'
#print (dirName)


# Get norm
#norm = pd.read_csv(dirName + '/../data/norm_tidy_data.gz')
#norm = norm.set_index('words',drop=False)


voc_50k          = pd.read_pickle(dirName + adjust_path+'data/vocabulary.pkl')
sum_year         = pd.read_pickle(dirName + adjust_path+'data/sum_year.pkl')
kernel_year_freq = pd.read_pickle(dirName + adjust_path+'data/year_count.pkl')


suffix = '_svd_PPMI.npy'
svd_vector = []
for year_i in range(1800,2000,10):
    svd = np.load(dirName + adjust_path + 'data/embeddings_10years/'+str(year_i)+suffix)
    svd_vector.append(svd)

# load data
v = np.array(pd.read_csv(dirName + adjust_path + 'data/linguistic_property/valence.csv'))
a = np.array(pd.read_csv(dirName + adjust_path + 'data/linguistic_property/arousal.csv'))
c = np.array(pd.read_csv(dirName + adjust_path + 'data/linguistic_property/concreteness.csv'))

errorFrequency = "The frequency of the word in the chosen year is too small to perform this analysis. Please refer to the Frequency Figure to change the year for analysis"

def checkFrequency (wordT,year_i): # if temp < 30, only display return error message
    temp = kernel_year_freq[voc.index(wordT),][year_i-1800]
    if temp < 30:
        print (errorFrequency) 

def load_sparse_csr(filename):
    loader = np.load(filename)
    return sparse.csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])
    
def procrustes_align(base_embed, other_embed):
    #""" 
    #    Align other embedding to base embeddings via Procrustes.
    #    Returns best distance-preserving aligned version of other_embed
    #    NOTE: Assumes indices are aligned
    #"""
    basevecs = base_embed - base_embed.mean(0)
    othervecs = other_embed - other_embed.mean(0)
    m = othervecs.T.dot(basevecs)
    u, _, v = np.linalg.svd(m) 
    ortho = u.dot(v)
    fixedvecs = othervecs.dot(ortho)
    #fixedvecs = preprocessing.normalize(fixedvecs)
    return fixedvecs
        
def load_embeddings(method):
    embeddings = []
    year_c = list(range(1800,2000,10))
    for year_i in year_c:
        if method == 'svd':
            m =  np.load(dirName + adjust_path + '/data/embeddings_10years/'+str(year_i)+suffix)
            m = preprocessing.normalize(m)
            
        elif method =='sgns':
            m  = np.load(dirName + adjust_path + '/data/0sgns_hamilton/'+ str(year_i) + '-w.npy')
            
        embeddings.append(m)
    return embeddings
 

 ################################# Below are functions required to visualize plot###################################################
################################## ---------- Generate data to D3   --------     ###################################################
# Change of context
# It return a tupe consisting of two lists. One is word, the other is change. 



def plot_co_occurence_add(wordT,contextW,year_s= 1800, year_e = 2009, normalize=True):
    file = dirName + adjust_path + '/data/50kMatrix_front/'+ wordT[:2] +'/'+str(voc_50k.index(wordT))+'.npz'
    
    temp = load_sparse_csr(file)
    contextW = contextW.split()
    cor_occur = []
    for w in contextW:
        cor_occur.append(temp.getcol(voc_50k.index(w)))
        
    cor_occur = sparse.hstack(cor_occur).todense()
    cor_occur = cor_occur[(1800-1700):,:]
    if normalize == False:
        cor_occur = cor_occur/sum_year[:,None]
    else:
        cor_occur = cor_occur/kernel_year_freq[voc_50k.index(wordT),][:,None]
    
    cor_occur = cor_occur.sum(axis=1)
    cor_occur = cor_occur.flatten().tolist()[0]

    # subset according to year input
    cor_occur = cor_occur[(year_s-1800):(year_e-1800)]
    outputD = pd.DataFrame({'date':range(year_s,year_e),'frequency':cor_occur })
    outputD.to_csv(dirName + save_path +  'co_occurence_add.csv',index= False)
    #return outputD


################### CODE BELOW IS READY ############################################################################
# closest, returning k-nearest word [three lists: word index, word, cosine similarity]
# Synonym-based analysis 
def closest ( target_w, year_i = 2000, k = 10):
    # Explain hyper-parameter
    # target_w: a list of words separated by space
    # year_i  : select year from 1800 to 2000
    # k       : number of k-nearest neighbors to display
    target_w = target_w.split()

    m = np.load(dirName + adjust_path + 'data/embeddings_10years/'+ str(year_i) +suffix)  
    voc = np.load(dirName + adjust_path + 'data/vocabulary.pkl',allow_pickle=True)
    m = preprocessing.normalize(m)
    
    raw_score, score, closeIndex, closeWords, closeScore = [],[],[],[],[]
    for w in target_w:
        raw_score_i = m.dot(m[voc.index(w),:])
        score_i = heapq.nlargest(k, zip(raw_score_i,range(0,m.shape[0])))
        closeIndex_i = [ x[1] for x in score_i]
        closeWords_i = [voc[x] for x in closeIndex_i]
        closeScore_i = [x[0] for x in score_i]
        
        raw_score.append(raw_score_i)
        score.append(score_i)
        closeIndex.append(closeIndex_i)
        closeWords.append(closeWords_i)
        closeScore.append(closeScore_i)
    
    #return closeIndex, closeWords,closeScore
    # print out closeWords to users

    file = open(dirName + save_path +'identify_synonym.txt','w') 
    file.write(' '.join(list(itertools.chain.from_iterable(closeWords)))) 
    file.close()

    

    temp = pd.DataFrame(raw_score[0])
    temp.columns=['value']
    temp = temp[temp.value>0]
    temp.to_csv(dirName + save_path +  'similarity_dist.csv',index= False)

    return closeIndex, closeWords,closeScore,raw_score

    # [raw_score[voc.index(v)] for v in v.split()] 
    # example of v: risk happy sad

def plot_synonym_structure (target_w, year_i = 2000, k =5,minSim = 0.72): # remove this function 
    # Explain hyper-parameters
    '''
    year_i : which year we are looking at 
    k: number of synonyms of each target word is included in the structure 
    minSim: link if semantic similarity is larger than minSIm
    
    '''
    target_w = target_w.split()
    w_combined = []
    for i in range(len(target_w)):
        w_combined.append(closest(target_w = target_w[i],year_i = year_i, k=k)[1][0])
    w_combined=sum(w_combined,[])
    #print (w_combined)
    
    m = np.load(dirName + adjust_path + 'data/embeddings_10years/'+ str(year_i) +suffix)  
    voc = np.load(dirName + adjust_path + 'data/vocabulary.pkl',allow_pickle=True)
    m = preprocessing.normalize(m)
    
    
    tempInd = [voc.index(w) for w in w_combined]
    word_embeddings = m[tempInd,:]


    # construct similarity table, prepare for network plot
    convert_index = {key:value for key,value in zip(range(0,len(w_combined)),w_combined)} 

    sim_matrix = np.einsum('xj,yj->xy', word_embeddings, word_embeddings)
    sim_martix = np.triu(sim_matrix,k=1)
    coo = scipy.sparse.coo_matrix(sim_matrix)
    coo = scipy.sparse.triu(coo,k=1)

    coo = pd.DataFrame({'word1': coo.row, 'word2': coo.col, 'value': coo.data}
                         )[['word1', 'word2', 'value']].sort_values(['word1', 'word2']
                         )

    coo.word1 = [convert_index.get(i) for i in coo.word1]
    coo.word2 = [convert_index.get(i) for i in coo.word2]
    network_df=coo.copy()

    #x=plt.hist(network_df.value)
    # select threshold of similarity value
    network_df=network_df[network_df.value >= minSim ]
    
    
    ######### Plot 
    
    #import random
    #random.seed(100)
    G=nx.Graph() 

    # weight by similarity
    addWeight = []
    for i in range(network_df.shape[0]):
        addWeight.append((network_df.word1.iloc[i],network_df.word2.iloc[i],network_df.value.iloc[i]))

    G.add_weighted_edges_from(addWeight)

    # don't remove any nodes
    remove = [node for node, degree in G.degree() if degree < D]
    
    G.remove_nodes_from(remove)
    K=0.2

    # Find modularity
    part = community.best_partition(G)
    mod = community.modularity(part,G)
    nx.set_node_attributes(G,'modularity',part) # add modularity of each nodes to network
    #nx.get_node_attributes(G,'modularity')
    color_values = [part.get(node) for node in G.nodes()] # set colors 

    #tempInd = [voc_50k.index(w) for w in G.nodes()]
    weights = np.array([G[u][v]['weight'] for u,v in G.edges()])
    #weights = ((weights+2)**10)/1000

    '''
    plt.figure(figsize=(40,30))
    selectPosPatten = nx.spring_layout(G,iterations=100,weight = 'weight',scale=900,k = K)  #k controls the distance between the nodes and varies between 0 and 1; default k=0.1
    nx.draw_networkx_nodes(G,pos = selectPosPatten, cmap = plt.cm.plasma,node_color = color_values ,alpha=0.55,
                           #node_size = [v*50 for v in nx.degree(G).values()]) 
                           node_size = [v*200 for v in nx.degree(G).values()]) 

    nx.draw_networkx_edges(G,pos = selectPosPatten, camp = plt.cm.plasma, edge_color = 'grey', width = weights)

    x=nx.draw_networkx_labels(G,pos = selectPosPatten, font_size=45, font_family='sans-serif')
    plt.savefig('../#6_figure/emotion_'+' '.join(target_w)+'.jpg' )
    '''
    nodes_json = pd.DataFrame({'id':G.nodes(),'group':color_values}).to_json(orient = 'records')
    links_json = pd.DataFrame({'source':[x[0] for x in G.edges()],'target': [x[1] for x in G.edges()],'weight':weights}).to_json(orient = 'records')
    
    json_file = '{"nodes":'+ nodes_json + ',' + '"links":' +links_json + '}'
    
    # save json file 
    file = open(dirName + save_path + 'synonym_structure_network.json','w') 
    file.write(json_file) 
    file.close()

    return json_file
    
def plot_semantic_drift_path(wordT,wordCompare, year_s = 1850, year_e = 2000, interval=40,k=15,components=2,reduce='pca',method = 'svd',mirror=False,align=True,size=23,addinBetween=True):
    #print (method)
    '''  
    Explain Hyper-Parameter
    Doesn't allow customisation:
    WordT:             Take either one word or several words, separated by space (etc: risk danger hazard). Only the first word will be showed path. 
    year_s:            start year
    year_e:            end year
    interval:          if you want to show path of semantic drift, interval regulate number of time points in between of starting and end year. 
    k:                 number of k-nearest neighbors included for each word
    size=23:           size of fonts
    addinBetween =     True: True one wants to see a path, False if only want to visualize the two ends 

    Doesn't allow customisation:
    components: FIXED parameter in PCA analysis
    reduce:     pca method. Also fixed. t-sne doesn't work well here
    method:     svd. Fixed. Alternatively one can choose word2vec trained by Hamilton 2016. We may provide that feature in the future
    mirror:     aestetics if one wants the image to be mirrored, so that path flows from left to right
    align=True: fixed, esepcially if one wants a historical drift. 
    

    '''
    wordCompare = wordCompare.split()
    wordT = wordT.split()
    wordT = wordT + wordCompare
    '''
    ####### Print error message
    notinWord = ''
    for i in wordT:
        if i not in voc_50k:
            notinWord = notinWord+' '+i
    if len(notinWord) !=0:
        print('Error: The following words are not found in our vocabulary list: {}.'.format(notinWord))
        exit()
    if year_s >2000 or year_s <1800 or year_e >2000 or year_e <1800:
        print ('Error: please set the year between 1800 and 2000 (inclusive)')
        exit()
    if len(wordT) >6:
        print ('Error: Please limit number of words within 5')
        exit()
    if interval % 10 != 0 or year_s % 10 !=0 or year_e % 10 !=0:
        print ('Error: Interval and year have to be multiples of 10')
        exit()
    elif interval > year_e - year_s:
        print ('Error: Interval is too large')
        exit()
    elif year_s >= year_e:
        print ('Error: The starting year has to be earlier than ending year.')
        exit()
    if k >40:
        print ('Error: Please set k less than 40')
        exit()
    '''

    year_i = list(np.arange(year_s, year_e, interval))
    if year_i[-1] != year_e:
        year_i.append(year_e)
        
    year_i = np.sort(year_i)
    
    year_old = year_i[0]
    year_new = year_i[-1]

    '''

    for w in wordT:
        if kernel_year_freq[voc_50k.index(wordT),year_old-1800] == 0:
            print ('Warning: The frequencies of following words are too low in {}: {}'.format(year_old, w))
            exit()
    '''

    year_continues = year_i[1:-1]
    title = wordT
    
    mainW = wordT[0]
    
    #closest ( target_w,year_i=2000, k = 10,method='svd')
    #2. get k-nearest neighbor words as contextW, prepare their colour
    old,new = [],[]
    for w in wordT:
        old = old + closest(target_w = w, year_i = year_old, k=k)[1][0][1:]
        new = new + closest(target_w = w, year_i = year_new, k=k)[1][0][1:]
    #print (old)
    #print ('-------')
    #print (new)
    contextW = set(old + new)
    contextW = [w for w in contextW if w not in wordT]
    color_w = ['blue']*len(contextW)
    size_w = [20]*len(contextW)
    alpha_w = [0.5]*len(contextW)
    contextW = ' '.join(contextW )
    

    
    
    #3. load wordembeddings based on vector. normalize svd. sgns does not need normalized. 
    if method == 'sgns':
        m     = np.load(dirName + adjust_path + 'data/0sgns_hamilton/'+ str(year_new) + '-w.npy')
        m_old = np.load(dirName + adjust_path + 'data/0sgns_hamilton/'+ str(year_old) + '-w.npy')
        voc = np.load(dirName + adjust_path + 'data/0sgns_hamilton/'+ str(year_new) + '-vocab.pkl')
    elif method == 'svd':
        m = np.load(dirName + adjust_path + 'data/embeddings_10years/'+ str(year_new) + suffix) ; m = preprocessing.normalize(m)
        m_old = np.load(dirName + adjust_path + 'data/embeddings_10years/'+ str(year_old) + suffix) ;m_old = preprocessing.normalize(m_old)
        voc = np.load(dirName + adjust_path + 'data/vocabulary.pkl',allow_pickle=True)
    
    # 4. historical alignment
    if align:
        m_old = procrustes_align(m,m_old)    
    
    # 5. decompose target word(wordT) --> 'gay gays' --> ['gay_1900', 'gays_1900', 'gay_1800', 'gays_1800']
    
    wordTIndex = [voc.index(w) for w in wordT] #index for gay and gays
    wordT_annotation = sum([[w + '_' + x for w in wordT] for x in [str(year) for year in [year_new,year_old]] ],[])
    contextW = contextW.split()
    compiledW = contextW + wordT 
    compiledIndex = [voc.index(w) for w in compiledW]  
    
    #  targetW = [voc.index(T) for T in wordT]
    #  indexCW = [voc.index(w) for w in contextW]
    #  indexCW = targetW + indexCW
    
    compiledM    = m[compiledIndex,:]   # get mordern embeddings
    compiledM_old = m_old[wordTIndex,:] # get anxient embeddings for gay and gays /gay_1800,gays_1800   
    compiledM = np.vstack([compiledM,compiledM_old])# 顺序： context_1900+.......+gay_new1900,gays_new1900,gay_old1800,gays_1800.
    annotation = contextW  + wordT_annotation       # 顺序：一致
    color_w =   color_w + ['red']*len(wordT_annotation)             # 绿色*len(wordT_annotation)
    size_w = size_w + [100]*len(wordT_annotation)
    alpha_w = alpha_w + [1]*len(wordT_annotation)

    betweenWordT = []
    if addinBetween == True:
        if method =='sgns':
            for year_c in year_continues:
                m_b = np.load(dirName + adjust_path + 'data/0sgns_hamilton/'+ str(year_new) + '-w.npy')
                voc = np.load(dirName + adjust_path + 'data/0sgns_hamilton/'+ str(year_new) + '-vocab.pkl')
                if align:
                    m_b = procrustes_align(m,m_b)  
                betweenWordT.append(m_b[voc.index(mainW),:])
        elif method == 'svd':
            for year_c in year_continues:
                m_b = np.load(dirName + adjust_path + 'data/embeddings_10years/'+ str(year_c) + suffix) ; m_b = preprocessing.normalize(m_b)
                voc = np.load(dirName + adjust_path + 'data/vocabulary.pkl',allow_pickle=True)
                if align:
                    m_b = procrustes_align(m,m_b)  
                betweenWordT.append(m_b[voc.index(mainW),:])
        
        betweenWordT = np.vstack(betweenWordT)
        #annotation_between = [mainW + '_' + x for x in [str(year) for year in year_continues] ]
        annotation_between = ['' + '' + x for x in [str(year) for year in year_continues] ]
        compiledM = np.vstack([compiledM,betweenWordT])
        annotation = annotation + annotation_between

        color_w = color_w + ['red']*len(annotation_between)
        size_w = size_w + [20]*len(annotation_between)
        alpha_w = alpha_w + [1]*len(annotation_between)
    
        
    #print (len(annotation))
    #print (len(compiledW))
    #print (compiledM.shape)    
    
    
    if reduce =='pca':
        chosen = PCA(n_components = components).fit_transform(compiledM)
    elif reduce == 't-sne':
        pca_30 = PCA(n_components = 30).fit_transform(compiledM)
        chosen = TSNE(n_components = components,init='pca',verbose=1, perplexity=40,n_iter=500,learning_rate=30).fit_transform(pca_30)
    
    if mirror:
        chosen[:,0] = -1*chosen[:,0]

    texts = []
    for x, y, l,alpha in zip(chosen[:,0], chosen[:,1], annotation,alpha_w):
        texts.append(plt.text(x, y, l, size = size,alpha=alpha))

    # plot
    #print (chosen.shape)
    
    #for i, word in enumerate(annotation):
    #    x = chosen[:,0][i]
    #    y = chosen[:,1][i]
    #    pylab.scatter(x,y,marker='o',c=color_w[i],s=size_w[i])
    #    #pylab.annotate(word,xy=(x,y))
    #plt.title(str(adjust_text(texts, arrowprops=dict(arrowstyle="-", color='k', lw=1)))+' iterations')
    #plt.title(title,fontsize = size/2)

    outputD = pd.DataFrame({'label':annotation, 'xValue':chosen[:,0], 'yValue':chosen[:,1], 'color':color_w, 'size': size_w, 'alpha': alpha_w} )
    outputD.to_csv(dirName + save_path + 'semanticDrift.csv')
    return outputD



def plotNetwork(word, year_i=1990, C = 30, nPMI=300, cap=90,
               weight_by = 'PMI', nodesizeoption ='cor', P = 3, D=4, mustInclude_topN_PMI=15):
    ##### Step 1: Load data | takes maximum 2.5 s 
    word=word.lower()
    voc=voc_50k # It read voc_50k from  vocabulary.pkl

    index = voc.index(word)
    # read the co-occurence martrix of the year
    fileName = dirName + adjust_path + 'data/compiled_coMatrix/' + str(year_i) + '.npz'
    tempData = load_sparse_csr(fileName)

    front = tempData.getrow(index).toarray()
    back  = tempData.getcol(index).toarray().transpose()
    combined_i = front + back 


    combined_i = combined_i[0][0:45000] 
    year_i_freq = kernel_year_freq[:,year_i-1800] # frequency of context words
    word_freq = year_i_freq[index]                # frequency of cue word 
    lexicon_size_year_i = sum_year[str(year_i)]   # lexicon size of the year
    

    ##### Step 2: Select top **N** words associated with word based on PMI

    # compute smoothed PPMI for all context words
    contextFreq = year_i_freq[:45000]/lexicon_size_year_i

    top  = combined_i[:45000]/lexicon_size_year_i
    base = (word_freq/lexicon_size_year_i)*contextFreq

    PMI = np.log( top / base )
    #PMI = np.log( (top-base)*(top-base) / base )

    # Remove inf, -inf and set negative value to zero
    # Arrange PMI from large to small. 
    order = np.argsort(PMI)            
    orderedPMI = [PMI[x] for x in order] 
    orderedPMI = pd.DataFrame({'order':order, 'orderedPMI':orderedPMI})
    orderedPMI = orderedPMI[ (orderedPMI.orderedPMI > -np.inf) & (orderedPMI.orderedPMI < np.inf)] # remove inf, -inf, NaN
    orderedPMI = orderedPMI.sort_values(by='orderedPMI',ascending=False)

    orderedPMI[orderedPMI.orderedPMI<=0]=0
    orderedPMI = orderedPMI[orderedPMI.orderedPMI>1]
    pmi_0 = orderedPMI.shape[0] 


    network_words = [voc[x] for x in orderedPMI.order[:nPMI].tolist()] 
    mustInclude = network_words[0:mustInclude_topN_PMI] # words that must be included in the network regardless of criteria set in the following section.
    #mustInclude = list(set(mustInclude + [word]))
    PMI_dic =orderedPMI.orderedPMI[:nPMI].tolist()

    
    ##### Step 3: prepare node information: 
    # Compute PMI with cue word, Cor with cur word, and its own Frequency
    
    # Co-occurence p(w|c)
    cor_dic = [combined_i[x] for x in orderedPMI.order[:nPMI].tolist()]
    cor_dic = list(cor_dic/lexicon_size_year_i)

    # p(c)
    freq_dic = kernel_year_freq[orderedPMI.order[:nPMI].tolist(), year_i-1800]
    freq_dic = freq_dic/lexicon_size_year_i

    PMI_dic = list(PMI_dic)
    cor_dic = list(cor_dic)
    freq_dic = list(freq_dic)

    # set for key words PMI and cor vale
    if word in network_words:
        PMI_dic[network_words.index(word)]=max(PMI_dic)/2
        cor_dic[network_words.index(word)]=max(cor_dic)/2
        freq_dic[network_words.index(word)]=max(freq_dic)/2

    if word not in network_words:
        network_words.append(word)
        PMI_dic.append(max(PMI_dic)/2)
        cor_dic.append(max(cor_dic)/2)
        freq_dic.append(max(freq_dic)/2)

    nodePMI_K = {key:value for key,value in zip(network_words,PMI_dic)}  
    nodeCor_K = {key:value for key,value in zip(network_words,cor_dic)}  
    nodeFreq_K = {key:value for key,value in zip(network_words,freq_dic)}  

    
    ##### Step 4: Compute pair-wise PMI for network words
    
    # Construct co-occurence matrix that contains only network_words
    tempIndex = [voc.index(w) for w in network_words]
    tempData_r = tempData[tempIndex,:][:,tempIndex]
    tempData_r = tempData_r.transpose()+tempData_r

    convert_index = {key:value for key,value in zip(range(0,len(tempIndex)),tempIndex)}  

    # Return a Coordinate (coo) representation of the Compresses-Sparse-Column (csc) matrix.
    coo = tempData_r.tocoo(copy=False)
    coo = scipy.sparse.triu(coo,k=1)

    # Access `row`, `col` and `data` properties of coo matrix.
    coo = pd.DataFrame({'word1': coo.row, 'word2': coo.col, 'co_occur': coo.data}
                     )[['word1', 'word2', 'co_occur']].sort_values(['word1', 'word2']
                     )

    coo = coo[coo.co_occur>=C]
    coo = coo.reset_index(drop=True)

    coo.word1 = [convert_index.get(i) for i in coo.word1]
    coo.word2 = [convert_index.get(i) for i in coo.word2]

    # compute PMI for each pair of context words
    p_w1 = year_i_freq[coo.word1.tolist()]/lexicon_size_year_i
    p_w2 = year_i_freq[coo.word2.tolist()]/lexicon_size_year_i
    p_w1_w2 = np.array(coo.co_occur)/lexicon_size_year_i #(p(a,b))
    tempPMI = np.log(p_w1_w2/(p_w1*p_w2))

    network_df = coo.copy()
    network_df['PMI'] = tempPMI

    network_df=network_df.sort_values('co_occur',ascending=False)
    network_df['word1'] = [voc[i] for i in network_df.word1]
    network_df['word2'] = [voc[i] for i in network_df.word2]

    ##### Step 5: Trim network_df by pairwise co_occurence, PMI, and maximal of nodes
    network_df = network_df[(network_df.co_occur >= C) & (network_df.PMI >= P)]
    network_df =  network_df.reset_index(drop=True)
    network_df['co_occur'] = network_df['co_occur']/lexicon_size_year_i

    # construct a network
    G=nx.Graph() 
    addWeight = []
    for i in range(network_df.shape[0]):
        if weight_by == 'cor':
            addWeight.append((network_df.word1[i],network_df.word2[i],network_df.co_occur[i]))
        if weight_by == 'PMI':
            addWeight.append((network_df.word1[i],network_df.word2[i],network_df.PMI[i]))
    G.add_weighted_edges_from(addWeight)

    # remove nodes if connection (degree of nodes) is less than D
    remove = [node for node, degree in G.degree() if degree < D]
    G.remove_nodes_from(remove)
    remove = [node for node, degree in G.degree() if degree == 0]
    G.remove_nodes_from(remove)

    nodesizePMI = np.array([nodePMI_K.get(x) for x in G.nodes()]); 
    nodesizeCor = np.array([nodeCor_K.get(x) for x in G.nodes()]); 
    nodesizeFreq = np.array([nodeFreq_K.get(x) for x in G.nodes()]); 
    
    clusteringCoefficient = list(nx.clustering(G,weight = 'weight').values())
        
    mustInclude=[x for x in mustInclude if x in G.nodes()]  # some of the mustInclude words have been removed becuase D<5
    print (len(mustInclude))
    
    # set criteria, so that small cc and PMI were ranked, from large to small
    capCriteria =   1 * np.array(clusteringCoefficient).argsort().argsort() + 1* np.array(nodesizePMI).argsort().argsort()# + 1*np.array(nodesizeCor).argsort().argsort()
    capDf = pd.DataFrame({'cap':capCriteria,'words':G.nodes()}).sort_values(by='cap',ascending=False)
    capDf = capDf[capDf.words.isin(mustInclude)==False]
    
    # If number of network nodes is larger than cap, reduce it. 
    if len(G.nodes())>cap:#
        if len(mustInclude)<cap:
            print ('cut')
            cut_it=True
        else:
            cut_it=False
    else:
        cut_it=False

    if cut_it:
        print ('Before cut, number of nodes in this graph: ', len(G.nodes()))
        
        #keep = [y for x,y  in heapq.nlargest(cap, zip(capCriteria,G.nodes()))]
        keep = [y for x,y  in heapq.nlargest((cap-len(mustInclude)), zip(capDf.cap,capDf.words))]
        print ('{} words kept because they are the winner of criteria'.format(len(keep)))
        print ('{} Words forced to include in Network: {}'.format(len(mustInclude),mustInclude))
        
        keep = keep + [word] + mustInclude
        remove = [x for x in G.nodes() if x not in keep]
        G.remove_nodes_from(remove)
        print ('After cut down to cap, number of nodes in this graph: ', len(G.nodes()))
        
        
        remove = [node for node, degree in G.degree() if degree <=1]
        G.remove_nodes_from(remove)
        print ('Is Chernobyl in?','chernobyl' in G.nodes())
        print ('After removing degree less than 1, number of nodes in this graph: ', len(G.nodes()))


    # Specify weights of the edges
    weights = np.array([G[u][v]['weight'] for u,v in G.edges()])
    weights = weights-min(weights)
    weights = weights/10

    # Find modularity/node color
    part = community.best_partition(G)
    mod = community.modularity(part,G)
    nx.set_node_attributes(G,part,'modularity') # add modularity of each nodes to network
    #nx.get_node_attributes(G,'modularity')
    color_values = [part.get(node) for node in G.nodes()]

    # prepare node size
    #if nodesizeoption == 'cor': #!!!!!!! default <<--
    nodesize = np.array([nodeCor_K.get(w) for w in G.node()])
    if word in G.nodes():
        nodesize[list(G.nodes()).index(word)] = nodesize[np.argsort(nodesize)][-2]
    nodesize = (nodesize-min(nodesize))/(max(nodesize)-min(nodesize))

    #nodesize = np.array(np.log(nodesize-0.01 ))
    #nodesize = (nodesize-min(nodesize))/(max(nodesize)-min(nodesize))
    
    # generate node
    nodes_csv =  pd.DataFrame({'id':G.nodes(),'nodeSize': nodesize,'group':color_values})
    nodes_json = nodes_csv.to_json(orient = 'records')
    links_csv = pd.DataFrame({'source':[x[0] for x in G.edges()],'target': [x[1] for x in G.edges()],'weight':weights})
    links_json = links_csv.to_json(orient = 'records')

    json_file = '{"nodes":'+ nodes_json + ',' + '"links":' +links_json + '}'
    if word not in G.nodes():
        print ('Warning: "{}" does not have link to any words in the network because the weight is below threhold. Suggest to lower threshold on PMI or co-occurence'.format(word))

    print ('2') 

    #save json file  
    file = open(dirName + save_path + 'contextualNetwork.json','w') 
    file.write(json_file) 
    file.close()
    




def change_of_contextW(wordT,year_i='1850 2000',k=10,increase=True):
    folderName = wordT[:2]
    year_i = year_i.split()
    year_i = [int(y) for y in year_i]
    year_old = min(year_i)
    year_new = max(year_i)
    
    
    oldW = closest(target_w = wordT,year_i = year_old,k=100)[1][0]
    newW = closest(target_w = wordT,year_i = year_new,k=100)[1][0]
    #print (oldW);print (newW)
    oldIndex = [voc_50k.index(w) for w in oldW]
    newIndex = [voc_50k.index(w) for w in newW]
    index_o = list(set(oldIndex+newIndex))
    
    
    
    fileName = dirName + adjust_path + '/data/50KMatrix_front/'+folderName + '/' + str(voc_50k.index(wordT)) + '.npz'
    data = load_sparse_csr(fileName)
    
    old = data.getrow(year_old - 1700)/sum_year[str(year_old)]
    new = data.getrow(year_new - 1700)/sum_year[str(year_new)]
    #old = data.getrow(year_old - 1700)/kernel_year_freq[voc_50k.index(wordT),old_year-1800]
    #new = data.getrow(year_new - 1700)/kernel_year_freq[voc_50k.index(wordT),new_year-1800]
    
    old = old.toarray()[:,index_o]
    new = new.toarray()[:,index_o]
    
    #dif_add = new - old
    #dif_min = old-new
    #dif = np.concatenate ((dif_add,dif_min))
    
    w,dif = [],[]
    
    for increase in [False,True]:
        if increase:
            dif_i = new-old
        else:
            print ('here')
            dif_i = old-new
        order = numpy.argsort(-dif_i[0])
        index = np.array(index_o)[order]
        dif_i = np.sort(-dif_i)*-1
        w_i  = [voc_50k[i] for i in index[:k]]
        dif_i = dif_i[0,:k]
        w=w+w_i;dif.append(dif_i)
    
    dif[0]=dif[0]*-1   
    dif = np.concatenate(dif)
    print (w);print (dif)
    
    
    
    #plot
    #keys = w; keys=np.array(keys)
    #y = dif
    #x = np.arange(1, len(keys) + 1) 

    #order = np.argsort(y)
    #x,y,keys = x[order], y[order], keys[order]

    #plt.barh(x,y,align="center")
    #plt.yticks(x, keys)
    
    dif=list(dif)
    #w = [i.capitalize() for i in w]
    
    outputD = pd.DataFrame({'name':w, 'value':dif})
    outputD = outputD.sort_values('value',ascending=False)
    outputD.to_csv(dirName + save_path + 'barchart.csv', index=False)
    
    return outputD # increase: what words becomes more associated after xx years



def plot_co_occurence(wordT,contextW,normalize=True):
    file = dirName + adjust_path + '/data/50KMatrix_front/'+ wordT[:2] +'/'+str(voc_50k.index(wordT))+'.npz'
    
    temp = load_sparse_csr(file)
    contextW = contextW.split()
    cor_occur = []
    for w in contextW:
        cor_occur.append(temp.getcol(voc_50k.index(w)))
        
    cor_occur = sparse.hstack(cor_occur).todense()
    cor_occur = cor_occur[(1800-1700):,:]
    if normalize == False:
        cor_occur=cor_occur/sum_year[:,None]
    else:
        cor_occur=cor_occur/kernel_year_freq[voc_50k.index(wordT),][:,None]
    
    cor_occur[cor_occur>1]=0 # remove items that have infinitely large value ################
    outputD = pd.DataFrame(cor_occur)
    outputD.columns= contextW
    insert=list(range(1800,2009))
    outputD.insert(loc=0, column='date', value=insert)
    outputD.to_csv(dirName + save_path + 'co_occurence.csv',index=False)
    
    return outputD   
    #return wordT, contextW, cor_occur

##################################################################################################
# basic frequency and valence
# wrapper_freq, plotV, plotA, plotC 这四个function，输入的参数是wordT，year_s, year_e. 

def wrapper_freq(wordT, year_s = 1800, year_e = 2009):
    if wordT.count('-')==2:
        saveOutPut = plotFreq_middle(wordT)
    
    if wordT.count('-')==1:
        if wordT[0]=='-':
            saveOutPut = plotFreq_end(wordT)
        
        elif wordT[-1]=='-':
            saveOutPut = plotFreq_start(wordT)
    
    if wordT.count('-')==0:
        saveOutPut = plotFreq(wordT)
    saveOutPut.to_csv(dirName + save_path + 'frequency.csv',index = False)
    #return saveOutPut

     
def plotV(wordT, year_s = 1800, year_e = 2009):
    wordT = wordT.split()
    tempIndex = [voc_50k.index(w) for w in wordT]
    temp_data = v[tempIndex]
    print(temp_data.shape)
    outputD = pd.DataFrame(np.transpose(temp_data))
    outputD.columns=wordT
    insert=list(range(year_s,year_e))
    outputD.insert(loc=0, column='date', value=insert)
    outputD.to_csv(dirName + save_path + 'valence.csv',index = False)

def plotA(wordT, year_s = 1800, year_e = 2009):
    wordT = wordT.split()
    tempIndex = [voc_50k.index(w) for w in wordT]
    temp_data = a[tempIndex]
    print(temp_data.shape)
    outputD = pd.DataFrame(np.transpose(temp_data))
    outputD.columns=wordT
    insert=list(range(year_s,year_e))
    outputD.insert(loc=0, column='date', value=insert)
    outputD.to_csv(dirName + save_path + 'arousal.csv',index = False)
        
def plotC(wordT, year_s = 1800, year_e = 2009):
    wordT = wordT.split()
    tempIndex = [voc_50k.index(w) for w in wordT]
    temp_data = c[tempIndex]
    print(temp_data.shape)
    outputD = pd.DataFrame(np.transpose(temp_data))
    outputD.columns=wordT
    insert=list(range(year_s,year_e))
    outputD.insert(loc=0, column='date', value=insert)
    outputD.to_csv(dirName + save_path + 'concreteness.csv',index = False)

def plotFreq(wordT, year_s = 1800, year_e = 2009):
    wordT = wordT.split()
    tempIndex = [voc_50k.index(w) for w in wordT]
    freq  = kernel_year_freq[tempIndex,]/sum_year[None,:]
    outputD = pd.DataFrame(np.transpose(freq))
    outputD.columns=wordT
    insert=list(range(year_s,year_e))
    outputD.insert(loc=0, column='date', value=insert)
    print (outputD)
    return outputD
    #outputD.to_csv(dirName + '/co_occurence.csv',index=False)
        
# end
def plotFreq_end(wordT, year_s = 1800, year_e = 2009):
    label = wordT
    wordT = wordT[1:]
    tempIndex = [voc_50k.index(x) for x in voc_50k if x.endswith(wordT)]
    tempIndex_freq=kernel_year_freq[tempIndex,:]/sum_year[None,:]
    tempIndex_freq = tempIndex_freq.sum(axis=0)
    outputD = pd.DataFrame({ label: tempIndex_freq})
    insert=list(range(year_s,year_e))
    outputD.insert(loc=0, column='date', value=insert)
    return outputD
    
# start
def plotFreq_start(wordT, year_s = 1800, year_e = 2009):
    label = wordT
    wordT = wordT[:-1]
    tempIndex = [voc_50k.index(x) for x in voc_50k if x.startswith(wordT)]
    tempIndex_freq=kernel_year_freq[tempIndex,:]/sum_year[None,:]
    tempIndex_freq = tempIndex_freq.sum(axis=0)
    outputD = pd.DataFrame({ label: tempIndex_freq})
    insert=list(range(year_s,year_e))
    outputD.insert(loc=0, column='date', value=insert)
    return outputD
    
def plotFreq_middle(wordT, year_s = 1800, year_e = 2009):
    label = wordT
    wordT = wordT[1:-1]
    tempIndex = [voc_50k.index(x) for x in voc_50k if wordT in x]
    tempIndex_freq=kernel_year_freq[tempIndex,:]/sum_year[None,:]
    tempIndex_freq = tempIndex_freq.sum(axis=0)
    outputD = pd.DataFrame({ label: tempIndex_freq})
    insert=list(range(year_s,year_e))
    outputD.insert(loc=0, column='date', value=insert)
    return outputD


