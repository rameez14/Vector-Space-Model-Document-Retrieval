import math

class Retrieve:
    # Create new Retrieve object storing index and termWeighting scheme and computes size of documents
    def __init__(self,index, termWeighting):
        self.index = index
        self.termWeighting = termWeighting
        self.docVector = self.calculateDocVector()

    # Method performing retrieval for specified query
    def forQuery(self, query):
        if self.termWeighting == 'binary':
            r = self.binary(query)
        elif self.termWeighting == 'tf':
            r = self.tf(query)
        elif self.termWeighting == 'tfidf':
            r = self.tf_idf(query)
        else:
            r = self.binary(query)
        return r

    #Creates a new dictionary including all documents and their frequencys.
    def calculateDocVector(self):
        idx = self.index
        doc_vectors = dict()
        for a,b in idx.items():
            for v,f in b.items():
                if v in doc_vectors:
                    doc_vectors[v].append([f, len(b)])
                else:
                    doc_vectors[v] = [[f, len(b)]]
        return doc_vectors
    
    
     #Computes binary               
    def binary(self, query):
        idx = self.index
        docs = self.getRelevantDocs(query)
        s = 0
        vQ = 0
        vD = 0
        allranked = []
        allSim = []
        for d in docs:
            for w,v in query.items():
                if w in idx:
                    #vQ += 1
                    doc_values = idx[w]
                    if d in doc_values:
                        s += 1
                        vD += 1
            sim = s / (math.sqrt(vD))
            allSim += [[d, sim]]
            s = 0
            vQ = 0
            vD = 0
        allSim.sort(key = lambda x: x[1], reverse=True)
        for i in range(10):
           allranked  += [allSim[i][0]]
        return allranked


    def tf(self, query):
        idx = self.index
        docs = self.getRelevantDocs(query)
        s = 0
        vD = 0
        allranked = []
        allSim = []
        
        for d in docs:
            for w,v in query.items():
                if w in idx:
                    #Calculating the query can be dropped, but if it had to be calculated,
                    #It whould be calculated like this here:
                    #vQ += v * v
                    doc_values = idx[w]
                    if d in doc_values:
                        s += doc_values[d] * v
                        
            #Calculates document vector
            values = self.docVector[d]
            for i in values:      
                vD += i[0] * i[0]
            sim = s / math.sqrt(vD)
            allSim += [[d, sim]]
            s = 0
            vD = 0
        allSim.sort(key = lambda x: x[1], reverse=True)
        for i in range(10):
           allranked  += [allSim[i][0]]
        return allranked


    def tf_idf (self, query):
        idx = self.index
        docs = self.getRelevantDocs(query)
        totalDocs =len(self.docVector)
        s = 0

        vD = 0
        allranked = []
        allSim = []
        for d in docs:
            for w,v in query.items():
                if w in idx:
                    doc_values = idx[w]
                    #Calculating the query can be dropped, but if it had to be calculated,
                    #It whould be calculated like this here:
                    #vQ += (v * math.log(totalDocs/len(doc_values)))**2 
                    if d in doc_values:
                        s += (doc_values[d] * math.log(totalDocs/len(doc_values))) * (v * math.log(totalDocs/len(doc_values)))

            values = self.docVector[d]
            #Calculates document vector
            for i in values:
                vD += (i[0] * math.log(totalDocs/i[1]))**2
            sim = s / math.sqrt(vD)
            allSim += [[d, sim]]
            s = 0
            vD = 0
        allSim.sort(key = lambda x: x[1], reverse=True)
        for i in range(10):
           allranked  += [allSim[i][0]]
        return allranked



    #Creating the set of relevant docs.
    def getRelevantDocs(self, qry):
        idx = self.index
        #Geting all the document values that have at least one word relevant to our query.
        doc_values = set()
        for a,b in qry.items():
            if a in idx:
                d = idx[a]
                for v,w in d.items():
                    if v in d:
                        doc_values.add(v)
        return doc_values

