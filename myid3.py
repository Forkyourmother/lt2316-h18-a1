import os, sys
import pandas as pd
import numpy as np
import sklearn.metrics
import dill


class DecisionTree:
    def __init__(self, load_from = None):
        if load_from is not None:
            print("Loading from file object.")
            loaded_model = dill.load(load_from)
            self.tree = loaded_model.tree
            self.leaf = loaded_model.leaf
            self.root = loaded_model.root
            self.biasterm = loaded_model.biasterm
        else:
            self.tree = []
            self.leaf = "Nothing"
            self.root = None
            self.biasterm = None
            #print("Initializing classifier.") Since I used actual class instances this print will appear in every part of the recursive function
            #Therefor I commented it out

            

    def train(self, dataframe, class_series, attrs):
        if len(set(class_series)) == 1:
            self.leaf = (class_series.iloc[0])
            return self
        
        if len(attrs) == 0:
            #print(class_series.value_counts().idxmax())
            self.leaf = class_series.value_counts().idxmax()
            
            return self

        self.biasterm = str(class_series.value_counts().idxmax())


        if len(attrs) >= 1:
            column_entropy = {}
            for attribute in attrs:
                entropy = 0     
                for value in list(set(dataframe[attribute])):
                    Index = dataframe.index[dataframe[attribute] == value].tolist() 
                    indexed_class = class_series.loc[Index]
                    class_counts = indexed_class.value_counts()
                    entropy_slice = np.sum(-np.log2(class_counts/len(indexed_class)) * class_counts/len(indexed_class)) * len(dataframe[attribute]) / len(indexed_class)
                    entropy += entropy_slice        
                column_entropy[attribute] = entropy
            next_node = min(column_entropy.items(), key = lambda x: x[1])[0]  

            self.root = next_node

        for value in list(set(dataframe[next_node])):
            Index = dataframe.index[dataframe[next_node] == value].tolist()
            attrs2 = [x for x in attrs if x not in next_node]
            if len(attrs) >= 2:
                dataframe2 = dataframe.drop(next_node, axis = 1).loc[Index]
            else:
                dataframe2 = dataframe.loc[Index]
            class_series2 = class_series.loc[Index]
            instance = DecisionTree().train(dataframe2, class_series2, attrs2)

            self.tree.append((next_node, value, instance))
            # print(next_node, value, instance)
            # # print(self.tree)

        return self

    def predict(self, instance, decisiontree):
        if decisiontree.leaf != "Nothing":  #This is the reason that leaf is set to a string instead of None in the class by default. 
            return decisiontree.leaf             #otherwise we encounter the error "The truth value of a series is ambiguous..." and can't solve this by
        value = instance[decisiontree.root] #remaking the series-value into a string
        for node in decisiontree.tree:
            if str(node[1]) == value:
                newdecisiontree = node[2]
                return self.predict(instance, newdecisiontree)


    def test(self, X, y, display=False):
        if self.root == None:
            print("Class is untrained")
            return
        classified = []
        attributes = X.columns
        for index in range(len(X)):
            row = X.iloc[index]
            prediction = self.predict(row, self)
            classified.append(prediction)
            classified = [prediction if prediction is not None else self.biasterm for prediction in classified]

        precision = sklearn.metrics.precision_score(y,classified, average="micro")
        recall = sklearn.metrics.recall_score(y,classified, average="micro")
        F1 = sklearn.metrics.f1_score(y,classified, average="micro")
        accuracy = sklearn.metrics.accuracy_score(y,classified)
        confusion_matrix = sklearn.metrics.confusion_matrix(y,classified)
        
        result = {'precision':precision,
                  'recall':recall,
                  'accuracy':accuracy,
                  'F1':F1,
                  'confusion-matrix':confusion_matrix}
        if display:
            print (result)
        return result

    def __str__(self):
        if self.root == None:
            return "ID3 untrained"
        else:
            Treedict = self.make_tree_dictionary(self)
            return str(Treedict)

    def make_tree_dictionary(self, trained_class):
        nr_to_string_dict = {1:"one", 2:"two", 3:"three", 4:"four", 5:"five"}
        if trained_class.leaf != "Nothing":
            return trained_class.leaf
        subdict = {}
        if trained_class.root != None:
            treelist = []
            subtree = trained_class.tree
            print(subtree)
            attribute = trained_class.root
            for i in range(len(subtree)):
                treelist.append([nr_to_string_dict[i + 1], self.make_tree_dictionary(subtree[i][2])])
            subdict[attribute] = treelist
        return subdict               #Since I actually used instances of a class for my decision tree the simplest way of getting a readable string out of
                                     #of the tree was using a mis of dictionaries and lists where the attributes are dictionary keys and the attribute 
                                     #values are represented as lists of lists

    def save(self, output):
        dill.dump(self, output)
