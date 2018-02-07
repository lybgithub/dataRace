#!/usr/bin/python
#coding:utf-8
import numpy as np
import codecs
import chardet
#import datrie
import sys
import time
#reload(sys)
#sys.setdefaultencoding('utf-8')

class TrieTree(object):
	def __init__(self):
		self.tree = {}
	def add(self,word):
		tree = self.tree
		for char in word:
			if char in tree:
				tree = tree[char]
			else:
				tree[char] = {}
				tree = tree[char]
		tree['exist'] = True
	def search(self,word):
		tree = self.tree
		for char in word:
			if char in tree:
				tree = tree[char]
			else:
				return False
		if 'exist' in tree and tree['exist'] == True:
			return True
		else:
			return False

def add_label(string,dictionary_list,max_length):
	label_all=[]
	for dictionary in dictionary_list:
		right_pointer=len(string)
		flag=[]
		# B:0,E:1,S:2,O:3,I:4
		# 01234，BIESO
		left_pointer=0
		while (True):
			if left_pointer==len(string):
				break
			while (True):
				segmention = string[left_pointer:right_pointer]
				segmention_len = len(segmention)
				if segmention: # if not null
				#	if segmention in dictionary:
					if dictionary.search(segmention):
						if segmention_len==1:
							flag.append(3)           ## single
						if segmention_len==2:
							flag.append(0)           ## begin
							flag.append(2)           ## end
						if segmention_len>2:
							flag.append(0)           ## begin
							for k in range(1,segmention_len-1):
								flag.append(1)       ## inside
							flag.append(2)           ## end
						left_pointer=right_pointer
						break
					else:
						right_pointer-=1
						continue
				else:
					flag.append(4)                 ## outside
					left_pointer=right_pointer+1
					break	
			right_pointer=len(string)
		label_all.append(flag)
	label_all=np.array(label_all)
	label_mat=[]
	for index in range(max_length):
		if index<=len(string)-1:
			label_mat.append(label_all[:,index])
		else:
			label_mat.append(np.array(len(dictionary_list)*[4]))
	label_mat=np.array(label_mat)
	index_for_embedding=[]
	for word in label_mat:
		index=0
		for i in range(len(word)):
			index=index+pow(5,i)*int(word[i])
		index_for_embedding.append(index)
	return np.array(index_for_embedding)

def get_dict(file_list):
	dictionary_list=[]
	# dict_file_list=["test_地区","test_指标","test_主体"]
	dict_file_list = file_list
	for dict_file in dict_file_list:
		print(dict_file)
		tree = TrieTree()
		with codecs.open("dictionary/" + dict_file ,"r", "utf-8") as f:
			for line in f:
				line = line.strip().split('\t')
				area=line[0]
				tree.add(area)
		dictionary_list.append(tree)
	return dictionary_list
'''
def get_dict(file_list):
	dictionary_list=[]
	trie_all=[]
	for dict_file in file_list:
		print(dict_file)
		with codecs.open('dictionary/'+dict_file,'r','utf-8') as f:
			trie = datrie.Trie(ranges=[(u'\u4e00',u'\u9fff')])
			dictionary=[]
			i=1
			for line in f:
				line=line.decode('utf-8')
				line = line.strip().split('\t')
				area = line[0]
				trie[area] = i
				i+=1
		trie_all.append(trie)
	return trie_all
'''

def matrix_index(sentence_matrix,dictionary_list,max_length):
	label_matrix=[]
	for sentence in sentence_matrix:
		#print(sentence)
		index=add_label(sentence,dictionary_list,max_length)
		label_matrix.append(index)
	return np.array(label_matrix)


if __name__=='__main__':
	# file_list=["test_地区","test_指标","test_主体"]
	file_list = ["trainSenDict"]
	dictionary_list=get_dict(file_list)
	print('dictionary all ready')
	string1="中钢期货持仓统计"
	string2="中钢期货持仓统计小喜欢"
	string_mat=np.array([string1,string2])
	
	start = time.clock()
	label_mat=matrix_index(string_mat,dictionary_list,40)
	end = time.clock()
	print(label_mat)
	print(string_mat)






