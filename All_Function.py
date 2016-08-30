import numpy as np
import pprint as pp 
from functools import reduce
from io import StringIO
import re, timeit, collections, random, math

def read_dataset(path):
	print("Reading data...")
	in_path = str(path)+".txt"	
	data_sets= []
	first_reg = re.sub(r'\<(.+?)\>|\!(.*?\n)','',open(in_path,'r').read(),flags=re.DOTALL)
	#print(first_reg)
	sec_reg = re.search(r'(\[(.+?)\])(.+?$)',first_reg,flags=re.DOTALL)
	#print(sec_reg)
	attr_d = filter(None, re.split(r'\s+',sec_reg.group(2),flags=re.DOTALL))
	#print("attr_d: ", attr_d)
	data_sets.append(attr_d)	#first row of data_sets is its attris and decision	
	elm_num = len(attr_d)		#elm_num = 16281
	mess_data = filter(None, re.split(r'\s+',sec_reg.group(3),flags=re.DOTALL))
	set_num = len(mess_data)/elm_num		#set_num = 68
	[data_sets.append(mess_data[i*elm_num:(i+1)*elm_num]) for i in range(0, set_num)]
	"""------------check the seperation of mess data in big txt file---------------
	for j in range(0, set_num+1):
		print(data_sets[j][16280]) 
	"""
	data = np.array(data_sets)	#change to ndarry type
	#print(data)
	return data 

def d_set(ConceptColumn):
	build = timeit.default_timer()
	#print(vectors)
	d_set = []
	d_set_dict = {}		
	uniques =  np.unique(ConceptColumn)
	for key in uniques:
		s_set = np.where(ConceptColumn == key)[0]
		d_set.append(s_set)
		d_set_dict[key] = s_set
	ad_build = timeit.default_timer()
	#print("******Time for a and d*********: ", ad_build-build)
	#print("d_set: ", d_set)
	#print("d_set_dict: ", d_set_dict)		
	return d_set, d_set_dict

def A_set(vectors):	
	#===================Use brute compare to cal A=========================#
	A_build = timeit.default_timer()
	Attrs = vectors.tolist()
	A_set =[]
	#unique_vecs = [vec for vec in set(tuple(x.tolist()) for x in Attrs)]
	vec_set = [list(vec) for vec in set(tuple(x) for x in Attrs)]
	for vec in vec_set:
		#print(vec)
		#A_set.append(np.where(np.prod(vectors==vec,axis=-1))[0])
		A_set.append([pos for pos, y in enumerate(Attrs) if y==vec])
	#print(A_set)
	stop = timeit.default_timer()
	#print("****Time for A_set: ", stop-A_build)
	return A_set

def lower(d_dict, A):	#vectors-contain decisions
	d_star_dict = d_dict
	lower_dict = {}
	lower_set = []
	conflict = False
	BigA = A
	#print("A: ", A)
	start = timeit.default_timer()	
	for key,value in d_star_dict.items():
		lower_pd = []		#lower_set for per d_set
		for sub_A in BigA:
			if set(sub_A) == set(np.intersect1d(sub_A, value)):
				#print("sub_A: ", sub_A)
				#print("value: ", value)
				lower_set.extend(sub_A)
				lower_pd.extend(sub_A)
		lower_dict[key] = lower_pd
	stop = timeit.default_timer() 
	return lower_dict		#conflict, conflictSet 	#lower = diff conflictSet

def upper(d, A):
	d_star_dict = d
	upper_dict = {}
	BigA = A
	for key,value in d_star_dict.items():
		upper_pd = []
		for sub_A in BigA:
			if np.in1d(value,sub_A).any()==True:
				upper_pd.extend(sub_A)
		upper_dict[key] = upper_pd
	return upper_dict
	#print("upper_dict: ", upper_dict)

def col_cutpoints(column,j):		#get all [(a,v)] for lem2, change to new matrix
	#print("Calculating cutpoints...")
	#print("column: ",column)
	column = column.astype(np.float)
	cut_start = timeit.default_timer()
	case_num = len(column)
	#print("case_num: ",case_num)
	sorted_element = []		#all unique elements in the column	
	cp_dict = [] #[[0, '0.8..1.0', array([0, 1, 2])],... format, cutpoints dict, used for lem2
	comp_set = []
	cp_list = []
	sorted_element = sorted(np.unique(column))
	start_point = sorted_element[0]
	end_point = sorted_element[-1]
	#print("sorted_element: ",sorted_element)
	if len(sorted_element) == 1:
		cp_dict.append([j,"%s..%s"%(sorted_element[0],sorted_element[0]),range(0, case_num)])
	else:
		for i in range(0, len(sorted_element)-1):
			mid_point = round((np.float(sorted_element[i])+np.float(sorted_element[i+1]))/2,4)
			cp_list.append(mid_point)	
		for cp in cp_list:
			#print("cp: ", cp)
			pos1 = np.where((column>=start_point)&(column<cp))[0]
			#print("pos1: ", pos1)
			comp_set.append(pos1)
			cp_dict.append([j,"%s..%s"%(start_point,cp),pos1])
			pos2 = np.where((column>cp)&(column<=end_point))[0]
			#print("pos2: ", pos2)
			comp_set.append(pos2)
			cp_dict.append([j,"%s..%s"%(cp, end_point),pos2])
	return cp_dict		#this can be used for lem2	

def col_av(symbolic_col,i):
	#print("In av function...")
	col_av_dict = []		#format is the same with cp_dict
	column = symbolic_col
	sorted_element = sorted(np.unique(column))
	for elem in sorted_element:
		pos = np.where((column==elem))[0]
		col_av_dict.append([i, elem, pos])
	#pp.pprint("av_dict: ")
	#pp.pprint(av_dict)
	return col_av_dict

def lem2(g_dict, av_dict):		#g_dict is either lower_dict or upper_dict
	#print("In lem2 function...")
	goal_dict = g_dict
	total_fat_T = {}
	to_show = {}
	#print("g_dict: ", g_dict)
	for key, value in goal_dict.items():
		real_T = []
		#print("G: ", value)
		G = value
		G_left = G
		while np.size(G)!=0:
			#pp.pprint("Instant G: ")
			#pp.pprint(G)
			T = []
			T_content = []
			T_G = []	#contain intersection of all [(a,v)] and G and their order 
			g_pos_list = []
			for i in range(0, len(av_dict)):
				T_G.append([i,np.intersect1d(av_dict[i][2], G)])	#only av contains the total information 
			#print("all [a,v] intersected with G: ", T_G)
			while T==[] or set(T_content)!=set(np.intersect1d(T_content, value)):
				length_set = np.array([len(T_G[i][1]) for i in range(0, len(T_G))])
				#print("length_set: ", length_set)
				max_item = max(length_set)
				#print("max_item: ", max_item)
				g_pos = np.where((length_set==max_item))[0]		#select the av with the largest intersection
				#print("g_pos: ", g_pos)
				if np.size(g_pos) != 1:
					items_num = np.array([len(av_dict[p][2]) for p in g_pos])
					#print("items_num: ", items_num)
					smallest_set = min(items_num)			#if there's a tie, select the smallest set 
					#print("smallest_set: ", smallest_set)
					g_pos = g_pos[np.where((items_num==smallest_set))]				
					if np.size(g_pos)!=1:
						g_pos = min(g_pos)
					#print("g_pos: ", g_pos)
				g_pos_list.append(g_pos)
				if T_content == []:
					T_content = av_dict[g_pos][2]
				T_content = np.intersect1d(av_dict[g_pos][2], T_content)
				T.append(av_dict[g_pos])	#T contains [attribute_number, attribute interval, case_number]
				G = np.intersect1d(T_content, G)	#new_G
				#print("new_G: ", G)
				T_G = []
				for i in range(0, len(av_dict)):
					T_G.append([i,set(np.intersect1d(av_dict[i][2], G))])
				for g in g_pos_list:
					T_G[g][1] = [] 
			#==========smallest while end====================# 
			if len(T) != 1:
				ts_remain = T
				for p in T:
					ts_sets = [t for t in ts_remain if t!=p]
					ts_sets_av = [ts[2] for ts in ts_sets]
					try:		#if there's two sets and the redundancy one has been deleted
						c_T_content = set(reduce(np.intersect1d, (ts_sets_av)))
						#print("c_T_content, value: ", c_T_content, value)
						if c_T_content <= set(value):
							ts_remain = ts_sets
							T_content = list(c_T_content)
					except:
						#print("Two element in set")
						continue
				real_T.append([[tsm[0:2] for tsm in ts_remain],T_content])
			else:
				#print("Shit: ", T[0:1])
				real_T.append([[T[0][0:2]],T_content])
			
			G_left = np.array(list(set(G_left)-set(T_content)))
			G = G_left
			#==========small for loop end=============#
		if len(real_T) != 1:
			Ts_remain = real_T
			for T in real_T:
				Ts_sets = [rT for rT in Ts_remain if rT != T]
				Ts_sets_keys = [Ts[1] for Ts in Ts_sets]
				try:
					c_fat_T = set(reduce(np.union1d, (Ts_sets_keys)))
					#print('c_fat_T: ', c_fat_T)
					if c_fat_T == set(value):
						#print("What the fuck!", Ts_sets)
						Ts_remain = Ts_sets
				except:
					continue
			#print("Ts_remain: ", Ts_remain)		
		else:
			Ts_remain = real_T
		#print("Ts_remain: ", Ts_remain)
		total_fat_T[key] = Ts_remain   		#with av
		#===========big for loop end================#
	return total_fat_T

def k_parts(cases_num, k):
	f_num = int(math.floor(cases_num/float(k)))
	rs = range(0, cases_num)
	remain = rs
	k_sets = []
	k_count = int(k)
	while k_count!=1:
		select = random.sample(remain, f_num)
		k_sets.append(select)
		remain = set(remain) - set(select)
		k_count = k_count - 1
	k_sets.append(list(remain))
	#print("k_sets: ", k_sets)
	return k_sets
