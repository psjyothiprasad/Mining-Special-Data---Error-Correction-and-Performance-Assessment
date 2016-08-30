import numpy as np
import pprint as pp 
from io import StringIO
import re, timeit, collections, random, math, os
from All_Function import d_set,A_set,lower,upper,read_dataset
from All_Function import lem2,col_cutpoints,col_av,k_parts

if __name__ == "__main__":
	"""#------------------for test------------------# 
	file_list = ["common_combined_lers.txt","keller-train-ca.txt",
			"austr.txt","iris-49-aca.txt","test.txt","people.txt"]
	for pf in file_list:
		data = read_dataset(pf)
		print (data)
	"""
#******************Input needed information*******************#
	InFilename = raw_input("What's the name of the input data file? For example: test\n")
	all_files = os.listdir(os.curdir)
	#print("current: ", all_files)
	while not "%s.txt"%InFilename in all_files:
		InFilename = raw_input("The file is not in this directory, please check and input again!\n")
	time1 = timeit.default_timer()	
	data = read_dataset(InFilename)
	time2 = timeit.default_timer()
	#print("**Data read time: ", time2-time1)
	values = data[1:]	#contain decision column
	cases_num = len(values[:,0]) 
	attr_num = len(values[0,0:-1])
	k = raw_input("What's the value of the parameter k?\n")
	while True:
		try:
			if k=='1' or int(k)>=cases_num:
				k=raw_input("Please select k more than 1 and small than dataset cases number: %s\n"%cases_num)
			else:
				break
		except ValueError:
			k=raw_input("Please input a number\n")		
	loru = raw_input("what kind of approximation should be used: lower or upper?\nPlease input lower or upper\n")
	while loru!="lower" and loru!="upper":
		loru = raw_input("Please check whether you input lower or upper correctly\n")
	OutFilename = raw_input("Pleaes give a name of the output data file\n")
	while "%s.txt"%OutFilename in all_files:
		OutFilename = raw_input("File name already exists, please check and input again\n")
#***********************For total dataset**************************#
	t_A_set = A_set(values[:,0:-1])		#t: total
	t_d_set, t_d_dict = d_set(values[:,-1])
	if loru == "lower":
		t_loru_set = lower(t_d_dict,t_A_set) 
	else: 
		t_loru_set = upper(t_d_dict,t_A_set)
	t_total_av = []
	for i in range(0, attr_num):	
		col = values[:,i]
		try:
			float(col[0])	
			cp_dict = col_cutpoints(col,i)
			t_total_av.extend(cp_dict)	

		except ValueError:
			col_av_dict = col_av(col,i)
			t_total_av.extend(col_av_dict)
	t_total_fat_T = lem2(t_loru_set, t_total_av)
	print("Rule sets induced from the whole dataset")
	f= open("%s.txt"%OutFilename, 'a')
	for concept, c_sets in t_total_fat_T.items():
		#print("%s: "%concept)
		#print("c_sets: ", c_sets)
		for s in range(0, len(c_sets)):		#len(c_sets) is the conditions number in rule
			show_string = ""
			for c in range(0, len(c_sets[s][0])-1):
				#print("c",  c)
				show_string = show_string+"(%s, %s) & "%(data[0][c_sets[s][0][c][0]],c_sets[s][0][c][1])
			show_string = show_string+"(%s, %s)"%(data[0][c_sets[s][0][-1][0]],c_sets[s][0][-1][1])+" -> %s"%concept
			specificity = len(c_sets[s][0])		#specificity: total # of conditions in rule
			#print("specificity: ", len(c_sets[s][0]))
			r_cases_num = len(c_sets[s][1])
			#print("Total number of cases matching the rule: ", len(c_sets[s][1]))
			strength = 0
			for j in c_sets[s][1]:
				if values[:,-1][j] != concept:
					continue
				strength += 1
			if strength == 0:
				strength = 1
			show_num = "(%s, %s, %s)"%(specificity, strength, r_cases_num)
			print(show_num)
			print(show_string)			
			f.write(show_num+'\n'+show_string+'\n')
	f.close()
#**************************k folder cross validation parts******************#
	k_sets = k_parts(cases_num, k)		#Second arg is # of k-folders
	for pf in k_sets:
		if k == 1:
			train_mat = values
		else:
			test_mat = [values[pos,:].tolist() for pos in pf]
			test_mat = np.array(test_mat)
			train_mat = np.delete(values, (pf), axis=0)
		part_A_set = A_set(train_mat[:,0:-1])
		part_d_set, part_d_dict = d_set(train_mat[:,-1])
		if loru == "lower":
			loru_set = lower(part_d_dict,part_A_set)  
		else:
			loru_set = upper(part_d_dict,part_A_set)
		#============Seperate symbolic and numeric dataset==========#
		total_av = []
		for i in range(0, attr_num):	
			#col = values[:,i]
			col = train_mat[:,i]
			try:
				float(col[0])	
				cp_dict = col_cutpoints(col,i)
				total_av.extend(cp_dict)	

			except ValueError:
				col_av_dict = col_av(col,i)
				total_av.extend(col_av_dict)			
		total_fat_T = lem2(loru_set, total_av)
		right_set = []
		wrong_set = []
		sum_error_rate = 0
		for decision, situations in total_fat_T.items():
			dr_pos = []
			dw_pos = []
			for s in range(0, len(situations)):		#len(c_sets) is the conditions number in rule
				show_string = ""
				all_pos = []
				for c in range(0, len(situations[s][0])):
					attr_order = situations[s][0][c][0]
					interval = situations[s][0][c][1]
					try: 
						float(test_mat[:,attr_order][0])
						get_int = re.match(r'(.*)(\.\.)(.*)',interval)
						pos1 = np.where((test_mat[:,attr_order].astype(np.float)>=float(get_int.group(1))))[0]
						all_pos.append(pos1)
						pos2 = np.where((test_mat[:,attr_order].astype(np.float)<=float(get_int.group(3))))[0]
						all_pos.append(pos2)
					except ValueError as e:
						pos = np.where((test_mat[:,attr_order]==interval))[0]
						all_pos.append(pos)
				scandi_set = set(reduce(np.intersect1d, (all_pos)))
				sr_set = [m for m in scandi_set if test_mat[:,-1][m]==decision]
				sw_set = [n for n in scandi_set if test_mat[:,-1][n]!=decision]
				dr_pos.extend(sr_set)
				dw_pos.extend(sw_set)
			dr_pos = set(dr_pos)
			dw_pos = set(dw_pos)
			right_set.extend(dr_pos)
			wrong_set.extend(dw_pos)
		right_set = set(right_set)
		wrong_set = set(wrong_set)
		abs_wrong = wrong_set - right_set & wrong_set 
		error_rate = len(abs_wrong)/float(len(test_mat[:,0]))
		sum_error_rate += error_rate
	sum_error_rate = sum_error_rate/float(k)
	print("Error rate: ", sum_error_rate)
	time3 = timeit.default_timer()
	print("**total time: ", time3-time2) 
