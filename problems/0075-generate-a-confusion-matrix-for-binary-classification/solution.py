
from collections import Counter

def confusion_matrix(data):
	# Implement the function here
	TP = 0
	FP = 0
	TN = 0
	FN = 0
	for l in data:
		y, yhat = l[0], l[1]
		if y == 1 and  yhat == 1:
			TP += 1
		elif y==0 and yhat == 0:
			TN += 1
		elif y==1 and yhat == 0:
			FN += 1
		else:
			FP += 1
		

	
	return [[TP, FN], [FP, TN]]