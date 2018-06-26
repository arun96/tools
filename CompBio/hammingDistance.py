import sys

def hamming(s1, s2):
	#iterate through the two strings, comparing each index
	l1 = len(s1)
	l2 = len(s2)

	if l1 != l2:
		s = "Must enter two strings of equal length."
		return s

	if l1 == 0:
		return 0
		
	#counter
	c = 0
	#iterate through the two strings, comparing each index
	for i in range(l1):
		if s1[i] != s2[i]:
			c = c + 1
	return c

if __name__ == '__main__':
	print(hamming(str(sys.argv[1]),str(sys.argv[2])))