import sys

def kmers(fname, k):

	f = open(fname)
	string = f.read()
	f.close()

	kmer = {'  '} # initialize dictionary with dummy element
	
	n = len(string)

	if n == 0:
		return 0

	for i in range(0, n-k+1):
		s = string[i:i+k]
		kmer.update({s}) #add kmer to dictionary

	return len(kmer) -1 #remove initial dummy element

if __name__ == '__main__':
	for i in range(1,11):
		print(kmers(sys.argv[1], i))