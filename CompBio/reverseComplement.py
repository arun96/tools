import sys

def reverse(s):

	if len(s) == 0:
		return s

	#reverse the string
	r = s[::-1]

	#iterate through and create the complement
	l = len(r)
	rc = list(r)
	for i in range(l):
		if r[i] == 'A':
			rc[i] = 'T'
		elif r[i] == 'T':
			rc[i] = 'A'
		elif r[i] == 'G':
			rc[i] = 'C'
		elif r[i] == 'C':
			rc[i] = 'G'

	r = "".join(rc)

	return r

if __name__ == '__main__':
	print(reverse(str(sys.argv[1])))