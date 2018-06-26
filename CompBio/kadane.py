import sys

def max_subarray(A):
	# initialize as zero
	max_ending_here = max_so_far = A[0]
	start = end = s = 0
	# iterate through the array
	for x in range(1,len(A)):
		if A[x] > max_ending_here + A[x]:
			max_ending_here = A[x]
		else:
			max_ending_here = max_ending_here + A[x]

		if max_so_far > max_ending_here:
			max_so_far = max_so_far

		else:
			max_so_far = max_ending_here
	return max_so_far

def max_subarray_indices(A):
	max_so_far = -float("inf")
	max_ending_here = 0
	start = end = s = 0

	for i in range(0, len(A)):
		max_ending_here += A[i]

		if (max_so_far < max_ending_here):
			max_so_far = max_ending_here
			start = s
			end = i

		if (max_ending_here < 0):
			max_ending_here = 0
			s = i+1

	return max_so_far, start, end

if __name__ == '__main__':
	array1 = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
	array2 = [1,-1, 5, 6]
	array3 = [0, 1, 3, 0, 0, 2, 9, 7, 10]
	array4 = [-3, 2, 1, -4, 5, 2, -1, 3, -1]
	array5 = [0, 1, 2, -2, 3, 2]
	array6 = [-500, 2345, 6980, -1000000, -1, 900000, -1000000, 10000]

	print max_subarray(array2)
	print max_subarray_indices(array2)
 