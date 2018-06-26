import sys

gap_penalty = 0
scoreMatrix = None

#Initializes gap_penalty to user defined value, and sets up the score matrix
def setup(sMatrix, gapPen):
    global gap_penalty
    gap_penalty = gapPen
    global scoreMatrix
    scoreMatrix = sMatrix

#reads the input sequence file into an array
def readSequences(filename):
    seqs = [line.rstrip('\n') for line in open(filename)]
    return seqs

#reads scoring matrix into an array
def getMatrix(filename):
    f = open (filename , 'r')
    l = [ map(str,line.split(' ')) for line in f ]

    l[0][4] = 'G'
    for i in range(1,5):
        for j in range(1,5):
            l[i][j] = int(l[i][j])

    return l

#creates a table (of a user chosen size) full of zeros
def zeros(shape):
    retval = []
    for x in range(shape[0]):
        retval.append([])
        for y in range(shape[1]):
            retval[-1].append(0)
    return retval

#gets the match score for a given pair of letters
def match_score(alpha, beta):

    #either is a gap, simply return the gap_penalty
    if alpha == '-' or beta == '-':
        return gap_penalty

    i = 0
    j = 0

    #Find alpha
    if ((alpha == 'C') or (alpha == 'c')):
        i = 1
    elif ((alpha == 'A') or (alpha == 'a')):
        i = 3
    elif ((alpha == 'T') or (alpha == 't')):
        i = 2
    elif ((alpha == 'G') or (alpha == 'g')):
        i = 4

    #Find beta
    if ((beta == 'C') or (beta == 'c')):
        j = 1
    elif ((beta == 'A') or (beta == 'a')):
        j = 3
    elif ((beta == 'T') or (beta == 't')):
        j = 2
    elif ((beta == 'G') or (beta == 'g')):
        j = 4

    #Return corresponding score of alpha and beta
    return scoreMatrix[i][j]

#outputs the alignment and score
def finalize(align1, align2):
    align1 = align1[::-1]    #reverse sequence 1
    align2 = align2[::-1]    #reverse sequence 2
    
    i,j = 0,0
    
    symbol = ''
    score = 0
    for i in range(0,len(align1)):
        # if two letters are the same, then outputthe letter
        if align1[i] == align2[i]:                
            symbol = symbol + align1[i]
            score += match_score(align1[i], align2[i])
    
        # if they are not identical and neither of them is a gap
        elif align1[i] != align2[i] and align1[i] != '-' and align2[i] != '-': 
            score += match_score(align1[i], align2[i])
            symbol += '-'
    
        #if one of them is a gap, output a "-"
        elif align1[i] == '-' or align2[i] == '-':          
            symbol += '-'
            score += gap_penalty
    
    print str.upper(align1)
    print str.upper(symbol)
    #print align2 - used to see sequences when testing my code
    print score

def water(seq1, seq2):
    m, n = len(seq1), len(seq2)  # length of two sequences
    
    # Generate DP table and traceback path pointer matrix
    score = zeros((m+1, n+1))      # the DP table
    pointer = zeros((m+1, n+1))    # matrix to store the traceback path
    
    max_score = 0        # initial maximum score in DP table - will use this as the base case
    # Calculate DP table and mark pointers when needed
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            score_diagonal = score[i-1][j-1] + match_score(seq1[i-1], seq2[j-1])
            score_up = score[i][j-1] + gap_penalty
            score_left = score[i-1][j] + gap_penalty
            score[i][j] = max(0,score_left, score_up, score_diagonal)
            if score[i][j] == 0:
                pointer[i][j] = 0 # 0 means end of the path
            if score[i][j] == score_left:
                pointer[i][j] = 1 # 1 means trace up
            if score[i][j] == score_up:
                pointer[i][j] = 2 # 2 means trace left
            if score[i][j] == score_diagonal:
                pointer[i][j] = 3 # 3 means trace diagonal
            if score[i][j] >= max_score:
                max_i = i
                max_j = j
                max_score = score[i][j];
    
    align1, align2 = '', ''    # initializing sequences
    
    i,j = max_i,max_j    # indices of path starting point - the bottom right corner of the matrix
    
    #traceback, follow pointers
    while pointer[i][j] != 0: #stay within bounds
        if pointer[i][j] == 3: # go diagonal - add in the two letters
            align1 += seq1[i-1]
            align2 += seq2[j-1]
            i -= 1
            j -= 1
        elif pointer[i][j] == 2: # add a "-" to one sequence, and the letter to the other
            align1 += '-'
            align2 += seq2[j-1]
            j -= 1
        elif pointer[i][j] == 1: # add a "-" to one sequence, and the letter to the other
            align1 += seq1[i-1]
            align2 += '-'
            i -= 1

    #output the final alignment and scoring
    finalize(align1, align2)


if __name__ == '__main__':
    #input will be of the form python <filename> <sequences file> <scoring matrix file> <gap penalty)
    matrix = getMatrix(str(sys.argv[2]))
    setup(matrix, int(sys.argv[3]))
    sequences = readSequences(str(sys.argv[1]))

    #can now call the main program - DO NOT PRINT
    water(sequences[0],sequences[1])