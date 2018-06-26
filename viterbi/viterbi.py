import sys

# viterbi implementation
def viterbi(obs, states, start_p, trans_p, emit_p):
    # initialize the dictionary
    V = [{}]

    # iterate through states
    for st in states:
        V[0][st] = {"prob": start_p[st] * emit_p[st][obs[0]], "prev": None}

    # We will run Viterbi when t > 0
    for t in range(1, len(obs)):
        #append a value to the dictionary
        V.append({})

        for st in states:
            # for each state, compute the maximum transition probability
            max_tr_prob = max(V[t-1][prev_st]["prob"]*trans_p[prev_st][st] for prev_st in states)

            # now we check the previous states
            for prev_st in states:
                # check which is the previous state at this point - check probability of previous state * transition probability
                if V[t-1][prev_st]["prob"] * trans_p[prev_st][st] == max_tr_prob:
                    # compute the max prob
                    max_prob = max_tr_prob * emit_p[st][obs[t]]
                    # add to the dictionary
                    V[t][st] = {"prob": max_prob, "prev": prev_st}
                    break

    # initialize opt array
    opt = []

    # Highest probability value
    max_prob = max(value["prob"] for value in V[-1].values())
    previous = None

    # Get most probable state and its backtrack - we will use this to get the sequence
    for st, data in V[-1].items():
        # if this is the max
        if data["prob"] == max_prob:
            # add to the opt
            opt.append(st)
            # set this as the previous
            previous = st
            break

    # Follow the backtrack we obtain till the first observation
    for t in range(len(V) - 2, -1, -1):
        opt.insert(0, V[t + 1][previous]["prev"])
        previous = V[t + 1][previous]["prev"]

    # join the opt array into a string to print
    print(''.join(opt))

    # return the max_prob
    print(max_prob)

# Does final formatting of the data using the arrays made by setup()
# Stores the data as dictionaries
def generateDicts(numStates, numObs, stateNames, obsNames, stateProbs, transitionMatrix, observationMatrix, observations):

    # get number of states
    numStates = int(numStates)

    # get number of observations
    numObs = int(numObs)

    # states is already stateNames
    states = stateNames

    # observations is already observations

    # take stateProbs, and convert into start_probability
    firstState = stateNames[0]
    firstProb = float(stateProbs[0])

    start_probability = {
        firstState: firstProb
    }

    #make the dictionary of states and their probabilities
    for i in range(1,numStates):
        start_probability[stateNames[i]] = float(stateProbs[i])

    # take transitionMatrix, and convert into transition_probability dictionary
    transition_probability = {}

    for i in range(0,numStates):
        transition_probability[stateNames[i]] = {}
        for j in range(0, numStates):
            transition_probability[stateNames[i]][stateNames[j]] = float(transitionMatrix[i][j])

    # take observationMatrix, and convert into emission_probaility dictionary
    emission_probability = {}
    for i in range(0,numStates):
        emission_probability[stateNames[i]] = {}
        for j in range(0, numObs):
            emission_probability[stateNames[i]][obsNames[j]] = float(observationMatrix[i][j])


    return states, observations, start_probability, transition_probability, emission_probability


    
# Extracts data from input file
def setup(filename):
    f = open(filename,"r")
    lines = f.readlines()

    numLines = lines[0]

    numArrays = getNumArray(numLines)

    # states
    numStates = numArrays[0]
    numStates = numStates.replace(",","")

    # observations
    numObs = numArrays[1]

    stateLine = lines[1]

    # state names
    stateNames = getArray(stateLine)

    obsLine = lines[2]

    # observation names
    obsNames = getArray(obsLine)

    probsLine = lines[3]

    # initial probabilities
    stateProbs = getNumArray(probsLine)

    transitionLine = lines[4]

    # transition matrix 
    transitionMatrix = getMatrix(transitionLine)

    observationLine = lines[5]

    # observation matrix
    observationMatrix = getMatrix(observationLine)

    return numStates, numObs, stateNames, obsNames, stateProbs, transitionMatrix, observationMatrix

# converts the input line into an array
def getArray(line):

    # strip the \n character
    line= line.rstrip('\n')

    # split using the commas
    y = line.split(",")

    # remove the spaces in front of each one
    for i in range(1,len(y)):
        length = len(y[i]) -1
        y[i] = y[i][-length:]

    return y

# reads the observations file into an array
def getObservations(filename):
    f = open(filename,"r")
    lines = f.readlines()

    lines = ''.join(lines)

    lines = list(lines)

    return lines

# creates an array of numbers from the input line
def getNumArray(line):

    # split into numbers
    outArray = line.split()

    # remove trailing commas
    for i in range(0, len(outArray)):
        outArray[i] = outArray[i].replace(",","")

    return outArray

# creates a matrix from an input line
def getMatrix(line):

    # strip \n character
    line= line.rstrip('\n')

    # split line based on ; delimiter
    splitline = line.split(";")

    # split based on spacing
    splitline = [x.strip(' ') for x in splitline]

    # split each line into a number array, making the matrix
    for i in range(0, len(splitline)):
        splitline[i] = getNumArray(splitline[i])
    

    # return the matrix
    return splitline


if __name__ == '__main__':
    numStates, numObs, stateNames, obsNames, stateProbs, transitionMatrix, observationMatrix = setup(str(sys.argv[1]))

    observations = getObservations(str(sys.argv[2]))

    states, observations, start_probability, transition_probability, emission_probability = generateDicts(numStates, numObs, stateNames, obsNames, stateProbs, transitionMatrix, observationMatrix, observations)

    viterbi(observations, states, start_probability, transition_probability, emission_probability)