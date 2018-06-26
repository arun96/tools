import difflib
import string
import sys
import numpy as np
import re
from itertools import *

# prints 20 '-' in a single line
def printDivider():
    print "--------------------"
	
# get a list containing the reads from the input file
def getReads(filename):
	reads = [line.rstrip('\n') for line in open(filename)]
	return reads

# Get all the kmers of the read
def readKmers(reads, k):
	# dictionary of kmers
	kmers = {}
	# iterate through reads, and get all kmers
	for j in range(0, len(reads)):
		l = len(reads[j])
		i = 0
		limit = l-k
		# while within the read
		while (i <= limit):
			# get the kmer
			window = reads[j][i:i+k]
			# if we've encountered this one before, append the index where it has occurred
			if (kmers.has_key(window)):
				kmers[window].append((j,i))
			# add the kmer and its index to the dictionary
			else:
				kmers[window] = [(j,i)]
			i += 1
	return kmers

# function that finds the overlap between two strings
def findOverlappingRegion(s1, s2):
	s = difflib.SequenceMatcher(None, s1, s2)
	pos_a, pos_b, size = s.find_longest_match(0, len(s1), 0, len(s2)) 
	return s1[pos_a:pos_a+size], size, pos_a, pos_b

# function that finds the superstring
def findSuperstring(s1, s2):
	s = difflib.SequenceMatcher(None, s1, s2)
	pos_a, pos_b, size = s.find_longest_match(0, len(s1), 0, len(s2))
	return s1[0:pos_a] + s2

# function that writes the de Bruijn graph to a dot file
def writeDotFile(filename, edgeList, nodeList, edges, nodes, incomingEdge, outgoingEdge, fromNode, toNode):
	
	file = open(filename, "w")

	s0 = "digraph G {\n"
	file.write(s0)

	# finds each edge, and the incident nodes, and writes it to the file
	for e in edgeList:
		fN = fromNode[e]
		tN = toNode[e]
		s = '\t%s -> %s [label = "%s"];\n' %(fN,tN,e)
		file.write(s)

	s1 = "}"
	file.write(s1)

# function that collapses/simplifies the graph
def collapse(edgeList, nodeList, edges, nodes, incomingEdge, outgoingEdge, fromNode, toNode):

	# we collapse two types of nodes
	# 1. source = no incoming, one outgoing
	# 2. sink = one incoming, no outgoing

	for node in nodeList:

		i = 0
		o = 0

		# count incoming
		if incomingEdge.has_key(node):
			 i = len(incomingEdge[node])

		# count outgoing
		if outgoingEdge.has_key(node):
			o = len(outgoingEdge[node])

		# source
		if i == 0 and o == 1:

			# get the next node
			next = toNode[outgoingEdge[node][0]]

			# check number of incoming and outgoing edges
			incoming = len(incomingEdge[next])

			# can combine if this node has just one incoming

			if incoming == 1:

				# get label of new node
				newNode = incomingEdge[next][0]

				# update the edges
				# edges to update - edge connecting these two must be removed, edges out of next must be updated

				# remove connecting edge
				connectingEdge = outgoingEdge[node][0]
				toNode.pop(connectingEdge, None)
				fromNode.pop(connectingEdge, None)
				edgeList.remove(connectingEdge)

				outgoingEdge[newNode] = [None] * len(outgoingEdge[next])

				# update edges out of next, and the nodes they touch
				for i in range(0, len(outgoingEdge[next])):

					# get the new edge
					newEdge = findSuperstring(incomingEdge[next][0], outgoingEdge[next][i])

					# get the node the edge goes to
					nodeToUpdate = toNode[outgoingEdge[next][i]]

					# print nodeToUpdate

					for a in range(0, len(incomingEdge[nodeToUpdate])):
						if incomingEdge[nodeToUpdate][a] == outgoingEdge[next][i]:
							incomingEdge[nodeToUpdate][a] = newEdge
							#print incomingEdge[nodeToUpdate]

					# no need to touch their outgoing edges in this case

					# remove this edge from the relevant places
					fromNode.pop(outgoingEdge[next][i], None)
					toNode.pop(outgoingEdge[next][i], None)

					# edge is no longer in graph - remove, and add new one
					edgeList.remove(outgoingEdge[next][i])
					edgeList.append(newEdge)

					# update the edge coming out of next
					outgoingEdge[next][i] = newEdge

					# fill in the details of this edge
					fromNode[outgoingEdge[next][i]] = newNode
					toNode[outgoingEdge[next][i]] = nodeToUpdate

					# finally, update a pointer from this node
					outgoingEdge[newNode][i] = outgoingEdge[next][i]

				# remove the pointers to/from these nodes
				outgoingEdge.pop(node, None)
				incomingEdge.pop(next, None)

				# replace the old node with the new node
				nodeList.remove(node)
				nodeList.remove(next)
				nodeList.append(newNode)

		# sink
		elif i == 1 and o == 0:

			# get the previous node
			prev = fromNode[incomingEdge[node][0]]

			# check number of outgoing edges
			outgoing = len(outgoingEdge[prev])

			if outgoing == 1:

				newNode = outgoingEdge[prev][0]

				# update the edges
				# edges to update - edge connecting the two, and edges into prev

				# remove the connecting edge
				connectingEdge = incomingEdge[node][0]
				toNode.pop(connectingEdge, None)
				fromNode.pop(connectingEdge, None)
				edgeList.remove(connectingEdge)

				incomingEdge[newNode] = [None] * len(incomingEdge[prev])

				# update the edges into next, and the nodes they touch
				for j in range(0, len(incomingEdge[prev])):

					# get the new edge
					#newEdge = incomingEdge[prev][j] + outgoingEdge[prev][0][-1:]

					newEdge = findSuperstring(incomingEdge[prev][j], outgoingEdge[prev][0])

					# get the node it comes from
					nodeToUpdate = fromNode[incomingEdge[prev][j]]

					# print nodeToUpdate

					for b in range(0, len(outgoingEdge[nodeToUpdate])):
						if outgoingEdge[nodeToUpdate][b] == incomingEdge[prev][j]:
							outgoingEdge[nodeToUpdate][b] = newEdge

					# no need to touch their incoming edges

					# remove that edge from the relevant places
					fromNode.pop(incomingEdge[prev][j],None)
					toNode.pop(incomingEdge[prev][j],None)

					# edge is no longer in the graph - remove, and add a new one
					edgeList.remove(incomingEdge[prev][j])
					edgeList.append(newEdge)

					# update the edge going into prev
					incomingEdge[prev][j] = newEdge

					# fill in the details of this edge
					fromNode[incomingEdge[prev][j]] = nodeToUpdate
					toNode[incomingEdge[prev][j]] = newNode

					# finally, update a pointer to this node
					incomingEdge[newNode][j] = incomingEdge[prev][j]

				# remove the points to/from these nodes
				incomingEdge.pop(node, None)
				outgoingEdge.pop(prev, None)

				# replace the old node with the new one
				nodeList.remove(node)
				nodeList.remove(prev)
				nodeList.append(newNode)

		# chained
		elif i == 1 and o == 1:

			# get the next node
			next = toNode[outgoingEdge[node][0]]

			# check number of incoming and outgoing edges
			incoming = len(incomingEdge[next])

			# can combine if this node has just one incoming

			if incoming == 1:

				# get label of new node
				newNode = incomingEdge[next][0]

				# update the edges
				# edges to update -  edge connecting these two must be removed, edges out of next must be updated, edge into node

				# remove connecting edge
				connectingEdge = outgoingEdge[node][0]
				toNode.pop(connectingEdge, None)
				fromNode.pop(connectingEdge, None)
				edgeList.remove(connectingEdge)

				# deal with incoming

				incomingEdge[newNode] = [None] * len(incomingEdge[node])

				prevNodeToUpdate = fromNode[incomingEdge[node][0]]

				newIncomingEdge = findSuperstring(incomingEdge[node][0], outgoingEdge[node][0])

				incomingEdge[newNode][0] = newIncomingEdge

				for b in range(0, len(outgoingEdge[prevNodeToUpdate])):
						if outgoingEdge[prevNodeToUpdate][b] == incomingEdge[node][0]:
							outgoingEdge[prevNodeToUpdate][b] = newIncomingEdge

				fromNode[newIncomingEdge] = prevNodeToUpdate
				toNode[newIncomingEdge] = newNode

				fromNode.pop(incomingEdge[node][0],None)
				toNode.pop(incomingEdge[node][0],None)

				edgeList.remove(incomingEdge[node][0])
				edgeList.append(newIncomingEdge)

				outgoingEdge[newNode] = [None] * len(outgoingEdge[next])

				# update edges out of next, and the nodes they touch
				for i in range(0, len(outgoingEdge[next])):

					# get the new edge
					newEdge = findSuperstring(incomingEdge[next][0], outgoingEdge[next][i])

					# get the node the edge goes to
					nodeToUpdate = toNode[outgoingEdge[next][i]]

					# print nodeToUpdate

					for a in range(0, len(incomingEdge[nodeToUpdate])):
						if incomingEdge[nodeToUpdate][a] == outgoingEdge[next][i]:
							incomingEdge[nodeToUpdate][a] = newEdge
							#print incomingEdge[nodeToUpdate]

					# no need to touch their outgoing edges in this case

					# remove this edge from the relevant places
					fromNode.pop(outgoingEdge[next][i], None)
					toNode.pop(outgoingEdge[next][i], None)

					# edge is no longer in graph - remove, and add new one
					edgeList.remove(outgoingEdge[next][i])
					edgeList.append(newEdge)

					# update the edge coming out of next
					outgoingEdge[next][i] = newEdge

					# fill in the details of this edge
					fromNode[outgoingEdge[next][i]] = newNode
					toNode[outgoingEdge[next][i]] = nodeToUpdate

					# finally, update a pointer from this node
					outgoingEdge[newNode][i] = outgoingEdge[next][i]

				# remove the pointers to/from these nodes
				outgoingEdge.pop(node, None)
				incomingEdge.pop(next, None)

				# replace the old node with the new node
				nodeList.remove(node)
				nodeList.remove(next)
				nodeList.append(newNode)

		# not a node than can currently be collapsed
		else:
			# make no changes
			continue

# function that finds source and sink nodes in the graph
def getSourceSink(edgeList, nodeList, edges, nodes, incomingEdge, outgoingEdge, fromNode, toNode):
	sources = set()
	sinks = set()
	for node in nodeList:
		i = 0
		o = 0
		# count incoming
		if incomingEdge.has_key(node):
			i = len(incomingEdge[node])
		# count outgoing
		if outgoingEdge.has_key(node):
			o = len(outgoingEdge[node])
		#print node, i, o

		# no incoming
		if i == 0:
			sources.add(node)

		# no outgoing
		elif o == 0:
			sinks.add(node)

	return sources, sinks

def findAssemblies(edgeList, nodeList, edges, nodes, incomingEdge, outgoingEdge, fromNode, toNode, sources, sinks):

	#findAssemblies(edgeList, nodeList, edges, nodes, incomingEdge, outgoingEdge, fromNode, toNode, sources, sinks)

	assemblies = []

	for source in sources:
		# get the next
		current = source

		next = None
		
		assembly = ""
		while (next not in sinks):

			next = toNode[outgoingEdge[current][0]]

			labelToAppend = outgoingEdge[next][0]

			oldLabel = outgoingEdge[current][0]

			assembly = findSuperstring(findSuperstring(assembly, oldLabel), labelToAppend)

			#print current, next, labelToAppend

			current = next
			next = toNode[outgoingEdge[next][0]]

		assemblies.append(assembly)
		print assembly

# recursively implemented function that finds all paths to a given node
def getPaths(node, incomingEdge, outgoingEdge, fromNode, toNode):

	# if it is not a source node
	if incomingEdge.has_key(node):

		paths = []

		# for each edge
		for edge in incomingEdge[node]:

			fN = fromNode[edge]

			# append edge to previous paths to each node
			for p in getPaths(fN, incomingEdge, outgoingEdge, fromNode, toNode):

				paths.append(findSuperstring(p, edge))

	# base case - it is a source node
	else:

		paths = [""]


	return paths


if __name__ == '__main__':
	
	# get the reads, filename and k
	reads = getReads(str(sys.argv[1]))

	filename = str(sys.argv[2])

	# k = 11 for DNA

	k = len(reads[0])

	# get the edges and nodes
	edgeMers = readKmers(reads, k)
	nodeMers = readKmers(reads, (k-1))

	# get a list of edges and nodes
	edgeList = edgeMers.keys()
	nodeList = nodeMers.keys()

	# store as a set for easy checking
	edges = set(edgeList)
	nodes = set(nodeList)

	# Dictionaries to keep track of things
	incomingEdge = {}
	outgoingEdge = {}
	fromNode = {}
	toNode = {}

	# populate the dictionaries
	for e in edgeList:
		if e[0:k-1] in nodes:
			fromNode[e] = e[0:k-1]
			if outgoingEdge.has_key(e[0:k-1]):
				outgoingEdge[e[0:k-1]].append(e)
			else:
				outgoingEdge[e[0:k-1]] = [e]
		if e[1:k] in nodes:
			toNode[e] = e[1:k]
			if incomingEdge.has_key(e[1:k]):
				incomingEdge[e[1:k]].append(e)
			else:
				incomingEdge[e[1:k]] = [e]

	# preliminary dot file
	writeDotFile(filename, edgeList, nodeList, edges, nodes, incomingEdge, outgoingEdge, fromNode, toNode)

	# collapse the graph
	# continue collapsing until no more changes are made (i.e. it is fully simplified)
	while(1):
		l = len(nodeList)
		collapse(edgeList, nodeList, edges, nodes, incomingEdge, outgoingEdge, fromNode, toNode)
		nl = len(nodeList)
		if l == nl:
			break

	# write the final dot file
	writeDotFile(filename, edgeList, nodeList, edges, nodes, incomingEdge, outgoingEdge, fromNode, toNode)

	# get the sources and sinks
	sources, sinks = getSourceSink(edgeList, nodeList, edges, nodes, incomingEdge, outgoingEdge, fromNode, toNode)

	# get the assemblies by getting all the paths to the sinks
	assemblies = []
	for s in sinks:
		assemblies.append(getPaths(s, incomingEdge, outgoingEdge, fromNode, toNode))

	# finding the longest assembly
	maxLen = 0
	for i in range(0, len(assemblies)):
		for j in range(0, len(assemblies[i])):
			if len(assemblies[i][j]) > maxLen:
				maxLen = len(assemblies[i][j])

	longestAssemblies = []
	for i in range(0, len(assemblies)):
		for j in range(0, len(assemblies[i])):
			if len(assemblies[i][j]) == maxLen:
				longestAssemblies.append(assemblies[i][j])

	# sort alphabetically
	longestAssemblies.sort()

	# print the assemblies
	print "My proposed assemblies:"
	printDivider()
	for i in longestAssemblies:

		print i

