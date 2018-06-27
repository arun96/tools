import sys

def roman2dec(s):
	l = len(s)
	s_list = list(s)

	translation = {}
	translation['I'] = 1
	translation['V'] = 5
	translation['X'] = 10
	translation['L'] = 50
	translation['C'] = 100
	translation['D'] = 500
	translation['M'] = 1000

	position = {}
	position['I'] = 1
	position['V'] = 2
	position['X'] = 3
	position['L'] = 4
	position['C'] = 5
	position['D'] = 6
	position['M'] = 7

	val = 0

	for pos in range(0,l-1):

		if position[s_list[pos+1]] > position[s_list[pos]]:

			val -= translation[s_list[pos]]

		else:

			val += translation[s_list[pos]]

	val += translation[s_list[l-1]]

	return val


def dec2roman(v):

	number = int(v)

	conversion = [(1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'), (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'), (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')]

	roman = ''

	for (i,r) in conversion:
		(factor, number) = divmod(number, i)
		roman += r * factor

	return roman

	return s
if __name__ == '__main__':

	if str(sys.argv[1]) == 'd2r':
		dec = dec2roman(str(sys.argv[2]))
		print(dec)

	elif str(sys.argv[1]) == 'r2d':
		roman = roman2dec(str(sys.argv[2]))
		print(roman)

	else:
		print("Use r2d to convert from Roman to Decimal, or d2r to convert from Decimal to Roman")
