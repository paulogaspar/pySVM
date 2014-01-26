
def linear_kernel(v1, v2):

	s = 0
	for q in xrange(len(v1)):
		s += v1[q] * v2[q]

	return s


def rbf_kernel(v1, v2, sigma):

	s = 0
	for q in xrange(len(v1)):
		s += (v1[q] - v2[q]) * (v1[q] - v2[q])
	return Math.exp(-s/(2.0*sigma*sigma))
