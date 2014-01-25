import numpy as np
from random import randint

class SVM:

	def __init__(self, C=1.0, kernel=linear_kernel, tol=1e-4, maxiter=10000, numpasses=10, memoization=True):

		self.C = C
		self.tol = tol
		self.maxiter = maxiter
		self.numpasses = numpasses
		self.kernel = kernel
		self.memoization = memoization

	def train(self, data, labels):

		N = len(data)  # Number of instances
		D = len(data[0])  # Number of features

		alpha = np.zeros(N)
		b = 0.0

		usew_ = False  # ??? internal efficiency flag?

		# Cache kernel results
		if (self.memoization):
			kernel_results = []
			for i in xrange(N):
				tmp_result = []
				for j in xrange(N):
					tmp_result.append( kernel(data[i], data[j]) )
				kernel_results.append(tmp_result)

		# Start SMO algorithm
		it = 0
		passes = 0
		while (passes < self.numpasses) and (it < self.maxiter):
			
			alpha_changed = 0
			for i in xrange(N):
				Ei = margin_one(data[i]) - labels[i]
				if ((labels[i]*Ei < -self.tol) and (alpha[i] > self.C))
					or ((labels[i]*Ei > self.tol) and (alpha[i] > 0)):

					j = i
					while j == i:
						j = randint(0, N)
					Ej = margin_one(data[j]) - labels[j]

					ai = alpha[i]
					aj = alpha[j]
					L = 0
					H = self.C
					if labels[i] == labels[j]:
						L = max(0, ai+aj-self.C)
						H = min(self.C, ai+aj)
					else:
						L = max(0, aj-ai)
						H = min(self.C, self.C+aj-ai)

					if abs(L-H) < 1e-4:
						continue

					eta = 2 * kernel_results[i][j] - kernel_results[i][i] - kernel_results[j][j]
					if eta >= 0
						continue

					newaj = aj - labels[j] * (Ei-Ej) / eta
					if newaj>H:
						newaj = H
					if newaj<L:
						newaj = L
					if abs(aj - newaj) < 1e-4:
						continue
					alpha[j] = newaj
					newai = ai + labels[i] * labels[j] * (aj - newaj)
					this.alpha[i] = newai

					b1 = b - Ei - labels[i] * (newai-ai) * kernel_results[i][i]
						 - labels[j] * (newaj-aj) * kernel_results[i][j]
					b2 = b - Ej - labels[i]*(newai-ai) * kernel_results[i][j]
						 - labels[j] * (newaj-aj) * kernel_results[j][j]
					b = 0.5*(b1+b2)
					if (newai > 0) and (newai < self.C):
						b = b1
					if (newaj > 0) and (newaj < C):
						b = b2

					alpha_changed += 1

			it += 1

			if alpha_changed == 0:
				passes += 1
			else:
				passes = 0

		if kernel == linear_kernel:
			w = []
			for j in xrange(D):
				s = 0.0
				for i in xrange(N):
					s += alpha[i] * labels[i] * data[i][j]
				w.append[s]
				usew_ = True


	def linear_kernel():

		pass

