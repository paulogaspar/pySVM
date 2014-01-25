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
		self.self.alphatol = 1e-7;

	def train(self, data, labels):

		self.data = data
		self.labels = labels

		self.N = len(self.data)  # Number of instances
		self.D = len(self.data[0])  # Number of features

		self.alpha = np.zeros(self.N)
		self.b = 0.0

		self.usew_ = False  # ??? internal efficiency flag?

		# Cache kernel results
		if (self.memoization):
			self.kernel_results = []
			for i in xrange(self.N):
				tmp_result = []
				for j in xrange(self.N):
					tmp_result.append( kernel(self.data[i], self.data[j]) )
				self.kernel_results.append(tmp_result)

		# Start SMO algorithm
		it = 0
		passes = 0
		while (passes < self.numpasses) and (it < self.maxiter):
			
			alpha_changed = 0
			for i in xrange(self.N):
				Ei = margin_one(self.data[i]) - self.labels[i]
				if ((self.labels[i]*Ei < -self.tol) and (self.alpha[i] > self.C))
					or ((self.labels[i]*Ei > self.tol) and (self.alpha[i] > 0)):

					j = i
					while j == i:
						j = randint(0, self.N)
					Ej = margin_one(self.data[j]) - self.labels[j]

					ai = self.alpha[i]
					aj = self.alpha[j]
					L = 0
					H = self.C
					if self.labels[i] == self.labels[j]:
						L = max(0, ai+aj-self.C)
						H = min(self.C, ai+aj)
					else:
						L = max(0, aj-ai)
						H = min(self.C, self.C+aj-ai)

					if abs(L-H) < 1e-4:
						continue

					eta = 2 * kernel_result(i, j) - kernel_result(i, i) - kernel_result(j, j)
					if eta >= 0
						continue

					newaj = aj - self.labels[j] * (Ei-Ej) / eta
					if newaj>H:
						newaj = H
					if newaj<L:
						newaj = L
					if abs(aj - newaj) < 1e-4:
						continue
					self.alpha[j] = newaj
					newai = ai + self.labels[i] * self.labels[j] * (aj - newaj)
					self.alpha[i] = newai

					b1 = self.b - Ei - self.labels[i] * (newai-ai) * kernel_results[i][i]
						 - self.labels[j] * (newaj-aj) * kernel_result(i, j)
					b2 = self.b - Ej - self.labels[i]*(newai-ai) * kernel_results[i][j]
						 - self.labels[j] * (newaj-aj) * kernel_result(j, j)
					self.b = 0.5*(b1+b2)
					if (newai > 0) and (newai < self.C):
						self.b = b1
					if (newaj > 0) and (newaj < C):
						self.b = b2

					alpha_changed += 1

			it += 1

			if alpha_changed == 0:
				passes += 1
			else:
				passes = 0

		if kernel == linear_kernel:
			self.w = []
			for j in xrange(self.D):
				s = 0.0
				for i in xrange(self.N):
					s += self.alpha[i] * self.labels[i] * self.data[i][j]
				self.w.append[s]
				self.usew_ = True
		else:

			newdata = []
			newlabels = []
			newalpha = []
			for i in xrange(self.N):
				if self.alpha[i] > self.alphatol:
					newdata.append(self.data[i])
					newlabels.append(self.labels[i])
					newalpha.append(self.alpha[i])

			self.data = newself.data
			self.labels = newself.labels
			self.alpha = newself.alpha
			self.N = len(self.data)

		trainstats = {iterations: it}

		return trainstats


	# Calculate margin of given instance
	def margin_one(self, arr):
		f = self.b

		if self.usew_:
			for j in xrange(self.D):
				f += arr[j] * self.w[j]
		else:
			for i in xrange(self.N):
				f += self.alpha[i] * self.labels[i] * self.kernel(arr, self.data[i])
		
		return f			


	def predict_one(self, arr):
		return margin_one(arr) > 0 ? 1 : -1


	def margins(self, data):

		N = len(data)
		for i in xrange(N):
			margins.append(margin_one( data[i] ))

		return margins


	def kernel_result(self, i, j):

		if not(self.kernel_results is None):
			return self.kernel_results
		else:
			return self.kernel(self.data[i], self.data[j])


	def predict(self, data):

		margs = margins(data)
		for i in xrange(len(margs)):
			margs[i] = margs[i] > 0 ? 1 : -1

		return margs


	def linear_kernel(v1, v2):

		s = 0
		for q in xrange(len(v1)):
			s += v1[q] * v2[q]

		return s


