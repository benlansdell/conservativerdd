class A(object):

	def run(self):
		self.method()

	def method(self):
		print("parent method")

class B(A):
	def method(self):
		print("child method")

a = A()
b = B()

b.run()