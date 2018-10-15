import container as contain
import minimizer as mini
from minimizer import tracing
from multiprocessing import Pool
from multiprocessing import cpu_count
import numpy as np
np.seterr(all='raise')

class Controller(object):
	@tracing
	def __init__(self,num_threads,bundle):
		self.bundle = bundle
		self.pool = Pool(num_threads)
		self.runs = [self.pool.apply_async(
									mini.do_instance,
									(self.bundle.n,self.bundle.u,
										self.bundle.Tstep,self.bundle.numT)
										  )
								  for i in range(num_threads)]

	@tracing
	def flip_runs(self,quit_time):
		old_runs = self.runs
		self.runs = []
		for run in old_runs:
			if run.ready():
				print "Run ready"
				try:
					run_result = run.get()
				except Exception as e:
					self.bundle.record_err('u',repr(e))
				else:
					self.bundle.record(run_result)

				if self.bundle.more_runs(quit_time):
					self.runs.append(self.pool.apply_async(
													mini.do_instance,
													(self.bundle.n,self.bundle.u,
													self.bundle.Tstep,self.bundle.numT)
														  )	
									)
			else:
				self.runs.append(run)

	@tracing
	def runs_remaining(self):
		if self.runs:
			return True
		return False

	@tracing
	def finish(self):
		self.pool.close()
		self.pool.join()

	@tracing
	def abandon(self):
		self.pool.terminate()
		self.pool.join()
