import container as contain
import minimizer as mini
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import scipy.stats as scst
import numpy as np
import os
import pickle

path = "/home/dish/data_v2/"
plot_path = "/home/dish/Dropbox/work/research/SYK/paper/plots/"

formatter = ScalarFormatter()
formatter.set_powerlimits((2,2))

cutoff = scst.chi2.ppf(0.99,1)
class stats(object):
	def __init__(self):
		files = os.listdir(path)
		files = filter(lambda x: x[-2:]==".d" if len(x)>=2 else False,files)
		Ns = np.array([],dtype=int)
		us = np.array([])
		print "Analyzing files"
		for file in files:
			try:
				spec = file.split("_")
				N = int(spec[0])
				u = float(spec[1][:-2])
				if not file == "{0}_{1}.d".format(N,u):
					raise Exception()
			except:
				continue
			else:
				f = open(path+"{0}_{1}.d".format(N,u),'r')
				_=f.readline()
				data = pickle.load(f)
				self.numT = data.numT
				self.Tstep = data.Tstep
				f.close()
				if data.num_runs>0:
					Ns = np.append(Ns,N)
					us = np.append(us,u)

		numpoints = Ns.shape[0]
		if numpoints == 0:
			print "No runs completed to analyze"
			return

		self.u = np.sort(np.unique(us))
		self.N = {}
		for u in self.u:
			self.N[u] = np.sort(Ns[us==u])

		self.Ts = self.Tstep*(1+np.arange(self.numT))
		self.Ts.shape = (1,self.numT)


		self.num_runs = {}

		self.EV = {}
		self.EV2 = {}
		self.EV4 = {}

		self.F = {}
		self.F2 = {}
		self.F4 = {}

		self.S = {}
		self.S2 = {}
		self.S4 = {}

		self.FS = {}
		self.FS2 = {}

		self.CV = {}
		self.CV2 = {}

		self.C2 = {}
		self.C4 = {}
		self.C8 = {}

		self.G2 = {}
		self.G4 = {}
		self.G8 = {}

		print "opening files"
		for u in self.u:
			self.num_runs[u] = np.zeros((self.N[u].shape[0]))

			self.EV[u] = np.zeros((self.N[u].shape[0]))
			self.EV2[u] = np.zeros((self.N[u].shape[0]))
			self.EV4[u] = np.zeros((self.N[u].shape[0]))

			self.F[u] = np.zeros((self.N[u].shape[0],self.numT))
			self.F2[u] = np.zeros((self.N[u].shape[0],self.numT))
			self.F4[u] = np.zeros((self.N[u].shape[0],self.numT))

			self.S[u] = np.zeros((self.N[u].shape[0],self.numT))
			self.S2[u] = np.zeros((self.N[u].shape[0],self.numT))
			self.S4[u] = np.zeros((self.N[u].shape[0],self.numT))

			self.FS[u] = np.zeros((self.N[u].shape[0],self.numT))
			self.FS2[u] = np.zeros((self.N[u].shape[0],self.numT))

			self.CV[u] = np.zeros((self.N[u].shape[0],self.numT))
			self.CV2[u] = np.zeros((self.N[u].shape[0],self.numT))

			self.C2[u] = np.zeros((self.N[u].shape[0],self.numT))
			self.C4[u] = np.zeros((self.N[u].shape[0],self.numT))
			self.C8[u] = np.zeros((self.N[u].shape[0],self.numT))

			self.G2[u] = np.zeros((self.N[u].shape[0],self.numT))
			self.G4[u] = np.zeros((self.N[u].shape[0],self.numT))
			self.G8[u] = np.zeros((self.N[u].shape[0],self.numT))

			for nind in range(self.N[u].shape[0]):
				# print "Reading  (N={0},u={1})".format(self.N[u][nind],u)
				f = open(path+"{0}_{1}.d".format(self.N[u][nind],u),'r') 
				_ = f.readline()
				data = pickle.load(f)
				f.close()

				if not data.numT == self.numT:
					raise Exception("Inconsistent number of temperature points")
				if not data.Tstep == self.Tstep:
					raise Exception("Inconsistent temperature point spacing")

				self.num_runs[u][nind] = data.num_runs

				self.EV[u][nind] = data.EV
				self.EV2[u][nind] = data.EV2
				self.EV4[u][nind] = data.EV4

				self.F[u][nind,:] = data.F
				self.F2[u][nind,:] = data.F2
				self.F4[u][nind,:] = data.F4

				self.S[u][nind,:] = data.S
				self.S2[u][nind,:] = data.S2
				self.S4[u][nind,:] = data.S4

				self.FS[u][nind,:] = data.FS
				self.FS2[u][nind,:] = data.FS2

				self.CV[u][nind,:] = data.CV
				self.CV2[u][nind,:] = data.CV2

				self.C2[u][nind,:] = data.C2
				self.C4[u][nind,:] = data.C4
				self.C8[u][nind,:] = data.C8

				self.G2[u][nind,:] = np.sum(data.G2[:,:data.num_runs],axis=1)/data.num_runs
				self.G4[u][nind,:] = np.sum(data.G2[:,:data.num_runs]**2,axis=1)/data.num_runs
				self.G8[u][nind,:] = np.sum(data.G2[:,:data.num_runs]**4,axis=1)/data.num_runs

		print "Computing statistics"
		self.fit()
		f = open(path+'stats','w')
		pickle.dump(self,f)
		f.close()

	def fit(self):
		

		self.des1 = {}
		self.des1e = {}
		self.desN = {}
		self.desN2 = {}
		self.bess = {}

		self.fEV = {}
		self.VEV = {}
		self.fvarEV = {}
		self.VvarEV = {}

		self.fF = {}
		self.VF = {}
		self.fvarF = {}
		self.VvarF = {}

		self.fE = {}
		self.VE = {}
		self.fvarE = {}
		self.VvarE = {}

		self.fS = {}
		self.VS = {}
		self.fvarS = {}
		self.VvarS = {}

		self.fCV = {}
		self.VCV = {}
		self.fvarCV = {}
		self.VvarCV = {}

		self.fC2 = {}
		self.VC2 = {}
		self.fvarC2 = {}
		self.VvarC2 = {}

		self.fG2 = {}
		self.VG2 = {}
		self.fvarG2 = {}
		self.VvarG2 = {}
		

		for u in self.u:
			self.des1[u] = np.zeros((self.N[u].shape[0],3))
			self.des1[u][:,0] = 1.0/self.N[u]**2
			self.des1[u][:,1] = 1.0/self.N[u]
			self.des1[u][:,2] = 1.0

			self.des1e[u] = np.zeros((self.N[u].shape[0],4))
			self.des1e[u][:,0] = 1.0/self.N[u]**3
			self.des1e[u][:,1:] = self.des1[u]

			self.desN[u] = np.diag(self.N[u]).dot(self.des1[u])

			self.desN2[u] = np.zeros((self.N[u].shape[0],5))
			self.desN2[u][:,0] = 1.0/self.N[u]**2
			self.desN2[u][:,1:] = np.diag(self.N[u]**2).dot(self.des1e[u])


			self.bess[u] = np.diag(self.num_runs[u]/(self.num_runs[u]-1))

			XPX1 = np.transpose(self.des1[u]).dot(np.diag(self.num_runs[u])).dot(self.des1[u])
			reg1 = np.linalg.inv(XPX1).dot(np.transpose(self.des1[u])).dot(np.diag(self.num_runs[u]))

			XPX1E = np.transpose(self.des1e[u]).dot(self.des1e[u])
			reg1e = np.linalg.inv(XPX1E).dot(np.transpose(self.des1e[u]))

			XPXN = np.transpose(self.desN[u]).dot(np.diag(self.num_runs[u])).dot(self.desN[u])
			regN = np.linalg.inv(XPXN).dot(np.transpose(self.desN[u])).dot(np.diag(self.num_runs[u]))

			XPXN2 = np.transpose(self.desN2[u]).dot(np.diag(self.num_runs[u])).dot(self.desN2[u])
			regN2 = np.linalg.inv(XPXN2).dot(np.transpose(self.desN2[u])).dot(np.diag(self.num_runs[u]))

			AXPXN2 = np.transpose(self.desN2[u]).dot(self.desN2[u])
			aregN2 = np.linalg.inv(AXPXN2).dot(np.transpose(self.desN2[u]))	

			self.fEV[u] = reg1.dot(self.EV[u])
			varEV = self.bess[u].dot(self.EV2[u] - self.EV[u]**2)
			varEV[varEV<0] = 0
			V = self.num_runs[u]*(varEV 
				+(self.EV[u] - self.des1[u].dot(self.fEV[u]))**2)
			V = np.transpose(self.des1[u]).dot(np.diag(V)).dot(self.des1[u])
			self.VEV[u] = np.linalg.inv(XPX1).dot(V).dot(np.linalg.inv(XPX1))

			self.fvarEV[u] = reg1e.dot(varEV)
			V = (varEV - self.des1e[u].dot(self.fvarEV[u]))**2
			V = np.transpose(self.des1e[u]).dot(np.diag(V)).dot(self.des1e[u])
			self.VvarEV[u] = np.linalg.inv(XPX1E).dot(V).dot(np.linalg.inv(XPX1E))
			 

			self.fF[u] = regN.dot(self.F[u])
			varF = self.bess[u].dot(self.F2[u] - self.F[u]**2)
			varF[varF<0] = 0
			varE = varF
			V = np.diag(self.num_runs[u]).dot(
				(varF + (self.F[u] - self.desN[u].dot(self.fF[u]))**2))
			V = np.reshape(np.transpose(V),(self.numT,self.N[u].shape[0],1))
			V = V*np.reshape(self.desN[u],(1,self.N[u].shape[0],self.desN[u].shape[1]))
			V = np.transpose(self.desN[u]).dot(V)
			V = np.transpose(V,(1,0,2))
			self.VF[u] = np.transpose(np.linalg.inv(XPXN).dot(V).dot(np.linalg.inv(XPXN)),(1,0,2))
			self.fvarF[u] = aregN2.dot(varF)
			V = (varF - self.desN2[u].dot(self.fvarF[u]))**2
			V = np.reshape(np.transpose(V),(self.numT,self.N[u].shape[0],1))
			V = V*np.reshape(self.desN2[u],(1,self.N[u].shape[0],self.desN2[u].shape[1]))
			V = np.transpose(self.desN2[u]).dot(V)
			V = np.transpose(V,(1,0,2))
			self.VvarF[u] = np.transpose(np.linalg.inv(AXPXN2).dot(V).dot(np.linalg.inv(AXPXN2)),(1,0,2))

			self.fS[u] = regN.dot(self.S[u])
			varS = self.bess[u].dot(self.S2[u] - self.S[u]**2)
			varS[varS<0] = 0
			varE = varE+(self.Ts**2)*varS
			V = np.diag(self.num_runs[u]).dot(
				(varS + (self.S[u] - self.desN[u].dot(self.fS[u]))**2))
			V = np.reshape(np.transpose(V),(self.numT,self.N[u].shape[0],1))
			V = V*np.reshape(self.desN[u],(1,self.N[u].shape[0],self.desN[u].shape[1]))
			V = np.transpose(self.desN[u]).dot(V)
			V = np.transpose(V,(1,0,2))
			self.VS[u] = np.transpose(np.linalg.inv(XPXN).dot(V).dot(np.linalg.inv(XPXN)),(1,0,2))
			self.fvarS[u] = aregN2.dot(varS)
			V = (varS - self.desN2[u].dot(self.fvarS[u]))**2
			V = np.reshape(np.transpose(V),(self.numT,self.N[u].shape[0],1))
			V = V*np.reshape(self.desN2[u],(1,self.N[u].shape[0],self.desN2[u].shape[1]))
			V = np.transpose(self.desN2[u]).dot(V)
			V = np.transpose(V,(1,0,2))
			self.VvarS[u] = np.transpose(np.linalg.inv(AXPXN2).dot(V).dot(np.linalg.inv(AXPXN2)),(1,0,2))

			self.fE[u] = regN.dot(self.F[u]+self.Ts*self.S[u])
			varE = varE + self.bess[u].dot(2*self.Ts*(self.FS[u] - self.F[u]*self.S[u]))
			varE[varE<0] = 0
			V = np.diag(self.num_runs[u]).dot(
				(varE + (self.F[u]+self.Ts*self.S[u] - self.desN[u].dot(self.fE[u]))**2))
			V = np.reshape(np.transpose(V),(self.numT,self.N[u].shape[0],1))
			V = V*np.reshape(self.desN[u],(1,self.N[u].shape[0],self.desN[u].shape[1]))
			V = np.transpose(self.desN[u]).dot(V)
			V = np.transpose(V,(1,0,2))
			self.VE[u] = np.transpose(np.linalg.inv(XPXN).dot(V).dot(np.linalg.inv(XPXN)),(1,0,2))
			self.fvarE[u] = aregN2.dot(varE)
			V = (varE - self.desN2[u].dot(self.fvarE[u]))**2
			V = np.reshape(np.transpose(V),(self.numT,self.N[u].shape[0],1))
			V = V*np.reshape(self.desN2[u],(1,self.N[u].shape[0],self.desN2[u].shape[1]))
			V = np.transpose(self.desN2[u]).dot(V)
			V = np.transpose(V,(1,0,2))
			self.VvarE[u] = np.transpose(np.linalg.inv(AXPXN2).dot(V).dot(np.linalg.inv(AXPXN2)),(1,0,2))

			self.fCV[u] = regN.dot(self.CV[u])
			varCV = self.bess[u].dot(self.CV2[u] - self.CV[u]**2)
			varCV[varCV<0] = 0
			V = np.diag(self.num_runs[u]).dot(
				(varCV + (self.CV[u] - self.desN[u].dot(self.fCV[u]))**2))
			V = np.reshape(np.transpose(V),(self.numT,self.N[u].shape[0],1))
			V = V*np.reshape(self.desN[u],(1,self.N[u].shape[0],self.desN[u].shape[1]))
			V = np.transpose(self.desN[u]).dot(V)
			V = np.transpose(V,(1,0,2))
			self.VCV[u] = np.transpose(np.linalg.inv(XPXN).dot(V).dot(np.linalg.inv(XPXN)),(1,0,2))
			self.fvarCV[u] = aregN2.dot(varCV)
			V = (varCV - self.desN2[u].dot(self.fvarCV[u]))**2
			V = np.reshape(np.transpose(V),(self.numT,self.N[u].shape[0],1))
			V = V*np.reshape(self.desN2[u],(1,self.N[u].shape[0],self.desN2[u].shape[1]))
			V = np.transpose(self.desN2[u]).dot(V)
			V = np.transpose(V,(1,0,2))
			self.VvarCV[u] = np.transpose(np.linalg.inv(AXPXN2).dot(V).dot(np.linalg.inv(AXPXN2)),(1,0,2))

			self.fC2[u] = regN.dot(self.C2[u])
			varC2 = self.bess[u].dot(self.C4[u] - self.C2[u]**2)
			varC2[varC2<0] = 0
			V = np.diag(self.num_runs[u]).dot(
				(varC2 + (self.C2[u] - self.desN[u].dot(self.fC2[u]))**2))
			V = np.reshape(np.transpose(V),(self.numT,self.N[u].shape[0],1))
			V = V*np.reshape(self.desN[u],(1,self.N[u].shape[0],self.desN[u].shape[1]))
			V = np.transpose(self.desN[u]).dot(V)
			V = np.transpose(V,(1,0,2))
			self.VC2[u] = np.transpose(np.linalg.inv(XPXN).dot(V).dot(np.linalg.inv(XPXN)),(1,0,2))
			self.fvarC2[u] = aregN2.dot(varC2)
			V = (varC2 - self.desN2[u].dot(self.fvarC2[u]))**2
			V = np.reshape(np.transpose(V),(self.numT,self.N[u].shape[0],1))
			V = V*np.reshape(self.desN2[u],(1,self.N[u].shape[0],self.desN2[u].shape[1]))
			V = np.transpose(self.desN2[u]).dot(V)
			V = np.transpose(V,(1,0,2))
			self.VvarC2[u] = np.transpose(np.linalg.inv(AXPXN2).dot(V).dot(np.linalg.inv(AXPXN2)),(1,0,2))

			self.fG2[u] = regN.dot(self.G2[u])
			varG2 = self.bess[u].dot(self.G4[u] - self.G2[u]**2)
			varG2[varG2<0] = 0
			V = np.diag(self.num_runs[u]).dot(
				(varG2 + (self.G2[u] - self.desN[u].dot(self.fG2[u]))**2))
			V = np.reshape(np.transpose(V),(self.numT,self.N[u].shape[0],1))
			V = V*np.reshape(self.desN[u],(1,self.N[u].shape[0],self.desN[u].shape[1]))
			V = np.transpose(self.desN[u]).dot(V)
			V = np.transpose(V,(1,0,2))
			self.VG2[u] = np.transpose(np.linalg.inv(XPXN).dot(V).dot(np.linalg.inv(XPXN)),(1,0,2))
			self.fvarG2[u] = aregN2.dot(varG2)
			V = (varG2 - self.desN2[u].dot(self.fvarG2[u]))**2
			V = np.reshape(np.transpose(V),(self.numT,self.N[u].shape[0],1))
			V = V*np.reshape(self.desN2[u],(1,self.N[u].shape[0],self.desN2[u].shape[1]))
			V = np.transpose(self.desN2[u]).dot(V)
			V = np.transpose(V,(1,0,2))
			self.VvarG2[u] = np.transpose(np.linalg.inv(AXPXN2).dot(V).dot(np.linalg.inv(AXPXN2)),(1,0,2))

		self.Ts = np.squeeze(self.Ts)

	def plotNR(self):
		plt.figure()
		markers= {-1:'*',0:'o',1:'^'}
		for u in [-1,0,1]:
			plt.plot(self.N[u],np.log(self.num_runs[u])/np.log(10),'.',label='u='+str(u),marker = markers[u])

		plt.legend()
		ylabels = np.append(np.round(10*(1.3**np.arange(0,np.ceil(np.log(50)/np.log(1.3)),1))),500).astype('int')
		plt.yticks(np.log(ylabels)/np.log(10),ylabels)
		plt.gca().yaxis.grid(True)
		plt.title('Number of Samples')
		plt.ylabel('Number of Samples (log scale)')
		plt.xlabel('N')
		plt.savefig(plot_path+'NR.pdf')
		plt.show()


	def EVplot(self):
		for u in self.u:
			plt.figure()
			plt.errorbar(self.N[u],self.EV[u],yerr=np.sqrt(self.bess[u]).dot(np.sqrt(self.EV2[u]-self.EV[u]**2)),fmt='.',label='mean,stdev')
			plt.plot(self.N[u],self.des1[u].dot(self.fEV[u]),label='fit')
			plt.axhline(y=self.fEV[u][2],label='limit',color='m')
			plt.legend()
			plt.xlabel('N')
			plt.ylabel('$\\lambda_m$/J')
			plt.title('$\\lambda_m$, u='+str(u))
			plt.gca().yaxis.set_major_formatter(formatter)
			plt.savefig(plot_path+'EV'+str(int(u))+'.pdf')

		plt.show()

	def av_fit_plots(self):
		plt.figure()
		plt.errorbar(self.Ts[:50],self.fC2[1][-1,:50],yerr=np.sqrt(self.VC2[1][:50,-1,-1]*cutoff),fmt='.',marker='^',label='u=1')
		plt.errorbar(self.Ts[:50],self.fC2[0][-1,:50],yerr=np.sqrt(self.VC2[0][:50,-1,-1]*cutoff),fmt='.',marker='o',label='u=0')
		plt.errorbar(self.Ts[:50],self.fC2[-1][-1,:50],yerr=np.sqrt(self.VC2[-1][:50,-1,-1]*cutoff),fmt='.',marker='*',label='u=-1')
		plt.legend()
		plt.xlabel("$T/J$",fontsize='larger')
		plt.ylabel('$2||C||^2/N$ ',fontsize='large')
		plt.title("Leading order Behavior of $\\overline{||C_*||^2}$, low temperature")
		plt.gca().yaxis.set_major_formatter(formatter)
		plt.savefig(plot_path+'inset.pdf')

		plt.figure()
		plt.errorbar(self.Ts[:50],self.fG2[1][-1,:50],yerr=np.sqrt(self.VG2[1][:50,-1,-1]*cutoff),fmt='.',marker='^',label='u=1')
		plt.errorbar(self.Ts[:50],self.fG2[0][-1,:50],yerr=np.sqrt(self.VG2[0][:50,-1,-1]*cutoff),fmt='.',marker='o',label='u=0')
		plt.errorbar(self.Ts[:50],self.fG2[-1][-1,:50],yerr=np.sqrt(self.VG2[-1][:50,-1,-1]*cutoff),fmt='.',marker='*',label='u=-1')
		plt.legend()
		plt.xlabel("$T/J$",fontsize='larger')
		plt.ylabel('$2||G||^2/N$ ',fontsize='large')
		plt.title("Leading order Behavior of $\\overline{||G_*||^2}$, low temperature")
		plt.gca().yaxis.set_major_formatter(formatter)
		plt.savefig(plot_path+'insetg.pdf')
		for u in self.u:
			plt.figure()
			plt.errorbar(self.Ts,self.fC2[u][-1,:],yerr=np.sqrt(self.VC2[u][:,-1,-1]*cutoff),fmt='.',label='fit')
			plt.axvline(x=-self.fEV[u][-1],label="limit of $|\\overline{\\lambda_m}|$",color='m')
			plt.axvline(x=-self.EV[u][0],label="$|\\overline{\\lambda_m}|$ at $N=$"+str(self.N[u][0]),ls=':',color='m')
			plt.axvline(x=-self.EV[u][-1],label="$|\\overline{\\lambda_m}|$ at $N=$"+str(self.N[u][-1]),ls='-.',color='m')
			plt.legend()
			plt.xlabel("$T/J$",fontsize='larger')
			plt.ylabel('$2||C||^2/N$ ',fontsize='large')
			plt.title("Leading order Behavior of $\\overline{||C_*||^2}$, u="+str(u))
			plt.gca().yaxis.set_major_formatter(formatter)
			plt.savefig(plot_path+'C2'+str(int(u))+'.pdf')

			plt.figure()
			plt.errorbar(self.Ts,self.fG2[u][-1,:],yerr=np.sqrt(self.VG2[u][:,-1,-1]*cutoff),fmt='.',label='fit')
			plt.axvline(x=-self.fEV[u][-1],label="limit of $|\\overline{\\lambda_m}|$",color='m')
			plt.axvline(x=-self.EV[u][0],label="$|\\overline{\\lambda_m}|$ at $N=$"+str(self.N[u][0]),ls=':',color='m')
			plt.axvline(x=-self.EV[u][-1],label="$|\\overline{\\lambda_m}|$ at $N=$"+str(self.N[u][-1]),ls='-.',color='m')
			plt.legend()
			plt.xlabel("$T/J$",fontsize='larger')
			plt.ylabel('$2||G||^2/N$',fontsize='large')
			plt.title("Leading order Behavior of $\\overline{||G_*||^2}$, u="+str(u))
			plt.gca().yaxis.set_major_formatter(formatter)
			plt.savefig(plot_path+'G2'+str(int(u))+'.pdf')

			plt.figure()
			plt.errorbar(self.Ts,self.fF[u][-1,:],yerr=np.sqrt(cutoff*self.VF[u][:,-1,-1]),fmt='.',label='fit')
			plt.axvline(x=-self.fEV[u][-1],label="limit of $|\\overline{\\lambda_m}|$",color='m')
			plt.axvline(x=-self.EV[u][0],label="$|\\overline{\\lambda_m}|$ at $N=$"+str(self.N[u][0]),ls=':',color='m')
			plt.axvline(x=-self.EV[u][-1],label="$|\\overline{\\lambda_m}|$ at $N=$"+str(self.N[u][-1]),ls='-.',color='m')
			plt.legend()
			plt.xlabel("$T/J$",fontsize='larger')
			plt.ylabel('$\\mathscr{F}_t/NJ$',fontsize='large')
			plt.title("Leading order Behavior of $\\overline{\mathscr{F}}_{t*}$, u="+str(u))
			plt.gca().yaxis.set_major_formatter(formatter)
			plt.savefig(plot_path+'F'+str(int(u))+'.pdf')

			plt.figure()
			plt.errorbar(self.Ts,self.fCV[u][-1,:],yerr=np.sqrt(cutoff*self.VCV[u][:,-1,-1]),fmt='.',label='fit')
			plt.axvline(x=-self.fEV[u][-1],label="limit of $|\\overline{\\lambda_m}|$",color='m')
			plt.axvline(x=-self.EV[u][0],label="$|\\overline{\\lambda_m}|$ at $N=$"+str(self.N[u][0]),ls=':',color='m')
			plt.axvline(x=-self.EV[u][-1],label="$|\\overline{\\lambda_m}|$ at $N=$"+str(self.N[u][-1]),ls='-.',color='m')
			plt.legend()
			plt.xlabel("$T/J$",fontsize='larger')
			plt.ylabel('$C_V/N$',fontsize='large')
			plt.title("Leading order Behavior of $\\overline{C}_V$, u="+str(u))
			plt.gca().yaxis.set_major_formatter(formatter)
			plt.savefig(plot_path+'CV'+str(int(u))+'.pdf')

			plt.figure()
			plt.errorbar(self.Ts,self.fE[u][-1,:],yerr=np.sqrt(self.VE[u][:,-1,-1]*cutoff),fmt='.',label='fit')
			plt.axvline(x=-self.fEV[u][-1],label="limit of $|\\overline{\\lambda_m}|$",color='m')
			plt.axvline(x=-self.EV[u][0],label="$|\\overline{\\lambda_m}|$ at $N=$"+str(self.N[u][0]),ls=':',color='m')
			plt.axvline(x=-self.EV[u][-1],label="$|\\overline{\\lambda_m}|$ at $N=$"+str(self.N[u][-1]),ls='-.',color='m')
			plt.legend()
			plt.xlabel("$T/J$",fontsize='larger')
			plt.ylabel('$\\overline{E}_*/NJ$',fontsize='large')
			plt.title("Leading order Behavior of $\\overline{E}_*$, u="+str(u))
			plt.gca().yaxis.set_major_formatter(formatter)
			plt.savefig(plot_path+'E'+str(int(u))+'.pdf')

			plt.figure()
			plt.errorbar(self.Ts,self.fS[u][-1,:],yerr=np.sqrt(self.VS[u][:,-1,-1]*cutoff),fmt='.',label='fit')
			plt.axvline(x=-self.fEV[u][-1],label="limit of $|\\overline{\\lambda_m}|$",color='m')
			plt.axvline(x=-self.EV[u][0],label="$|\\overline{\\lambda_m}|$ at $N=$"+str(self.N[u][0]),ls=':',color='m')
			plt.axvline(x=-self.EV[u][-1],label="$|\\overline{\\lambda_m}|$ at $N=$"+str(self.N[u][-1]),ls='-.',color='m')
			plt.legend()
			plt.xlabel("$T/J$",fontsize='larger')
			plt.ylabel('$S/N$',fontsize='large')
			plt.title("Leading order Behavior of $\\overline{S}_*$, u="+str(u))
			plt.gca().yaxis.set_major_formatter(formatter)
			plt.savefig(plot_path+'S'+str(int(u))+'.pdf')
		
		plt.show()
	
	def SA_plots(self):
		for u in self.u:
			plt.figure()
			plt.plot(self.Ts,self.fvarC2[u][-1,:],'.',label='fit')
			plt.plot(self.Ts,self.fvarC2[u][-1,:]+np.sqrt(self.VvarC2[u][:,-1,-1]*cutoff),'r',ls='--',color='c',label="Confidence window")
			plt.plot(self.Ts,self.fvarC2[u][-1,:]-np.sqrt(self.VvarC2[u][:,-1,-1]*cutoff),'r',ls='--',color='c')
			plt.axvline(x=-self.fEV[u][-1],label="limit of $|\\overline{\\lambda_m}|$",color='m')
			plt.axvline(x=-self.EV[u][0],label="$|\\overline{\\lambda_m}|$ at $N=$"+str(self.N[u][0]),ls=':',color='m')
			plt.axvline(x=-self.EV[u][-1],label="$|\\overline{\\lambda_m}|$ at $N=$"+str(self.N[u][-1]),ls='-.',color='m')
			plt.axhline(y=0)
			plt.xlabel("$T/J$",fontsize='larger')
			plt.ylabel('$4$ var$(||C||^2)/N^2$',fontsize='large')
			plt.legend()
			plt.title("Leading Order Behavior of var$(||C_*||^2)$, $u = $"+str(u))
			plt.gca().yaxis.set_major_formatter(formatter)
			plt.savefig(plot_path+'VC2'+str(int(u))+'.pdf')

			plt.figure()
			plt.plot(self.Ts,self.fvarG2[u][-1,:],'.',label='fit')
			plt.plot(self.Ts,self.fvarG2[u][-1,:]+np.sqrt(self.VvarG2[u][:,-1,-1]*cutoff),'r',ls='--',color='c',label="Confidence window")
			plt.plot(self.Ts,self.fvarG2[u][-1,:]-np.sqrt(self.VvarG2[u][:,-1,-1]*cutoff),'r',ls='--',color='c')
			plt.axvline(x=-self.fEV[u][-1],label="limit of $|\\overline{\\lambda_m}|$",color='m')
			plt.axvline(x=-self.EV[u][0],label="$|\\overline{\\lambda_m}|$ at $N=$"+str(self.N[u][0]),ls=':',color='m')
			plt.axvline(x=-self.EV[u][-1],label="$|\\overline{\\lambda_m}|$ at $N=$"+str(self.N[u][-1]),ls='-.',color='m')
			plt.axhline(y=0)
			plt.xlabel("$T/J$",fontsize='larger')
			plt.ylabel('$4$ var$(||G||^2)/N^2$',fontsize='large')
			plt.title("Leading Order Behavior of var$(||G_*||^2)$, $u = $"+str(u))
			plt.legend()
			plt.gca().yaxis.set_major_formatter(formatter)
			plt.savefig(plot_path+'VG2'+str(int(u))+'.pdf')

			plt.figure()
			plt.plot(self.Ts,self.fvarF[u][-1,:],'.',label='fit')
			plt.plot(self.Ts,self.fvarF[u][-1,:]+np.sqrt(self.VvarF[u][:,-1,-1]*cutoff),'r',ls='--',color='c',label="Confidence window")
			plt.plot(self.Ts,self.fvarF[u][-1,:]-np.sqrt(self.VvarF[u][:,-1,-1]*cutoff),'r',ls='--',color='c')
			plt.axvline(x=-self.fEV[u][-1],label="limit of $|\\overline{\\lambda_m}|$",color='m')
			plt.axvline(x=-self.EV[u][0],label="$|\\overline{\\lambda_m}|$ at $N=$"+str(self.N[u][0]),ls=':',color='m')
			plt.axvline(x=-self.EV[u][-1],label="$|\\overline{\\lambda_m}|$ at $N=$"+str(self.N[u][-1]),ls='-.',color='m')
			plt.axhline(y=0)
			plt.xlabel("$T/J$",fontsize='larger')
			plt.ylabel('var$(\\mathscr{F}_{t})/N^2$',fontsize='large')
			plt.legend()
			plt.title("Leading Order Behavior of var$(\\mathscr{F}_t)$, $u=$"+str(u))
			plt.gca().yaxis.set_major_formatter(formatter)
			plt.savefig(plot_path+'VF'+str(int(u))+'.pdf')

			plt.figure()
			plt.plot(self.Ts,self.fvarCV[u][-1,:],'.',label='fit')
			plt.plot(self.Ts,self.fvarCV[u][-1,:]+np.sqrt(self.VvarCV[u][:,-1,-1]*cutoff),'r',ls='--',color='c',label="Confidence window")
			plt.plot(self.Ts,self.fvarCV[u][-1,:]-np.sqrt(self.VvarCV[u][:,-1,-1]*cutoff),'r',ls='--',color='c')
			plt.axvline(x=-self.fEV[u][-1],label="limit of $|\\overline{\\lambda_m}|$",color='m')
			plt.axvline(x=-self.EV[u][0],label="$|\\overline{\\lambda_m}|$ at $N=$"+str(self.N[u][0]),ls=':',color='m')
			plt.axvline(x=-self.EV[u][-1],label="$|\\overline{\\lambda_m}|$ at $N=$"+str(self.N[u][-1]),ls='-.',color='m')
			plt.axhline(y=0)
			plt.xlabel("$T/J$",fontsize='larger')
			plt.ylabel('var$(C_V)/N^2$',fontsize='large')
			plt.legend()
			plt.title("Leading Order Behavior of var$(C_V)$, $u=$"+str(u))
			plt.gca().yaxis.set_major_formatter(formatter)
			plt.savefig(plot_path+'VCV'+str(int(u))+'.pdf')

			plt.figure()
			plt.plot(self.Ts,self.fvarE[u][-1,:],'.',label='fit')
			plt.plot(self.Ts,self.fvarE[u][-1,:]+np.sqrt(self.VvarE[u][:,-1,-1]*cutoff),'r',ls='--',color='c',label="Confidence window")
			plt.plot(self.Ts,self.fvarE[u][-1,:]-np.sqrt(self.VvarE[u][:,-1,-1]*cutoff),'r',ls='--',color='c')
			plt.axvline(x=-self.fEV[u][-1],label="limit of $|\\overline{\\lambda_m}|$",color='m')
			plt.axvline(x=-self.EV[u][0],label="$|\\overline{\\lambda_m}|$ at $N=$"+str(self.N[u][0]),ls=':',color='m')
			plt.axvline(x=-self.EV[u][-1],label="$|\\overline{\\lambda_m}|$ at $N=$"+str(self.N[u][-1]),ls='-.',color='m')
			plt.axhline(y=0)
			plt.xlabel("$T/J$",fontsize='larger')
			plt.ylabel('var$(E)/N^2$',fontsize='large')
			plt.legend()
			plt.title("Leading Order Behavior of var$(E)$, $u=$"+str(u))
			plt.gca().yaxis.set_major_formatter(formatter)
			plt.savefig(plot_path+'VE'+str(int(u))+'.pdf')

			plt.figure()
			plt.plot(self.Ts,self.fvarS[u][-1,:],'.',label='fit')
			plt.plot(self.Ts,self.fvarS[u][-1,:]+np.sqrt(self.VvarS[u][:,-1,-1]*cutoff),'r',ls='--',color='c',label="Confidence window")
			plt.plot(self.Ts,self.fvarS[u][-1,:]-np.sqrt(self.VvarS[u][:,-1,-1]*cutoff),'r',ls='--',color='c')
			plt.axvline(x=-self.fEV[u][-1],label="limit of $|\\overline{\\lambda_m}|$",color='m')
			plt.axvline(x=-self.EV[u][0],label="$|\\overline{\\lambda_m}|$ at $N=$"+str(self.N[u][0]),ls=':',color='m')
			plt.axvline(x=-self.EV[u][-1],label="$|\\overline{\\lambda_m}|$ at $N=$"+str(self.N[u][-1]),ls='-.',color='m')
			plt.axhline(y=0)
			plt.xlabel("$T/J$",fontsize='larger')
			plt.ylabel('var$(S)/N^2$',fontsize='large')
			plt.legend()
			plt.title("Leading Order Behavior of var$(S)$, $u=$"+str(u))
			plt.gca().yaxis.set_major_formatter(formatter)
			plt.savefig(plot_path+'VS'+str(int(u))+'.pdf')

		plt.show()



