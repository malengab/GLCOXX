from __future__ import print_function
from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mtplt
import copy
import time
import sys
import torch
import torch.optim as optim
import torch.optim.lr_scheduler
###############################################################################
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
###############################################################################
def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)
###############################################################################	
def ties(os_months_ordered): # returns new array indexing: tied times have the same index
	n_samples = len(os_months_ordered)
	tie = np.zeros(n_samples,dtype='int')
	unq = np.unique(os_months_ordered)   # UNIQUE months
	D = dict()
	for (i,x) in enumerate(unq):
		D[x] = i;
	for jj in range(n_samples):
		tie[jj] = D[os_months_ordered[jj]]
	left = 0
	right = 1
	while(right<n_samples-1): # check for repetitions, sliding window
		while (tie[left] == tie[right]):
			right += 1
		reps = right-left-1  # nr of reps in window
		tie[right:] += reps
		left = copy.copy(right) # increase count 		
		right = left+1
	return tie
#####################################################################################
def get_data(cancer_type):
	x=pd.read_csv('surv_files/'+cancer_type+'_x.csv', header=None, low_memory=False).values
	gnames = x[0,1:].astype('str')		# gene names
	os_months=pd.read_csv('surv_files/'+cancer_type+'_os_months.csv',dtype='float32').values
	os_status=pd.read_csv('surv_files/'+cancer_type+'_os_status.csv',dtype='int').values
	pathway_names=pd.read_csv('surv_files/pathway_names.csv').values  # pathway names
	# remove first column = numberin line
	os_months = os_months[:,1:]
	os_status = os_status[:,1:]
	x = x[1:,1:].astype('float32')  # remove first row (names)
	pathway_names = pathway_names[:,1:].astype('str')
	# flatten into a list (not a list of single lists)
	os_months = os_months.flatten()
	os_status = os_status.flatten()
	indices = np.argsort(os_months)
	x_ordered = x[indices][:]
	os_status_ordered = os_status[indices] #np.array(os_status)[indices.astype(int)]
	os_months_ordered = os_months[indices] #np.array(os_months)[indices.astype(int)]
	return x_ordered,os_months_ordered,os_status_ordered, gnames, pathway_names
###############################################################################
def stnd(x_ordered):
	# Standardize
	meansx = torch.mean(x_ordered, axis=0)
	stdsx = torch.std(x_ordered, axis=0)
	stdin = (stdsx>0)   # only consider stdx > 0
	x_ordered[:,stdin] = (x_ordered[:,stdin] - meansx[stdin]) / stdsx[stdin]
	return x_ordered
###############################################################################
def npstnd(x_ordered):
	# Standardize
	meansx = np.mean(x_ordered, axis=0)
	stdsx = np.std(x_ordered, axis=0)
	stdin = (stdsx>0)   # only consider stdx > 0
	x_ordered[:,stdin] = (x_ordered[:,stdin] - meansx[stdin]) / stdsx[stdin]
	return x_ordered
###############################################################################
def prepare_data(cancer_types):
	nc = len(cancer_types)  # how many cancers
	ca = [can() for cc in range(nc)]   # initiate list of cancers
	
	# extract all files
	for cc in range(nc):
		ca[cc].name = cancers_type[cc]   # save c name
		ca[cc].x,ca[cc].os_months,ca[cc].os_status,ca[cc].gnames,ca[cc].pnames = get_data(cancers_type[cc])
		groups = pd.read_csv('surv_files/'+cancers_type[cc]+'_groups.csv').values
		groups = groups[:,1:].flatten()
		ca[cc].groups = groups
		
	# remove non-overlapping genes (pair-wise)
	for cc in range(nc):
		for cc2 in range(cc+1,nc):
			ind_A, ind_B = group_overlapp(ca[cc].gnames,ca[cc2].gnames)	# which indices give shared elements
			print('OBS: Removed '+str(np.sum(~ind_A))+', resp '+str(np.sum(~ind_B))+' genes.')			
			ca[cc].x = ca[cc].x[:,ind_A]
			ca[cc2].x = ca[cc2].x[:,ind_B]
			ca[cc].groups = ca[cc].groups[ind_A]
			ca[cc2].groups = ca[cc2].groups[ind_B]
			ca[cc].gnames = ca[cc].gnames[ind_A]
			ca[cc2].gnames = ca[cc2].gnames[ind_B]

			# check if genes correspond:
			if ~(np.all(ca[cc].gnames == ca[cc2].gnames)):
				raise ValueError('Gene lists do not coincide.')
	
	# count number of samples, genes; normalize
	for cc in range(nc):
		ca[cc].n_samples, ca[cc].n_genes = ca[cc].x.shape
#		ca[cc].x = np.log(ca[cc].x + 1) # log to make it biologically more relevant  # bring BACK!
		ca[cc].x = npstnd(ca[cc].x)
		ca[cc].x = torch.from_numpy(ca[cc].x)
		
	return ca
###############################################################################
###############################################################################
class Cox:
	def __init__(self,c1,beta):
		self.X = c1.x
		if len(self.X.shape) == 1:  # just one gene
			self.n_samples = np.squeeze(self.X.shape)
			self.n_genes = 1
			self.X = self.X[:,np.newaxis]  # need to add an empty dimension
		else:
			self.n_samples, self.n_genes = self.X.shape   # nr of samples and genes in the study
		self.Y = c1.os_status.astype(bool)	# os_status_ordered
		self.Z = c1.os_months 	# os_months_ordered
		self.beta = beta
		self.XB = torch.matmul(self.X,self.beta)#.flatten().astype(float)		# X*beta # make a one-dimensional vector
		self.theta = torch.exp(self.XB)			# exp(X*beta)

		# resolve ties
		self.tie = ties(self.Z)

		# used in l & lp
		self.p = torch.squeeze(torch.flip(torch.unsqueeze(torch.cumsum(torch.squeeze(torch.flip(torch.unsqueeze(self.theta,0),[0,1])),0),0),[0,1]))	# cumulative sum of theta_tilde # flip back to the original ordering
		self.sset = torch.tensor([i for i, x in enumerate(self.Y) if x])    # set of all event indices
		self.tiess = self.tie[self.sset]
				
	def l(self): # Cox negative partial log likelihood/loss function		
		temp3 = self.XB[self.sset] - torch.log(self.p[self.tiess])
		ll = -2/self.n_samples*torch.sum(temp3)	# vanilla loss function
		return ll
		
	def lp(self):		# first derivative wrt beta	
		qq = self.X*self.theta.view(self.n_samples,1)#[:,np.newaxis]   # matrix [theta_1*X_1, ..., theta_n*X_n] (same size as X)
		for ii in range(self.n_samples-2,np.min(self.tiess)-1,-1): # countdown from n-1 to the first deceased patient
			qq[ii]+=qq[ii+1]
		temp3 = self.X[self.sset] - qq[self.tiess]/((self.p[self.tiess]).view(len(self.tiess),1))#[:, np.newaxis]
		llp = -2/self.n_samples*torch.sum(temp3,axis=0) # loss fction derivative

		
		return llp
	
#################################################################################################################
def lasso(beta,glist):   # lasso regression
	ngenes = len(beta)  # number of genes
	gnel = len(glist)	# nr of groups
	weigh_long = torch.zeros(ngenes)  # weight repeated
	grnorm_long = torch.zeros(ngenes)  # norm vector long (with repetitions)
	garray = [np.array(glist[ii]) for ii in range(gnel)]   # list of lists to list of arrays
	grnorm2 = torch.zeros(gnel)#,device=device)	# create an empty bucket for each group norm
	weigh = torch.zeros(gnel)#, device = device)  # group weights
	for jj in range(gnel):	# group norms
		weigh[jj] = np.sqrt(len(glist[jj]))	# group weights = sqrt(length of the group)
		weigh_long[garray[jj]] = weigh[jj]
		grnorm2[jj]=torch.norm(beta[garray[jj]])	# sum squares within each group
		grnorm_long[garray[jj]] = grnorm2[jj]
		
	reg1 = torch.sum(weigh*grnorm2)   # sum groups: group lasso regression	
	
	reg2 = torch.zeros(ngenes)#, device = device)
	indx = torch.abs(grnorm_long) > 1e-6 # nonzero norm
	reg2[indx] = beta[indx]*weigh_long[indx]/grnorm_long[indx]
	
	return reg1 , reg2

######################################################################################
def multireg22(beta1,beta2,glist): # monster coupling term
	tol = 1e-3 #5e-4
	
	lb = len(beta1)
	gnel = len(glist)	# nr of groups
	grlen = torch.zeros(gnel)	# lengths of particular groups
	for ii in range(gnel):
		grlen[ii] = len(glist[ii])	# lengths of particular groups
	weigh = torch.sqrt(grlen)	# group weights = sqrt(length of the group)
	betadif = torch.zeros(gnel)
	beta1norm = torch.zeros(gnel)	# create an empty bucket for each group norm
	beta2norm = torch.zeros(gnel)	# create an empty bucket for each group norm
	p1norm = torch.zeros(gnel)#, dtype=torch.float64)
	p2norm = torch.zeros(gnel)#, dtype=torch.float64)
	p1 = torch.zeros(lb)#, dtype=torch.float64)
	p2 = torch.zeros(lb)#, dtype=torch.float64)
	
	for jj in range(gnel):	# group norms
		beta1norm[jj] = torch.norm(beta1[glist[jj][:]])
		beta2norm[jj] = torch.norm(beta2[glist[jj][:]])
		if (beta1norm[jj]>tol and beta2norm[jj]>tol): # this removes the sign
			p1[glist[jj][:]] = beta1[glist[jj][:]] - beta2[glist[jj][:]]/beta2norm[jj]*beta1norm[jj]
			p2[glist[jj][:]] = beta2[glist[jj][:]] - beta1[glist[jj][:]]/beta1norm[jj]*beta2norm[jj]
		p1norm[jj] = torch.norm(p1[glist[jj][:]])
		p2norm[jj] = torch.norm(p2[glist[jj][:]])
		betadif[jj] = p1norm[jj] + p2norm[jj]
	reg = np.linalg.norm(weigh*betadif)  # weighted 2norm of 2 norms

	regp12 = torch.zeros(lb)#, dtype=torch.float64)
	regp22 = torch.zeros(lb)#, dtype=torch.float64)
	for jj in range(gnel):
		if (beta1norm[jj]>tol and beta2norm[jj]>tol): # this removes the sign
				regp12[glist[jj][:]] = 1/reg*weigh[jj]**2*betadif[jj]*(1/p1norm[jj]*(p1[glist[jj][:]] - beta1[glist[jj][:]]/beta1norm[jj]/beta2norm[jj]*torch.sum(p1[glist[jj][:]]*beta2[glist[jj][:]])) \
											- 1/p2norm[jj]*(beta2norm[jj]/beta1norm[jj]*p2[glist[jj][:]] - beta1[glist[jj][:]]*beta2norm[jj]/(beta1norm[jj]**3)*torch.sum(p2[glist[jj][:]]*beta1[glist[jj][:]])))
				regp22[glist[jj][:]] = 1/reg*weigh[jj]**2*betadif[jj]*(1/p2norm[jj]*(p2[glist[jj][:]] - beta2[glist[jj][:]]/beta2norm[jj]/beta1norm[jj]*torch.sum(p2[glist[jj][:]]*beta1[glist[jj][:]])) \
											- 1/p1norm[jj]*(beta1norm[jj]/beta2norm[jj]*p1[glist[jj][:]] - beta2[glist[jj][:]]*beta1norm[jj]/(beta2norm[jj]**3)*torch.sum(p1[glist[jj][:]]*beta2[glist[jj][:]])))

	return reg, regp12, regp22
########################################################################################################
def multireg24(beta1,beta2,glist): #multireg 20 but better scale
	tol = 5e-4 #5e-4
	
	lb = len(beta1)
	gnel = len(glist)	# nr of groups
	grlen = torch.zeros(gnel)	# lengths of particular groups
	for ii in range(gnel):
		grlen[ii] = len(glist[ii])	# lengths of particular groups
	weigh = torch.sqrt(grlen)	# group weights = sqrt(length of the group)
	betadif = torch.zeros(gnel)
	beta1norm = torch.zeros(gnel)#, dtype=torch.float64)	# create an empty bucket for each group norm
	beta2norm = torch.zeros(gnel)#, dtype=torch.float64)	# create an empty bucket for each group norm
	betadif2 = torch.zeros(lb)#, dtype=torch.float64)
	
	for jj in range(gnel):	# group norms
		beta1norm[jj] = torch.norm(beta1[glist[jj][:]])
		beta2norm[jj] = torch.norm(beta2[glist[jj][:]])
		if (beta1norm[jj]>tol and beta2norm[jj]>tol): # this removes the sign
			betadif2[glist[jj][:]] = (beta1[glist[jj][:]]/beta1norm[jj] - beta2[glist[jj][:]]/beta2norm[jj])*torch.sqrt(beta1norm[jj])*torch.sqrt(beta2norm[jj])
		betadif[jj] = torch.norm(betadif2[glist[jj][:]])
	reg = np.linalg.norm(weigh*betadif)  # weighted 2norm of 2 norms

	regp12 = torch.zeros(lb)#, dtype=torch.float64)
	regp22 = torch.zeros(lb)#, dtype=torch.float64)
	for jj in range(gnel):
		if (beta1norm[jj]>tol and beta2norm[jj]>tol): # this removes the sign
			if reg>tol:
				regp12[glist[jj][:]] = 1/reg*weigh[jj]**2*betadif[jj]*1/betadif[jj]*(betadif2[glist[jj][:]]/torch.sqrt(beta1norm[jj])*torch.sqrt(beta2norm[jj]) \
										 - 1/2*beta1[glist[jj][:]]/(beta1norm[jj]**(5/2))*torch.sqrt(beta2norm[jj])*torch.sum(betadif2[glist[jj][:]]*beta1[glist[jj][:]]) \
										 - 1/2*beta1[glist[jj][:]]/torch.sqrt(beta2norm[jj])/((beta1norm[jj])**(3/2))*torch.sum(betadif2[glist[jj][:]]*beta2[glist[jj][:]]))
				regp22[glist[jj][:]] = -1/reg*weigh[jj]**2*betadif[jj]*1/betadif[jj]*(betadif2[glist[jj][:]]/torch.sqrt(beta2norm[jj])*torch.sqrt(beta1norm[jj]) \
										 - 1/2*beta2[glist[jj][:]]/(beta2norm[jj]**(5/2))*torch.sqrt(beta1norm[jj])*torch.sum(betadif2[glist[jj][:]]*beta2[glist[jj][:]]) \
										 - 1/2*beta2[glist[jj][:]]/torch.sqrt(beta1norm[jj])/((beta2norm[jj])**(3/2))*torch.sum(betadif2[glist[jj][:]]*beta1[glist[jj][:]]))

	return reg, regp12, regp22	
#####################################################################################################################
def group_list(group):	# map groups into lists
	unq = np.unique(group)	# unique group names
	neunq = len(unq)		# n of unique elements

	D = dict() #  mapping the group names to their index
	for (i,x) in enumerate(unq):
		D[x] = i;

	x = [[] for i in range(neunq)]	# empty list of lists
	for jj in range(len(group)):
		x[D[group[jj]]].append(jj)	# append indices of gloup elements into respective lists
	return x, unq
########################################################################################
def GD(method,lr,beta0,c1,lam,epoch):#,*args):
	tol2 = 1e-3	# coefficient cutoff
	c1_copy = copy.deepcopy(c1);
	beta = copy.deepcopy(beta0)

	if method == 'SGD':  # choose optimization method
		optimizer = optim.SGD([beta], lr=lr, momentum=0.5)
	elif method == 'Adam':
		optimizer = optim.Adam([beta], lr=lr)#,betas=(0.7, 0.1))
	elif method == 'Rprop':
		optimizer = optim.Rprop([beta], lr=lr)
	elif method == 'ASGD':
		optimizer = optim.ASGD([beta], lr=lr)
	
	glist_temp, gunq_temp = group_list(c1_copy.groups)	# groups in lists, unique groups
	
	i= 0 # epoch counter
	tol = 1e-4; # stopping criterium
	incr = 10; # initialize increment
	loss_v = np.zeros(epoch)  # loss function, save vector
	loss_old, lossdelta = 100, 100 # initialize to something silly
	
	while incr>tol and i<epoch and lossdelta>tol:
#		beta_old = beta.clone().detach().requires_grad_(True)		
		beta_old = beta.clone().detach().requires_grad_(False)		
		optimizer.zero_grad()
		Cfunc = Cox(c1_copy,beta)		
		lreg, lreg2 = lasso(beta,glist_temp)	# compute lasso regeression
		loss = Cfunc.l() + lam*lreg
		lossdelta = np.abs(loss.clone().detach().numpy()-loss_old)  # increment in loss
#    # backward
#		loss.backward()
		
		beta.grad = Cfunc.lp()+lam*lreg2
		optimizer.step()		
		
#		# element-wise cutoff
#		beta[torch.abs(beta)<tol2] = 0 # elements < tol -> 0

		# group-wise cutoff
		for jj in range(len(glist_temp)):
			if (torch.all(torch.abs(beta[glist_temp[jj]])<tol2)): # all elements in a group < tol
				beta[glist_temp[jj]] = 0
		
		loss_old = loss.clone().detach().numpy()
		loss_v[i] = loss_old
#		score_train = torch.matmul(c1_copy.x,beta)#.flatten().astype(float)
#		cind_train = c_index(score_train,c1_copy.os_status,c1_copy.os_months)			
		incr = (torch.max(torch.abs(beta_old - beta.clone().detach())))/torch.max(torch.abs(beta_old))
		i += 1

	retrn = ret() # return variables
	retrn.beta, retrn.loss = beta.clone().detach(), loss
	return(retrn)
#############################################################################
def GD_gen(beta0_comp,ca,lamv,muv,method,lr,kmax)	:
	loss_delta = 100	# initiate pga stopping criteria
	loss_old = 100 # initialize to something silly
	incr = 100   # initiate increment to something silly
	tol = 1e-6  # tolerance
	tol2 = 1e-3 # tolerance settin beta to 0
	kk = 0 	# initiate interation nr
	nc = len(ca)  # nr of cancers
	
	beta_it = copy.deepcopy(beta0_comp)  # used for iterating
	beta = torch.cat(beta_it)
	
	nonzerogr = [[] for cc in range(nc)] # indices of zero coefficients in every iteration
	nonzero = [np.asarray([ii for ii in range(ca[cc].n_genes)]) for cc in range(nc)]	
	flag, loss_part = np.ones(nc).astype(bool), np.zeros(nc)
	glist_orig, gunq_orig = group_list(ca[0].groups)
##############

	if method == 'SGD':
		optimizer = optim.SGD([beta], lr=lr, momentum=0.5)
	elif method == 'Adam':
		optimizer = optim.Adam([beta], lr=lr)#,betas=(0.7, 0.1))
	elif method == 'Rprop':
		optimizer = optim.Rprop([beta], lr=lr)
	elif method == 'ASGD':
		optimizer = optim.ASGD([beta], lr=lr)
	
################
	while (incr>tol) and (kk<kmax) and (loss_delta>tol):
		beta_old = beta.clone().detach().requires_grad_(False)		
		optimizer.zero_grad()		
		loss, der = 0, [[] for cc in range(nc)]
		# add single contributions
		for cc in range(nc):
			if flag[cc]:
				Cfunc = Cox(ca[cc],beta_it[cc])
				lreg,lpreg = lasso(beta_it[cc],glist_orig)	# compute lasso regeression
				der[cc] = Cfunc.lp() + lamv[cc]*lpreg
				loss_part[cc] = Cfunc.l() + lamv[cc]*lreg

			else:
				der[cc] = []
			loss += loss_part[cc]
				
		# add multitasking
		for cc in range(nc):
			for cc2 in range(cc+1,nc): 
				if (reg=='22'):
					rr, rp1, rp2 = multireg22(beta_it[cc],beta_it[cc2],glist_orig)	# multitask regression terms # OBS doesnt have to recomputed all the time
				elif (reg=='24'):
					rr, rp1, rp2 = multireg24(beta_it[cc],beta_it[cc2],glist_orig)	# multitask regression terms # OBS doesnt have to recomputed all the time
				else:
					raise ValueError('Given reg is not in the list.')			
		
				loss += muv[cc,cc2]*rr
				der[cc] += muv[cc,cc2]*rp1[nonzero[cc]]
				der[cc2] += muv[cc,cc2]*rp2[nonzero[cc2]]

		if np.all(~flag):
			break
		
#		der_long = np.concatenate(der)
#		der_long = torch.cat(der)
#		derold_long = np.concatenate(derold)
					
		beta.grad = torch.cat(der) #der_long
		optimizer.step()		
		
		# hard cutoff
#		beta[torch.abs(beta)<tol2] = 0 # elements < tol -> 0

		beta_it = [[] for cc in range(nc)]
		for ii in range(nc):
			beta_it[ii] = beta[ii*ca[0].n_genes:(ii+1)*ca[0].n_genes]

		# introduce group-wise cutoff
		for cc in range(nc):
			for jj in range(len(glist_orig)):
				if (torch.all(torch.abs(beta_it[cc][glist_orig[jj]])<tol2)): # all elements in a group < tol
					beta_it[cc][glist_orig[jj]] = 0
		
		incr = (torch.max(torch.abs(beta_old - beta.clone().detach())))/torch.max(torch.abs(beta_old)) #!!
		
		# update beta and increment
		loss_temp = loss.clone().detach().numpy()
		loss_delta = np.abs(loss_temp-loss_old)	# check for stopping criteria
		loss_old = copy.copy(loss_temp)
											
		kk += 1	# increase iteration count
		if(kk == kmax):
			eprint('GD did not converge for some values of lambda.');

	return beta_it, nonzero, nonzerogr
##################################################################################################
def c_index(score,y,z): # concordance index, must be in increasing order (time-wise): ties are ignored
	conc, disc = 0, 0  # initialize concordant and discordant pairs
	for jj in range(len(score)):
		t1 = z[jj]  # time1
		for kk in range(jj+1,len(score)):
			t2 = z[kk]	# time2
			if(t1!=t2): 	# if events/censoring happened at different times
				if((y[jj]==1) and (y[kk]==1)):	# both have died
					if(score[jj]>score[kk]):	# concordant pairs
						conc += 1
					else:
						disc += 1
				elif((y[jj]==1) and (y[kk]==0)): 	# first patient died, second later censored
					if(score[jj]>score[kk]):
						conc += 1
					else:
						disc += 1
	if(conc+disc==0):	# no concordant or discordant pairs
		c_index = 0
	else:
		c_index = conc/(conc+disc)	# Harrell's concordance index
	return c_index #, conc, disc
##################################################################################################
def bord(K,n_samples): # find borders of K splits on n_samples elements
	fold_length = np.floor(n_samples/K)  # fold length: split n_samples into K pieces
	kvar = n_samples % K	# modulo leftover
	borders = np.zeros(1)	# start with left end point
	for ii in range(1,K):
		if(ii<=kvar):
			borders = np.append(borders,int(ii * fold_length + ii))	# distribute the modulo remainder in a cumulative way
		else:
			borders = np.append(borders,int(ii * fold_length + kvar))	# after kvar is distributed everything is shifted with kvar
	borders = np.append(borders,n_samples)  # add right end point
	return borders
####################################################################################################
def CV_lambda1(c1,K,lam,beta0,rseed,c1done,lr,method,epoch):
#	nonzgr=[]	# collect nonzero groups for analysis
	score = np.zeros(c1.n_samples)	# store the score
	
	# random shuffling of the data
	indices = np.arange(c1.n_samples)
	np.random.seed(seed=rseed)
#	torch.manual_seed(rseed)
	np.random.shuffle(indices)
	
	x_shuffle = c1.x[indices]
	y_shuffle = c1.os_status[indices]	#os_status_ordered
	z_shuffle = c1.os_months[indices]	#os_months_ordered
	
	borders = bord(K,c1.n_samples)	# borders of folds
	beta0new = torch.zeros((c1.n_genes,K))	# initiate beta0 to 0
	cav = 0 # average c-index
	
	for ii in range(K):

		left = int(borders[ii])   # include left endpoint
		right = int(borders[ii+1]) # exclude right endpoint // that's why we artificially added the righmost point +1

  	############### Cross-validation groups
		testin = range(left,right)   # test indies
		# training set indices
		if(left==0):  # if we are in the first set
				trainin = range(right,c1.n_samples)
		elif(right==c1.n_samples):  # we are in the last set
				trainin = range(0,left) 
		else:  # otherwise
				trainin = np.append(range(0,left),range(right,c1.n_samples))
	#		print(testin, trainin)

		x_test_shuffle = x_shuffle[testin]  # D_k into test set
		x_train_shuffle = x_shuffle[trainin]  # rest into training set
		y_test_shuffle = y_shuffle[testin]
		y_train_shuffle = y_shuffle[trainin]
		z_train_shuffle = z_shuffle[trainin]
		z_test_shuffle = z_shuffle[testin]		
	
		in1 = indices[testin]	# choose test and train indices from the shuffled array
		in2 = indices[trainin]
		
		in1_sorted = torch.from_numpy(np.sort(in1))	# sort the test indices array
		in1_order = torch.from_numpy(np.argsort(in1))	# recod the sorting order
		
		in2_order = torch.from_numpy(np.argsort(in2))
				
		c1_test, c1_train = can(), can()
				
		c1_test.x = x_test_shuffle[in1_order]	# sort the test and train separately
		c1_test.os_status = y_test_shuffle[in1_order]	
		c1_test.os_months = z_test_shuffle[in1_order]
		c1_train.x = x_train_shuffle[in2_order]
		c1_train.os_status = y_train_shuffle[in2_order]
		c1_train.os_months = z_train_shuffle[in2_order]
		
		c1_test.n_samples, c1_test.n_c1_testgenes = c1_test.x.shape
		c1_train.n_samples, c1_train.n_genes = c1_train.x.shape
		c1_train.groups = c1.groups
			
		###########
		if (c1done==False):
			beta0temp = beta0[:,ii].clone().detach().requires_grad_(False)
			retrn = GD(method,lr,beta0temp,c1_train,lam,epoch)
			beta_train= retrn.beta
		else:
			beta_train = beta0[:,ii]

#		nonzgr.append(nonzerogr)	# keep a list of all nonzero groups
		beta0new[:,ii] = beta_train	# will use it as initial data in the next step
				
		score_test = torch.matmul(c1_test.x,beta_train)#.flatten().astype(float)
		score[in1_sorted] = score_test	
		cav += 1/K*c_index(score_test,c1_test.os_status,c1_test.os_months)
#		print('Nonzeros: '+str(len(nonzero))+' in fold '+str(ii+1)+'/'+str(K))
	return beta0new, score, cav
##############################################################################################################		
def CV_lambda_gen(ca,K,lamv,beta0_comp,rseed,muv,method,lr,epoch):
	nc = len(ca)  # nr of cancers
	n_samples, score, indices, x_shuffle, y_shuffle, z_shuffle = [],[],[],[],[],[]
	borders, beta0new = [],[]
	for cc in range(nc):
		ns_temp = ca[cc].n_samples  # nr of samples
		n_samples.append(ns_temp)   # store nr of samples
		score.append(np.zeros(ns_temp))  # store the score	
#	score = [np.zeros(n_samples[ii]) for ii in range(nc)]  # store the score	
		ind_temp = np.arange(ns_temp) 
		np.random.seed(seed=rseed)
		np.random.shuffle(ind_temp) # shuffle indices
		indices.append(ind_temp)  # store indices
		x_shuffle.append(ca[cc].x[ind_temp])
		y_shuffle.append(ca[cc].os_status[ind_temp]) 
		z_shuffle.append(ca[cc].os_months[ind_temp])#os_months_ordered
		borders.append(bord(K,ns_temp))	# borders of folds
		beta0new.append(np.zeros((ca[cc].n_genes,K)))	# initiate beta0 to 0
	cav = np.zeros([nc]) # average c-index
	nonz_len_av = np.zeros(nc)  # how many nonzero on average
	nonzgr = [[] for cc in range(nc)] # which groups are not zero

	for ii in range(K):
		testin, trainin, c_train, c_test, in1_sorted = [],[],[],[],[]
		for cc in range(nc):
#		print('Fold '+str(ii+1)+' out of '+str(K))

			left = int(borders[cc][ii])   # include left endpoint
			right = int(borders[cc][ii+1]) # exclude right endpoint // that's why we artificially added the righmost point +1
			
			############### Cross-validation groups
			testin_temp = range(left,right)   # test indies
			testin.append(testin_temp)
			# training set indices
			if(left==0):  # if we are in the first set
				trainin_temp = range(right,n_samples[cc])
			elif(right==n_samples[cc]):  # we are in the last set
				trainin_temp = range(0,left) 
			else:  # otherwise
				trainin_temp = np.append(range(0,left),range(right,n_samples[cc]))
			trainin.append(trainin_temp)

			x_test_shuffle_temp = x_shuffle[cc].squeeze()[testin_temp]  # D_k into test set
			x_train_shuffle_temp = x_shuffle[cc].squeeze()[trainin_temp]  # rest into training set
			y_test_shuffle_temp = y_shuffle[cc].squeeze()[testin_temp]
			y_train_shuffle_temp = y_shuffle[cc].squeeze()[trainin_temp]
			z_train_shuffle_temp = z_shuffle[cc].squeeze()[trainin_temp]
			z_test_shuffle_temp = z_shuffle[cc].squeeze()[testin_temp]	

			in1_temp = indices[cc][testin_temp]	# choose test and train indices from the shuffled array
			in2_temp = indices[cc][trainin_temp]

			in1_sorted_temp = np.sort(in1_temp)	# sort the test indices array
			in1_sorted.append(in1_sorted_temp)
			in1_order_temp = np.argsort(in1_temp)	# recod the sorting order
		
#		in2_sorted = np.sort(in2)
			in2_order_temp = np.argsort(in2_temp)
			
			c1_test_temp, c1_train_temp = can(), can()
				
			c1_test_temp.x = x_test_shuffle_temp[in1_order_temp]	# sort the test and train separately
			c1_test_temp.os_status = y_test_shuffle_temp[in1_order_temp]
			c1_test_temp.os_months = z_test_shuffle_temp[in1_order_temp]
			c1_train_temp.x = x_train_shuffle_temp[in2_order_temp]
			c1_train_temp.os_status = y_train_shuffle_temp[in2_order_temp]
			c1_train_temp.os_months = z_train_shuffle_temp[in2_order_temp]
		
			if len(c1_test_temp.x.shape) == 1:   # just one gene 
				c1_test_temp.n_samples = c1_test_temp.x.shape
				c1_test_temp.n_genes = 1
				c1_train_temp.n_samples = c1_train_temp.x.shape
				c1_train_temp.n_genes = 1
				c1_train_temp.groups = ca[cc].groups
			else:
				c1_test_temp.n_samples, c1_test_temp.n_genes = c1_test_temp.x.shape
				c1_train_temp.n_samples, c1_train_temp.n_genes = c1_train_temp.x.shape
				c1_train_temp.groups = ca[cc].groups
			
			c_train.append(c1_train_temp)
			c_test.append(c1_test_temp)
############
		beta0_temp = [beta0_comp[cc][:,ii] for cc in range(nc)]
		beta_train, nonzero, nonzerogr = GD_gen(beta0_temp,c_train,lamv,muv,method,lr,epoch)	# compute the best beta for the training set given lambda
	
		for cc in range(nc):	
			beta0new[cc][:,ii] = beta_train[cc]

			score_test = np.dot(c_test[cc].x,beta_train[cc])#.flatten().astype(float)
			score[cc][in1_sorted[cc]] = score_test
			cav[cc] += 1/K*c_index(score_test,c_test[cc].os_status,c_test[cc].os_months)
			nonz_len_av[cc] += 1/K*len(nonzero[cc])   # average nr of selected genes
			nonzgr[cc].append(nonzerogr[cc])   # nonzero groups
			
#		score_test = np.dot(c_test[0].x,beta_train)#.flatten().astype(float)
#		score[0][in1_sorted[0]] = score_test
#	
#		score_test2 = np.dot(c_test[1].x,beta_train)#.flatten().astype(float)
#		score[1][in1_sorted[1]] = score_test2

	return beta0new, score, cav, nonz_len_av, nonzgr    
##################################################################################
def CV1(c1,K,lamdata,beta0,rseed,gamma00,method,epoch):   # estimate goodness of fit

	m1 = len(lamdata)
	cdatanew = torch.zeros(m1)
	betabest = copy.deepcopy(beta0)
	betabest_all = torch.zeros((m1,K,c1.n_genes))
	c1_copy = copy.deepcopy(c1)
	cav_v = torch.zeros(m1) # average c-index vector
	c1done = False
	
	for jj in range(m1):
		lam = lamdata[jj]	# current lambda
		print('Lambda = '+str(lam)+' : '+str(jj+1)+' / '+str(m1))
		if (K==1):
			retrn = GD(method,gamma00,beta0,c1_copy,lam,epoch)
			betabest = retrn.beta;
			betabest_all[jj,0,:] = betabest;
			cav_v = 0
			cdatanew = 0 # only for 1 point
		else:		  
			beta0new ,score, cav = CV_lambda1(c1_copy,K,lam,beta0,rseed,c1done,gamma00,method,epoch) 
			cav_v[jj] = cav
			print('c-index = '+str(cav))
			for kkk in range(K):
				betabest_all[jj,kkk,:] = beta0new[:,kkk]
#			avnel = torch.sum([len(nonzgr[x]) for x in range(len(nonzgr))])/len(nonzgr)  # average number of groups
			if warmstart == True:
				beta0 = copy.copy(beta0new)   # warm start
			ctemp = c_index(score,c1.os_status,c1.os_months)
			cdatanew[jj] = ctemp

	retrn = ret()
	retrn.cval = cdatanew
	retrn.cav_v, retrn.beta_v = cav_v, betabest_all         
	return retrn
####################################################################################################################
def CV_rand(ca,K,lam_comp,beta0_comp,mu_comp,rseed,method,lr,epoch):   # estimate goodness of fit
	nc = len(ca)  # nr of cancers
	m = len(lam_comp) # nr of parameter combinations
	cdatanew = np.zeros((m,nc))
	cavm = np.zeros([m,nc])
	betabest = []
	nonz_len_av_v = np.zeros([nc,m])
	nonzgr_v = [[] for jj in range(m)]   # selected groups for all points
	for jj in range(m):
		cav = np.zeros([nc])
		c_train = copy.deepcopy(ca) # otherwise it gets overwritten
		lamv = lam_comp[jj]
		muv = mu_comp[jj]
		print('Lambda = '+str(lamv)+' : '+str(jj+1)+' / '+str(m), 'Mu = '+str(muv)+' : '+str(jj+1)+' / '+str(m))
		if torch.all(muv==0):
			beta0new = [[] for cc in range(nc)]
			for cc in range(nc):
				retrn = CV1(ca[cc],K,np.array([lamv[cc]]),beta0_comp[cc],rseed,gamma0[cc],method,epoch)
				beta0new[cc] = retrn.beta
		else:		
			if (K==1):
				beta0new, nonzero, nonzerogr = GD_gen(beta0_comp,c_train,lamv,muv,method,lr,epoch)	# compute the best beta for the training set given lambda
				score = []
				nonz_len_av = np.zeros(nc)
				for cc in range(nc):
					score_test = np.dot(ca[cc].x,beta0new[cc])#.flatten().astype(float)
					score.append(score_test)
					nonz_len_av[cc] = len(nonzero[cc])
					nonzgr = copy.copy(nonzero)
			else:
				beta0new, score, cav, nonz_len_av, nonzgr = CV_lambda_gen(ca,K,lamv,beta0_comp,rseed,muv,method,lr,epoch)				
			for cc in range(nc):
				ctemp = c_index(score[cc],ca[cc].os_status,ca[cc].os_months)
				nonz_len_av_v[cc,jj] = nonz_len_av[cc]
				cdatanew[jj,cc] = ctemp
			nonzgr_v[jj].append(nonzgr)
		cavm[jj,:] = cav  # averaged C-value
		betabest.append(beta0new)
		print(cav)                 
	return cdatanew, betabest, cavm, nonz_len_av_v, nonzgr_v
######################################################################################
def group_overlapp(gnames,gnames2):		# find which genes to remove to obtain identical sets
	dif1 = set(gnames).difference(set(gnames2)) #for (gnames - gnames2)
	dif2 = set(gnames2).difference(set(gnames)) #for (gnames2 - gnames)
		
	indices_A = np.zeros(len(gnames)).astype('bool')
	indices_B = np.zeros(len(gnames2)).astype('bool')
	for ii in range(len(dif1)):
		indices_A += gnames == dif1.pop()   # add all occurences of a gene in difference set
	for jj in range(len(dif2)):
		indices_B += gnames2 == dif2.pop()
	
	indices_A = ~indices_A  # flip True/False
	indices_B = ~indices_B  
	
	return indices_A, indices_B
###############################################################################
def lam0(c):
	x = c.x
	os_stat = c.os_status
	ne = c.n_samples
	xdif = torch.zeros([c.n_genes])
	for ii in range(ne):
		if (os_stat[ii] == 1):
			ci = ne-ii
			for jj in range(ii,ne):
				xdif += (x[ii]-x[jj])/ci
	return 2/(c.n_samples)*xdif
###############################################################################
def toy(n_genes,n_samples,rseed,*args):   # generate toy data	
	# args:
	# non_index : the list of nonzeros
	beta = np.zeros(n_genes)	# initiate beta
	if (len(args)==1):   # if only position is specified
		non_index = args[0]
		nnon = len(non_index)
		np.random.seed(seed=rseed)
		beta[non_index] = 2*(np.random.random(nnon) - .5)
	elif(len(args)==2):	 # if even the beta values are specified
		non_index = args[0]
		nnon = len(non_index)
		beta[non_index] = args[1]		
	else: 		# if nothing is specified
		nnon = 3   # nr of nonzeros
		np.random.seed(seed=rseed)
		non_index = np.random.randint(n_genes, size = nnon)
		np.random.seed(seed=rseed)
		beta[non_index] = 2*(np.random.random(nnon) - .5)
#	np.random.seed(seed=rseed)
	torch.manual_seed(rseed)	
	x = torch.rand(n_samples,n_genes)
	x = stnd(x)
	score = np.dot(x,beta)  # could be used as months
	idx = (-score).argsort()
#	score_sorted = score[idx]
	x_sorted = x[idx]
	months = np.array([ii for ii in range(n_samples)])
	status_sorted = np.ones(n_samples)
#	status_sorted = np.concatenate((np.ones(int(n_samples/2)),np.zeros(int(n_samples/2))))
#	status_sorted = np.random.randint(2, size=n_samples)
	ca = can()
	ca.n_genes = n_genes
	ca.n_samples = n_samples
	ca.name = 'toy'
	ca.os_months = months
	ca.os_status = status_sorted
	ca.x = x_sorted
	
	return ca, beta
##################################################################################
def grid(lammin,lammax,m,scale):  # generate 1D grid between lmin and lmax, either linear or logarithmic
	if (scale=='log'):
		if(m==1):
			lamdata = np.array([lammax])
		else:
			lamdata = [lammin*((lammax/lammin)**(j/(m-1))) for j in range(m)]	# all lambdas that we will test
	elif (scale=='lin'):
		lamdata = np.linspace(lammin,lammax,m)
	return lamdata
##################################################################################
def lamminmax(cinddata, lamdata):
	done1, ll1 = False, 1  # initiate max position
	done2, ll2 = False, 0  # initiate min position
	while(done1==False): # while max not found
		if (cinddata[-ll1]>1e-9):
#			lammin = lamdata[-ll1]
			lammax = lamdata[-ll1]
			done1 = True
		else:
			ll1 += 1
	while(done2==False): # while max not found
		if (cinddata[ll2]>1e-9):
#			lammax = lamdata[ll2]
			lammin = lamdata[ll2]
			done2 = True
		else:
			ll2 += 1
	return lammin,lammax
#####################################################################################
def congr(a,b):  # Tucker congruence coefficient
	rc = np.dot(a,b)/np.linalg.norm(a)/np.linalg.norm(b)
	return rc
#####################################################################################
def cong_av(bigvec):  # Tucker congruence coefficient
	nc,nrv,npars = bigvec.shape
	rcvec = np.zeros(nc)
	for cc in range(nc):
		rc = 0  # initialize
		for ii in range(nrv):
			for jj in range(ii+1,nrv): # all combinations only once
				rc += congr(bigvec[cc,ii,:],bigvec[cc,jj,:])
		rcvec[cc] = 2/nrv/(nrv-1)*rc
	return rcvec
#######################3#####3#####################################################
def corr_av(bigvec):  # correlation coefficient
	nc,nrv,npars = bigvec.shape
	rcvec = np.zeros(nc)
	for cc in range(nc):
		rc = 0  # initialize
		for ii in range(nrv):
			for jj in range(ii+1,nrv): # all combinations only once
				rc += np.corrcoef(bigvec[cc,ii,:],bigvec[cc,jj,:])[0,1]
		rcvec[cc] = 2/nrv/(nrv-1)*rc
	return rcvec
###################################################################################
def ldata(rpoints,nc,lammi,lamma,lambest0):   # generate lambda data
	lamdata_rand = [[] for ii in range(rpoints)]
	sigma = 0.005
	for jj in range(rpoints):
		for cc in range(nc):
			notdone_temp = True
			while notdone_temp:
				temp = np.random.normal(lambest0[cc], sigma,1)  # random normal distribution
				if ((temp[0]>lammi[cc]) and (temp[0]<lamma[cc])):  # if inbetween boundaries
					lamdata_rand[jj].append(temp[0])
					notdone_temp = False
	return lamdata_rand
###################################################################################
def mdata(rpoints,nc,lambest0):   # generate mu data: random
	ii = 0
	mu_max = .5 #1/2;
	mudata_rand = []
	while ii<rpoints:
		mu_temp = np.zeros([nc,nc])
		for ll in range(nc):
			for kk in range(ll+1,nc):
				ttemp = -10 # initiate to stng funny
				while ((ttemp<0) or (ttemp>mu_max)): # until within boundaries
					ttemp = np.abs(np.random.normal(0,mu_max/3,1)) # random normal dist
				mu_temp[ll,kk] = ttemp;
		mudata_rand.append(torch.from_numpy(mu_temp))
		ii+=1
	return mudata_rand
###################################################################################
###################################################################################
def mdata_reg(rpoints,nc,mmin,mmax):   # generate mu data: equispaced
	ii = 0
	hdata = grid(mmin,mmax,rpoints,'log')
	mudata_rand = []
	while ii<rpoints:
		mu_temp = np.zeros([nc,nc])
		for ll in range(nc):
			for kk in range(ll+1,nc):
				mu_temp[ll,kk] = hdata[ii]
		mudata_rand.append(torch.from_numpy(mu_temp))
		ii+=1
	return mudata_rand
###############################################################################
def validation(ca,q,*args): # q proportion of training-test, between 0 and 1
	if len(args)==1:
		rseed_local = args[0]
#		np.random.seed(seed=rseed)
	nc = len(ca) # nr of cancers
	### training and validation sets
	ca_train = [can() for cc in range(nc)]   # initiate list of cancers /training
	ca_val = [can() for cc in range(nc)]   # initiate list of cancers  /validation
	
	for cc in range(nc):
		ca_train[cc].name = ca[cc].name
		ca_val[cc].name = ca[cc].name
### create a validation set
		if 'rseed_local' in locals():  # if random seed set
			torch.manual_seed(rseed_local)
		sortset = torch.sort(torch.randint(0,len(ca[cc].os_months),(int(np.ceil(len(ca[cc].x)*q)),)))
		valset = torch.unique(sortset.values) # a subsample as validation set
		trainset = torch.ones(ca[cc].n_samples, dtype=bool)
		trainset[valset] = False
		ca_val[cc].x = ca[cc].x[valset,:]
		ca_val[cc].os_status = ca[cc].os_status[valset]
		ca_val[cc].os_months = ca[cc].os_months[valset]
		ca_val[cc].groups = ca[cc].groups
		ca_val[cc].gnames = ca[cc].gnames
		ca_val[cc].pnames = ca[cc].pnames
		ca_val[cc].n_samples, ca_val[cc].n_genes = ca_val[cc].x.shape
		ca_train[cc].x = ca[cc].x[trainset]
		ca_train[cc].os_status = ca[cc].os_status[trainset]
		ca_train[cc].os_months = ca[cc].os_months[trainset]
		ca_train[cc].groups = ca[cc].groups
		ca_train[cc].gnames = ca[cc].gnames
		ca_train[cc].pnames = ca[cc].pnames
		ca_train[cc].n_samples, ca_train[cc].n_genes = ca_train[cc].x.shape
		
	return ca_train, ca_val
########################################################################################		
class can( object ):   # create class to hold cancer attributes
	def __init__( self ):
		self.name = str
		self.x = float   # ordered
		self.os_months = float  # ditto
		self.os_status = bool   # ditto
		self.gnames = str
		self.pnames = str
		self.groups = int
		self.n_samples = int
		self.n_genes = int
#######################################################################	
class ret( object ):   # create class to hold return variables
	def __init__( self ):
		self.betabest = float  # best beta vector, full length with 0 padding
		self.cval = float      # c-values
		self.beta_fin = float  # full length beta with 0 padding
		self.loss = float      # loss function
		self.nonzero = int     # index of nonzero gene coefficients
		self.nonzerogr = int   # index of nonzero pathway groups
		self.beta = float      # general bucket for betas
		self.beta_v = float    # to save all betas throughout the iterations
		self.loss_v = float    # loss function for all iterations
		self.lambest = float   # one best lamma float
		self.cbest = float     # one best c value float
		self.cav_v = float     # vector of average c-indices
########################################################################################
# download and prepare data

#cancers = [['luad','stad'],['coadread','stad']]
#cancers = [['toy1','toy2']]
#cancers = [['ov','paad']]
cancers = [['toy_luad','toy_paad']]

for cancers_type in cancers:
	
#	gamma0 = [0.0001,0.0001]  # learning rates
	gamma0 = [0.0005,0.0005]  # learning rates
	################ collect all parameters ###############################################
	warmstart = False #True  # in 1D case
	visualisation = True #False  # visualise betas for 2 cancers
		
	## single parameter search: boundaries
	K1 = 10#3#2 # 5 #10  # nr of cross-validation folds # in single parameter search
	reg = '22' # '12', '11' # '7' #'10' #'1' # '9' # '13' # '16'
	
	## define parameter sets
	rpoints = 30#30#30#30 #30 #50   # nr of random points
	
	# select best
	nrv = 5#30 #5#30#20#20#1#5 #5 #5 # 20  # how many validation sets
	nk = 1#2#3#1#3 #3 # 3 # nk CV repeats
	rseeddata2 = np.array([60])+np.reshape(np.arange(nk*nrv),[nk,nrv]) #np.array([42])+np.reshape(np.arange(nk*nrv),[nk,nrv])
	rseeddata_val = np.arange(242,242+nrv)
	rseed_test_data = np.arange(500, 500 + nrv)   # initial data for testing
	
	K2 = 10#5 # 5 to choose best lambda + multitask CV+ single CV
	linpoints = 10#5#3#5 #10		# if stat = single, nr of steps between min and max lambda
	
	method = 'Adam' #'SGD' # Adam # 'Rprop' # 'ASGD'
	epoch = 300  # max nr of epochs
	lr = np.min(gamma0)#0.0001
	
	#####################################################################################################
	### create can classes
	#####################################################################################################
	#
	if not (cancers_type == ['toy1','toy2']):
		nc = len(cancers_type)  # nr of cancers
		ca = prepare_data(cancers_type)  # load, log and trim the data
	else:
		nc = 2
		ca = [can(), can()]
		n_genes = 10000  # number of genes
		imp1 = 500 # nr of important genes of can1
		imp2 = 300 # nr of important genes can2
		ca[0], beta1 = toy(n_genes,1000,47,np.arange(imp1),np.ones(imp1))#,[1,2,1,1,1])
		ca[1], beta2 = toy(n_genes,300,54,np.arange(imp2),np.ones(imp2))
		gr = np.floor(np.arange(n_genes)/100)
		ca[0].groups = gr
		ca[1].groups = gr
	
	npar = nc + nc**2  # nr of parameters in total lambda + mu^2
	parsel = np.zeros([nc,nrv,npar])  # save final selected parameters
	beta_single_vec = np.zeros([nc,nrv,ca[0].n_genes])  # save final betas
	beta_multi_vec = np.zeros([nc,nrv,ca[0].n_genes])  # save final betas
			
	#############################################################################################
	#### NESTED CV ##############################################################################
	#############################################################################################
	tic()
	nr = len(rseeddata2)   # nr of repeats
	cval_vec_multi = np.zeros([nc,nrv]) # save all validation set results
	cval_vec_single = np.zeros([nc,nrv]) # save all validation set results
	
	for nn in range(nrv):
		allpar_val = [np.zeros([nc,rpoints]) for ii in range(nr)]	  # save c-values for all parameters
		bestpar = [[[] for rr in range(nr)] for cc in range(nc)]
		rseed_val = rseeddata_val[nn]				
		#		### training and validation sets
		ca_train, ca_val = copy.deepcopy(validation(ca,0.2,rseed_val))	
		
		###############################################################################
		####### get good boundaries for lambda from single run  #######################
		###############################################################################
		cbest0, lambest0, lammi, lamma = np.zeros(nc), np.zeros(nc), np.zeros(nc), np.zeros(nc)
		lamall = np.zeros([nc,linpoints])
		if (nr > 1):  # save one computation by recycling the old results
			for cc in range(nc):
				xdif = lam0(ca_train[cc])
				lammax = np.max([torch.abs(torch.max(xdif)),torch.abs(torch.min(xdif))])# 0.05 #ov 0.3  #, brca 0.05 # 0.04 prad, paad 0.3
				lammin = 0.1*lammax #0.005
				lamdata = grid(lammin,lammax,linpoints,'log')

				torch.manual_seed(rseed_val)  # set random seed
				beta0 = 0.001*torch.rand(ca_train[cc].n_genes,K1)	# initialize beta to random small
				retrn = CV1(ca_train[cc],K1,lamdata,beta0,rseed_val,gamma0[cc],method,epoch)  # mu=0
				cbest0[cc] = torch.max(retrn.cav_v)
				lambest0[cc] = lamdata[torch.argmax(retrn.cav_v)]
				lammi[cc], lamma[cc] = lamminmax(retrn.cav_v.flatten(),lamdata)
				lamall[cc,:] = lamdata
		##		
			lamdata_rand = ldata(rpoints,nc,lammi,lamma,lambest0)   # produce random lambda set within boundaries
			mudata_rand = mdata(rpoints,nc,lambest0)  # random mu samples between 0 and 3
			
	#		lamdata_rand = [lambest0 for kk in range(rpoints)]
	#		mudata_rand = mdata_reg(rpoints,nc,lambest0) 
		
	#		mudata_rand = mdata_reg(rpoints,nc,1e-3,np.min([lambest0[cc] for cc in range(nc)])) 
	#		cmult = np.array([(1+lambest0[1]/lambest0[0])**(-1), lambest0[1]/lambest0[0]*(1+lambest0[1]/lambest0[0])**(-1)])
	#		lamdata_rand = [lambest0-mudata_rand[kk][0,-1].numpy()*cmult for kk in range(rpoints)]
		
	#		np1 = 4; # nr of mus
	#		np2 = 6; # nr of lambdas
	#		muu = mdata_reg(np1,nc,1e-3,1e-0) 
	##		laam1 = grid(lammi[0],lamma[0],np2,'log')
	##		laam2 = grid(lammi[1],lamma[1],np2,'log')
	#		laam1 = grid(0.05,0.3,np2,'log')
	#		laam2 = grid(0.03,0.2,np2,'log')
	#		l1,l2 = np.meshgrid(laam1,laam2)  # meshgrid
	#		l1 = l1.flatten()
	#		l2 = l2.flatten()
	#		mudata_rand = [muu[ii] for ii in range(np1) for kk in range(np2**2)]
	#		lamdata_rand = [np.array([l1[ii],l2[ii]]) for kk in range(np1) for ii in range(np2**2)]
	#			
	#	# single run
	#		np1 = 1; # nr of mus
	#		np2 = 10; # nr of lambdas
	#		mut = torch.zeros((2,2))
	#		mut[0,1] = 1e-6
	#		muu = [mut for ii in range(np2)]
	#		laam1 = grid(0.05,0.3,np2,'log')
	#		laam2 = grid(0.03,0.2,np2,'log')
	#		
	#		mudata_rand = copy.deepcopy(muu) #[muu[ii] for ii in range(np1) for kk in range(np2)]
	#		lamdata_rand = [np.array([laam1[ii],laam2[ii]]) for ii in range(np2)]
			
		allpar_av_multi = [np.zeros([nc,rpoints]) for ii in range(nr)]	  # save c-values for all parameters
		allpar_av_single = [np.zeros([nc,linpoints]) for ii in range(nr)]	  # save c-values for all parameters
		for kk in range(nr): # do an average over nr runs
			# random samples to determine the best combinations
			rseed = rseeddata2[kk,nn]
		################################################################################
			## multitasking
			# initial data
			if K2==1:
				beta0new = [np.zeros((ca_train[cc].n_genes,K2)).flatten() for cc in range(nc)]	# initialize beta to 0
			else:
				torch.manual_seed(rseed)
				beta0new = [0.001*torch.rand(ca_train[cc].n_genes,K2) for cc in range(nc)]	# initialize beta to small random
	################################################################################
			# compare to best single run
			for cc in range(nc):  # single runs to find best c-values with given best hyperparameters
				if nr == 1:  # only one repeat
					xdif = lam0(ca_train[cc])
					lammax = np.max([torch.abs(torch.max(xdif)),torch.abs(torch.min(xdif))])# 0.05 #ov 0.3  #, brca 0.05 # 0.04 prad, paad 0.3
					lammin = 0.1*lammax #0.005
					lamdata = grid(lammin,lammax,linpoints,'log')
					lamall[cc,:] = lamdata
				retrn = CV1(ca_train[cc],K2,lamall[cc,:],beta0new[cc],rseed,gamma0[cc],method,epoch)  # mu=0
				if nr == 1:
					cbest0[cc] = torch.max(retrn.cav_v)
					lambest0[cc] = lamdata[torch.argmax(retrn.cav_v)]
					lammi[cc], lamma[cc] = lamminmax(retrn.cav_v.flatten(),lamdata)	
	#####				
				allpar_av_single[kk][cc,:] = retrn.cav_v

			lamdata_rand = ldata(rpoints,nc,lammi,lamma,lambest0)   # produce random lambda set within boundaries
			mudata_rand = mdata(rpoints,nc,lambest0)  # random mu samples between 0 and 3
				
	################################################################################			
			# multi
			cinddata, betabest, cavm, nonz_len_av_v, nonzgr_v = CV_rand(ca_train,K2,lamdata_rand,beta0new,mudata_rand,rseed,method,lr,epoch)  # mu=0
			for cc in range(nc):
						allpar_av_multi[kk][cc] = cavm[:,cc]		
	#	###########################################################################
		allpar_weight_av_multi = np.zeros([nc,rpoints])  # select the most upvoted parameter set # in use
		for mn in range(rpoints): # in uselam_cv # in use
			for cc in range(nc):
				ctemp_av = np.zeros(nr)
				for kk in range(nr):
					ctemp_av[kk] = allpar_av_multi[kk][cc][mn]  # use
				allpar_weight_av_multi[cc,mn] = np.sum(ctemp_av)
	
		allpar_weight_av_single = np.zeros([nc,linpoints])  # select the most upvoted parameter set # in use
		for mn in range(linpoints): # in uselam_cv # in use
			for cc in range(nc):
				ctemp_av = np.zeros(nr)
				for kk in range(nr):
					ctemp_av[kk] = allpar_av_single[kk][cc][mn]  # use
				allpar_weight_av_single[cc,mn] = np.sum(ctemp_av)
		lam_cv_multi, mu_cv_multi = [torch.zeros(nc) for cc in range(nc)], [torch.zeros([nc,nc]) for cc in range(nc)]  # initiate CV lambda, mu # in use
		lam_cv_single = torch.zeros(nc) # initiate CV lambda, mu  # in use2
	
	#	xpts = np.array([mudata_rand[ii][0,1] for ii in range(rpoints)])
	#	plt.semilogx(xpts,cavm[:,0],'o-',xpts,cavm[:,1],'*-')
	##	plt.semilogx(laam1,cavm[:,0],'o-')
	##	plt.semilogx(laam2,cavm[:,1],'*-')
	##			plt.legend(['lihc','coadread'])
	##	plt.legend(['brca','ov'])
	#	plt.legend([cancers_type[0],cancers_type[1]])
	##			plt.legend(['toy1','toy2'])			
	#	plt.title('c-value')
	#	plt.xlabel('mu')
	##	plt.xlabel('lambda')
	##			plt.savefig('lihc_coadread_cvals.pdf')
	##	plt.savefig('brca_ov_cvals.pdf')
	##			plt.savefig('toy_cvals.pdf')			
	#	plt.savefig(ca[0].name+'_'+ca[1].name+'_multi.pdf')
	#	plt.show()
	#	
	#	# plot betadif
	#	betadif = np.zeros(rpoints)
	#	betaav1 = np.zeros((rpoints,ca[0].n_genes))
	#	betaav2 = np.zeros((rpoints,ca[0].n_genes))
	#	for ii in range(rpoints):
	#		# avergae beta for each cancer
	#		betaav1[ii,] = 1/K2*np.sum(betabest[ii][0], axis = 1)
	#		betaav2[ii,] = 1/K2*np.sum(betabest[ii][1], axis = 1)
	#		betadif[ii] = np.linalg.norm(betaav1[ii,:]-betaav2[ii,:])
			
	##	plt.loglog(xpts, betadif, 'o-')
	##	plt.xlabel('mu')
	##	plt.ylabel('||beta1-beta2||')
	##	plt.title('Average distance between beta1 and beta2')
	##	plt.savefig('betadif.pdf')
	##	plt.show()
	#	
	#	pltindices = [0,5,7,9]  # indices to plot
	##	ng = ca[0].n_genes
	#	ng = imp1 + 250
	#	for ii in range(len(pltindices)):
	#		plt.plot(np.arange(ng),betaav1[pltindices[ii],:ng],'o',np.arange(ng),betaav2[pltindices[ii],:ng],'*')
	##		plt.title('Average beta coefficients for mu = '+str(np.around(xpts[pltindices[ii]],2)))
	#		plt.title('Average beta coefficients for mu = '+str(np.around(xpts[pltindices[ii]],2)) + \
	#				', lam = ['+str(np.around(lamdata_rand[pltindices[ii]][0],2))+' ,'+str(np.around(lamdata_rand[pltindices[ii]][1],2))+']')
	#		plt.xlabel('covariates')
	#		plt.ylabel('beta')
	#		plt.legend([cancers_type[0],cancers_type[1]])
	#		plt.savefig('beta'+str(ii)+'.pdf')
	#		plt.show()
	##		
	##	ng = ca[0].n_genes
	###	ng = imp1
	##	for ii in range(len(pltindices)):
	##		plt.plot(np.arange(ng),betaav1[pltindices[ii],:ng],'bo-') #,np.arange(ng),betaav2[pltindices[ii],:ng],'*-')
	##		plt.title('Average beta coefficients for mu = '+str(xpts[pltindices[ii]]))
	##		plt.xlabel('covariates')
	##		plt.ylabel('beta')
	##		plt.legend([cancers_type[0]])#,cancers_type[1]])
	##		plt.savefig('beta'+str(cancers_type[0])+'_'+str(ii)+'.pdf')
	##		plt.show()
	##
	##	ng = ca[0].n_genes
	###	ng = imp1
	##	for ii in range(len(pltindices)):
	##		plt.plot(np.arange(ng),betaav2[pltindices[ii],:ng],'r*-')
	##		plt.title('Average beta coefficients for mu = '+str(xpts[pltindices[ii]]))
	##		plt.xlabel('covariates')
	##		plt.ylabel('beta')
	##		plt.legend([cancers_type[1]])#,cancers_type[1]])
	##		plt.savefig('beta'+str(cancers_type[1])+'_'+str(ii)+'.pdf')
	##		plt.show()
	#		
	#		
	
	#######################################################################################
	##### VALIDATION SET	
		K=1
		rseed_test = rseed_test_data[nn]
		torch.manual_seed(rseed_test)
		beta0new = [0.001*torch.rand(ca_train[cc].n_genes,K).flatten() for cc in range(nc)]	# initialize beta
		betabest_final = []
		betabest_single = []
		if True: #stat == 'multi':
			for cc in range(nc):
				c1 = copy.deepcopy(ca_train[cc])
				indmax = np.argmax(allpar_weight_av_single[cc,:])
				lam_cv_single[cc] = lamall[cc,indmax] # in use2
				retrn = CV1(c1,K,torch.tensor([lam_cv_single[cc]]),beta0new[cc],rseed,gamma0[cc],method,epoch)  # mu=0
				betta = torch.squeeze(retrn.beta_v)
				betabest_single.append(betta)
				beta_single_vec[cc,nn,:] = betta  # save best beta
				print(str(cc+1) + ' / ' + str(nc) + ', single, lambda =' + str(lam_cv_single[cc]) + ', c = ' + str(allpar_weight_av_single[cc,indmax]/nr))
				if (np.max(allpar_weight_av_multi[cc,:])>np.max(allpar_weight_av_single[cc,:])): # multitasking is better than single
					indmax = np.argmax(allpar_weight_av_multi[cc,:])
					lam_cv_multi[cc] = lamdata_rand[indmax]  # in use
					mu_cv_multi[cc] = mudata_rand[indmax]  # in use
					cinddata, betabest, cavm, nonz_len_av_v, nonzgr_v = CV_rand(ca_train,K,[lam_cv_multi[cc]],beta0new,[mu_cv_multi[cc]],rseed,method,lr,epoch)  # mu=0
					betabest_final.append(betabest[0][cc])
					print(str(cc+1) + ' / ' + str(nc) + ', multi, lambda =' + str(lam_cv_multi[cc]) + ', mu =' + str(mu_cv_multi[cc]) + ', c = ' + str(allpar_weight_av_multi[cc,indmax]/nr))
					parsel[cc,nn,:] = np.append(lam_cv_multi[cc], mu_cv_multi[cc].flatten())  # save final selection
					beta_multi_vec[cc,nn,:] = betabest[0][cc]  # multitaskin is better
				else:
					betabest_final.append(betta)
					lammm = np.zeros(nc)
					lammm[cc] = lam_cv_single[cc]
					parsel[cc,nn,:] = np.append(lammm, np.zeros(nc**2))  # save the final selection
					beta_multi_vec[cc,nn,:] = betta  # single is selected over multi
		for cc in range(nc):
				score_val_multi = np.dot(ca_val[cc].x, betabest_final[cc]) # in use 2
				score_val_single = np.dot(ca_val[cc].x, betabest_single[cc]) # in use 2
				cind_val_multi = c_index(score_val_multi,ca_val[cc].os_status,ca_val[cc].os_months)
				cind_val_single = c_index(score_val_single,ca_val[cc].os_status,ca_val[cc].os_months)
				cval_vec_multi[cc,nn] = cind_val_multi
				cval_vec_single[cc,nn] = cind_val_single
	
				if visualisation:
		
					# color groups 
					ng = len(set(ca[0].groups))#int(np.max(ca[0].groups)) #number of groups
#					colrange = [((ng-jj)/ng,(jj+1)/2/ng, 0) for jj in range(ng)]
					cmap = mtplt.cm.get_cmap('Spectral') # viridis
#					rgba = cmap(0.5)
										
					glist_orig, gunq_orig = group_list(ca[0].groups)
					
					plt.plot(betabest_single[cc],'o')
					plt.title('Single '+str(cancers_type[cc])+', lam = '+str(np.around(lam_cv_single[cc].numpy(),2)))
					plt.xlabel('covariate')
					plt.ylabel('beta')
					plt.savefig('single'+str(cancers_type[cc])+'.pdf')
					plt.show()
					
					# boxplot single
					beg = 0   # plot begin pointer
					colors = []
					for jj in range(len(glist_orig)):
						jjj = jj #sortind[jj] # sorted
						if (np.linalg.norm(betabest_single[cc][glist_orig[jjj]]) > 1e-5):
							c = cmap(jj/ng)
							bplot = plt.boxplot(betabest_single[cc][glist_orig[jjj]], positions = [beg], labels = [jjj+1], patch_artist=True,\
									boxprops=dict(facecolor=c, color=c))#,	color = ((ng-jjj)/ng,(jjj+1)/2/ng, 0)) # color grade)
							beg+=1;
					plt.title('Single '+str(cancers_type[cc])+', lam = '+str(np.around(lam_cv_single[cc].numpy(),2)))#, mu = '+str(np.around(mu_cv_single[1][0][1].numpy(),2)))
					plt.xlabel('pathway')
					plt.ylabel('beta')
					plt.savefig('single_'+str(cancers_type[0])+'_'+str(cancers_type[1])+'_'+str(cc)+'_boxplot.pdf')
					plt.show()
					
					maxpeak = np.zeros(ng)  # store ethe highest beta coefficient
					for jj in range(ng):
						maxpeak[jj] = torch.max(torch.abs(betabest_final[cc][glist_orig[jj]]))  # modulus of beta over pathway
						if (np.linalg.norm(betabest_final[cc][glist_orig[jj]]) > 1e-5):
							plt.plot(glist_orig[jj],betabest_final[cc][glist_orig[jj]],'*', color = cmap(jj/ng)) #((ng-jj)/ng,(jj+1)/2/ng, 0)) # color grade)
					plt.title('Multi '+str(cancers_type[cc])+', lam = ['+str(np.around(lam_cv_multi[cc][0],2))+', '\
						 +str(np.around(lam_cv_multi[cc][1],2))+'], mu = '+str(np.around(mu_cv_multi[1][0][1].numpy(),2)))
					plt.xlabel('covariate')
					plt.ylabel('beta')
					plt.xlim([0, ca[0].n_genes])
					plt.savefig('multi_'+str(cancers_type[0])+'_'+str(cancers_type[1])+'_'+str(cc)+'.pdf')
					plt.show()

		# BOXPLOT				
					beg = 0   # plot begin pointer
					colors = []
					for jj in range(len(glist_orig)):
						jjj = jj #sortind[jj] # sorted
						if (np.linalg.norm(betabest_final[cc][glist_orig[jjj]]) > 1e-5):
							c = cmap(jj/ng)
							bplot = plt.boxplot(betabest_final[cc][glist_orig[jjj]], positions = [beg], labels = [jjj+1], patch_artist=True,\
									boxprops=dict(facecolor=c, color=c))#,	color = ((ng-jjj)/ng,(jjj+1)/2/ng, 0)) # color grade)
							beg+=1;
					plt.title('Multi '+str(cancers_type[cc])+', lam = ['+str(np.around(lam_cv_multi[cc][0],2))+', '\
						 +str(np.around(lam_cv_multi[cc][1],2))+'], mu = '+str(np.around(mu_cv_multi[1][0][1].numpy(),2)))
					plt.xlabel('pathway')
					plt.ylabel('beta')
					plt.savefig('multi_'+str(cancers_type[0])+'_'+str(cancers_type[1])+'_'+str(cc)+'_boxplot.pdf')
					plt.show()

					
#					# sort based on max beta
#					sortind = np.argsort(-maxpeak)
#					
#					beg = 0   # plot begin pointer
#					for jj in range(len(glist_orig)):
#						jjj = sortind[jj] # sorted
#						if (np.linalg.norm(betabest_final[cc][glist_orig[jjj]]) > 1e-5):
#							plt.plot(np.arange(beg,beg+len(glist_orig[jjj])),np.sort(betabest_final[cc][glist_orig[jjj]]),'*', \
#							color = ((ng-jjj)/ng,(jjj+1)/2/ng, 0)) # color grade)
#							beg += 500+len(glist_orig[jjj])    # shift beginning with lengh of the pathway
#		#					aas.append(ca[0].pnames[jj][0][0:10])  # first 10 chars
#					plt.title('Multi '+str(cancers_type[cc])+', lam = ['+str(np.around(lam_cv_multi[cc][0],2))+', '\
#						 +str(np.around(lam_cv_multi[cc][1],2))+'], mu = '+str(np.around(mu_cv_multi[1][0][1].numpy(),2)))
#					plt.xlabel('covariate')
#					plt.ylabel('beta')
#		#			nclm = np.floor(len(aas)/10).astype(int)  # how many columns do we need
#		#			plt.xlim([None,ca[0].n_genes+12000*nclm])   # shift the plot so that we have space for labels
#		#			plt.legend(aas, ncol=nclm)
#					plt.savefig('multi_'+str(cancers_type[0])+'_'+str(cancers_type[1])+'_'+str(cc)+'_coefs.pdf')
#					plt.show()
					
	#				# plot legend
	#				for jj in range(ng):
	#					if (np.linalg.norm(betabest_final[cc][glist_orig[jj]]) > 1e-5):
	#						plt.plot(jj,0,'o',color = ((ng-jj)/ng,(jj+1)/2/ng, 0))
	#				nclm = np.floor(len(aas)/10).astype(int)  # how many columns do we need				
	#				plt.title('Legend')
	#				plt.legend(aas, ncol=nclm)
	#				plt.ylim([-0.005, None])   # shift the plot so that we have space for labels
	#				plt.savefig('legend'+ca[cc].name+'.pdf')
	#				plt.show()
	#			
#	congcoef = cong_av(parsel)  # average congruence coefficient
#	corrcoef = corr_av(parsel)
#	congcoef2 = cong_av(beta_single_vec)  # average congruence coefficient
#	congcoef3 = cong_av(beta_multi_vec)
#	print('cong = '+str(congcoef)+' ,corr = '+str(corrcoef))
#	#
	aas = ''
	for cc in range(nc):
	#	aas = aas + ca[cc].name;
		aas = aas + cancers_type[cc];
		aas = aas + '_';
	aas = aas + 'reg'
	aas = aas + str(reg)
	print(aas)
	
	np.save('cval_multi_'+aas,cval_vec_multi)
	np.save('cval_single_'+aas,cval_vec_single)
	np.save('beta_multi_'+aas,beta_multi_vec)
	np.save('beta_single_'+aas,beta_single_vec)
#	np.save('cong_'+aas,cong_av)
#	np.save('corr_'+aas,corr_av)	
			
	toc()
	#
	#
	##
	##def main(args):
	##	fsdfsd
	##	
	##if __name__ == '__main__':
	##	main(sys.argv)