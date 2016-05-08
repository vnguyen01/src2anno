'''
DTM Models - Mar 1, 2016
Andre Nguyen
'''

from __future__ import division
import numpy as np
import scipy as sp
from scipy.stats import multivariate_normal
from collections import namedtuple
import copy, time, datetime, os, re, operator
from urllib2 import urlopen
import time
import cPickle as pickle
import dateutil

from pgmult.lda import StandardLDA, StickbreakingDynamicTopicsLDA
from pybasicbayes.util.text import progprint, progprint_xrange
from pgmult.internals.utils import mkdir, compute_uniform_mean_psi, kappa_vec
from pylds.lds_messages_interface import filter_and_sample
from pgmult.utils import *
from ctm import get_sparse_repr

from pgmult.lda import *


# Results object.
Results = namedtuple('Results', ['loglikes', 'predictive_lls', 'samples', 'timestamps'])


'''
Functions for nummerical errors.
'''
def huge_to_big(array):
    '''
    Make near inf numbers a bit smaller.
    '''
    array[array > 1e300] = 1e300
    array[array < -1e300] = -1e300
    return array

# Modified to get no -inf.
def log_likelihood(data, wordprobs):
    return np.sum(huge_to_big(np.nan_to_num(np.log(wordprobs))) * data.data) \
        + gammaln(data.sum(1)+1).sum() - gammaln(data.data+1).sum()

'''
Run Standard DTM.
'''

def fit_sbdtm_gibbs(train_data, test_data, timestamps, K, Niter, alpha_theta, lda_model=None):
    def evaluate(model):
        ll, pll = \
            model.log_likelihood(), \
            model.log_likelihood(test_data)
        # print '{} '.format(ll),
        return ll, pll

    def sample(model):
        tic = time.time()
        model.resample()
        timestep = time.time() - tic
        return evaluate(model), timestep

    print 'Running sbdtm gibbs...'
    
    model = StickbreakingDynamicTopicsLDA(train_data, timestamps, K, alpha_theta, lda_model=lda_model)
        
    init_val = evaluate(model)
    
    vals, timesteps = zip(*[sample(model) for _ in progprint_xrange(Niter)])

    lls, plls = zip(*((init_val,) + vals))
    times = np.cumsum((0,) + timesteps)

    return Results(lls, plls, model.copy_sample(), times)


'''
Run DTM with H learning.
'''

def fit_obsdtm_gibbs(train_data, test_data, timestamps, K, Niter, alpha_theta, U, lda_model=None):
    def evaluate(model):
        ll, pll = \
            model.log_likelihood(), \
            model.log_likelihood(test_data)
        # print '{} '.format(ll),
        return ll, pll

    def sample(model):
        tic = time.time()
        model.resample()
        timestep = time.time() - tic
        return evaluate(model), timestep

    print 'Running obsdtm gibbs...'

    # TODO: assert U and H_mat shape consistent
    
    model = ObsDTM(train_data, timestamps, K, alpha_theta, H_mat='learn', U=U, lda_model=lda_model)
        
    init_val = evaluate(model)
    
    vals, timesteps = zip(*[sample(model) for _ in progprint_xrange(Niter)])

    lls, plls = zip(*((init_val,) + vals))
    times = np.cumsum((0,) + timesteps)

    return Results(lls, plls, model.copy_sample(), times)


'''
DTM with H learning DTM
'''

class ObsDTM(StickbreakingDynamicTopicsLDA):
    '''
    Extension of the Stickbreaking Dynamic Topic Model. Includes a learned observation model.
    Note: psi is (docs,V-1,topics) originally.
    '''
    
    def __init__(self, data, timestamps, K, alpha_theta, H_mat='learn', U=None, lda_model=None,
                sigmasq_states=0.1, **xargs):
        
        self.H_mat = H_mat
        self.u = None
        self.U_size = int(U)
        
        super(ObsDTM, self).__init__(data, timestamps, K, alpha_theta, 
                                       lda_model=lda_model,
                                       sigmasq_states=sigmasq_states)

        self.h_size = [self.V - 1, int(U)]


    def initialize_parameters(self, lda_model=None):
        """
        Initialize the model with either a draw from the prior
        or the parameters of a given LDA model
        """

        # Allocate auxiliary variables
        self.omega = np.zeros((self.T, self.V-1, self.K))

        # If LDA model is given, use it to initialize beta and theta
        if lda_model:
            assert lda_model.D == self.D
            assert lda_model.V == self.V
            assert lda_model.T == self.K

            self.beta = lda_model.beta
            self.theta = lda_model.theta

        else:
            # Initialize beta to uniform and theta from prior
            mean_psi = compute_uniform_mean_psi(self.V)[0][None,:,None]
            self.psi = np.tile(mean_psi, (self.T, 1, self.K))


            self.theta = sample_dirichlet(
                self.alpha_theta * np.ones((self.D, self.K)), 'horiz')

            self.H_mat = [np.random.randn(self.V - 1, self.U_size) for k in range(self.K)]
            for k in range(self.K):
                self.H_mat[k][:,0] = self.psi[0,:,k]
            self.u = np.zeros((self.T,self.U_size,self.K))  
            for t in range(self.T):
                for k in range(self.K):
                    self.u[:,0,:] = 1.0


        # Sample topic-word assignments
        self.z = np.zeros((self.data.data.shape[0], self.K), dtype='uint32')
        self.resample_z()


    # Modified to get no -inf.
    def log_likelihood(self, data=None, theta=None):
        if data is not None:
            return log_likelihood(
                data, self._get_wordprobs(
                    data, self._get_timeidx(self.timestamps, data), theta))
        else:
            # this version avoids recomputing the training gammalns
            wordprobs = self._get_wordprobs(self.data, self.timeidx, theta)
            return np.sum(huge_to_big(np.nan_to_num(np.log(wordprobs))) * self.data.data) \
                + self._training_gammalns

        
    def _get_lds_effective_params(self):
        
        mu_uniform, sigma_uniform = compute_uniform_mean_psi(self.h_size[1] + 1)
        mu_init = np.tile(mu_uniform, self.K)
        sigma_init = np.tile(np.diag(sigma_uniform), self.K)

        # For reduced H.
        sigma_states = np.repeat(self.sigmasq_states, self.h_size[1] * self.K)

        sigma_obs = 1./self.omega
        y = kappa_vec(self.time_word_topic_counts, axis=1) / self.omega

        return mu_init, sigma_init, sigma_states, \
            sigma_obs.reshape(y.shape[0], -1), y.reshape(y.shape[0], -1)


    def resample(self):
        self.resample_z()
        print 'z'
        print self.log_likelihood()
        self.resample_theta()
        print 'theta'
        print self.log_likelihood()
        self.resample_beta()
        print 'beta'
        print self.log_likelihood()
        self.resample_lds_params()
        print 'lds'
        print self.log_likelihood()

        # resamples psi (not u)
        self.resample_psi_proper()
        print 'psi proper'
        print self.log_likelihood()

    # psi = Hu + epsilon
    # no change in H or u resampling
    # instead resample psi | H, u, etc. polya-gamma parameters in a gaussian update
    # resamples topic by topic
    # TODO: make epsilon a global parameter to set and experiment with
    # TODO: can make more efficient, left it this way so it's more clear
    def resample_psi_proper(self):

        ep = np.diag([0.01 for _ in xrange(self.V - 1)])
        inv_ep = np.linalg.inv(ep) 

        # iterate over topics
        for j in xrange(self.K):
            
            k_x = kappa_vec(self.time_word_topic_counts, axis=1) 

            # iterate over each timestep
            for t in xrange(k_x.shape[0]):

                # calculate mean and variance

                Omega = np.diag(self.omega[t, :, j])   
                cv = np.linalg.inv(Omega + inv_ep)

                m1 = np.dot(cv, k_x[t, :, j])
                m2 = np.dot(cv, np.dot(inv_ep, np.dot(self.H_mat[j], self.u[t, :, j].T)))
                m = m1 + m2
                m = list(m)

                # update
                self.psi[t, :, j] = np.random.multivariate_normal(m, cv)

        return
        

    def resample_psi(self,reorder=False):
        
        # Get internal variables.
        mu_init_temp, sigma_init_temp, sigma_states_temp, sigma_obs_temp, y_temp = \
            self._get_lds_effective_params()
            
        # Us
        if self.u == None:
            self.u = np.zeros((y_temp.shape[0],self.h_size[1],self.K))
            
        #print mu_init_temp.shape, sigma_init_temp.shape, sigma_states_temp.shape, sigma_obs_temp.shape, y_temp.shape
        
        # Loop through topics.
        for top in range(self.K):

            #print 'ZERO'
        
            # Prep for filtering and sampling function.
            mu_init = mu_init_temp[(top*self.h_size[1]):((top+1)*self.h_size[1])]
            
            sigma_init = np.diag(sigma_init_temp[(top*self.h_size[1]):((top+1)*self.h_size[1])])
            
            A = np.stack([np.eye(self.h_size[1]) for _ in range(y_temp.shape[0])], axis=0) 
            
            if self.H_mat == None:
                H = np.stack([np.eye(self.V - 1) for _ in range(y.shape[0])], axis=0)
            elif self.H_mat == 'learn':
                #H = np.stack([np.eye((self.V - 1) * self.K) for _ in range(y_temp.shape[0])], axis=0)
                H = np.stack([np.random.randn(self.h_size[0], self.h_size[1]) for _ in range(y_temp.shape[0])], axis=0)
            else:
                H = np.stack([self.H_mat[top] for _ in range(y_temp.shape[0])], axis=0)

            #print 'ONE'
            sigma_states = np.stack([np.diag(sigma_states_temp[(top*self.h_size[1]):((top+1)*self.h_size[1])]) for _ in range(y_temp.shape[0])], axis=0)
            #print 'TWO'
            #print self.H_mat[0].shape
            #print (sigma_obs_temp.shape[0],self.h_size[0],self.h_size[0])
            sigma_obs = np.zeros((sigma_obs_temp.shape[0],self.h_size[0],self.h_size[0]), dtype='float32')
            #print 'TWO - 1'
            for t in xrange(sigma_obs_temp.shape[0]):
                #print 'TWO - 2'
                sigma_obs[t] = np.diag(sigma_obs_temp[t][(top*self.h_size[0]):((top+1)*self.h_size[0])])
                #print 'TWO - 3'
   
            #print 'THREE'
            y = y_temp[:,(top*self.h_size[0]):((top+1)*self.h_size[0])] 

            # Filter and sample u.
            #print 'HELLO'
            # print(mu_init.shape, sigma_init.shape, A.shape, sigma_states.shape, H.shape, sigma_obs.shape, y.shape)
            ll, u_flat = filter_and_sample(mu_init, sigma_init, A, sigma_states, H, sigma_obs, y)
            #print 'BYE'
            #print u_flat.shape
            #print self.u[:,:,top].shape, 'line 2'
            
            self.u[:,:,top] = u_flat.reshape((self.psi[:,:,top].shape[0],-1))  
            #print 'FOUR'
            # Compute the psi from u.
            # psi_flat = [] 
            # for t in xrange(H.shape[0]):
            #     psi_flat.append(np.dot(H[t],u_flat[t]))
            # psi_flat = np.asarray(psi_flat)
            # #print 'FIVE'
            # self.psi[:,:,top] = psi_flat.reshape(self.psi[:,:,top].shape)



    def resample_lds_params(self):
        
        def learn_H(psi_reg, u_reg):
            
            # Error metric.
            def mse(pred,actual):
                return ((pred-actual)**2).mean()
            
            '''
            # Format data.
            train_psi_set = np.copy(psi_reg.reshape((psi_reg.shape[0],psi_reg.shape[1]*psi_reg.shape[2])))
            train_u_set = np.copy(u_reg.reshape((train_psi_set.shape[0],self.h_size[1]*psi_reg.shape[2])))
            
            # Regressor. # rg = BayesRegressor(w0=np.zeros(self.V-1), V0=np.eye(self.V-1)*1.0, sigma=1.0)
            rg = BayesRegressor(w0=np.zeros(self.h_size[1]), V0=np.eye(self.h_size[1])*1.0, sigma=1.0)
            
            # Combine data to learn a single H for all topics. 
            train_u_set = np.concatenate([train_u_set[:,(i*self.h_size[1]):((i+1)*self.h_size[1])] for i in xrange(0,self.K)],axis=0)
            train_psi_set = np.concatenate([train_psi_set[:,(i*self.h_size[0]):((i+1)*self.h_size[0])] for i in xrange(0,self.K)],axis=0)
            
            # For each row of H.
            final_weights = np.zeros((self.h_size[0],self.h_size[1]))
            for i in xrange(train_psi_set.shape[1]):
                rg.fit(train_u_set,train_psi_set[:,i])
                # Sample the weights.
                final_weights[i] = rg.sample_w_posterior()
            # Build full H. 
            H_new = np.zeros((self.h_size[0]*self.K,self.h_size[1]*self.K))
            for i in xrange(0,self.K):
                H_new[(i*self.h_size[0]):((i+1)*self.h_size[0]),(i*self.h_size[1]):((i+1)*self.h_size[1])] = final_weights
            '''
            
            '''
            START: Learn a different H for each topic.
            '''
            # Format data.
            train_psi_set_orig = np.copy(psi_reg.reshape((psi_reg.shape[0],psi_reg.shape[1]*psi_reg.shape[2])))
            train_u_set_orig = np.copy(u_reg.reshape((train_psi_set_orig.shape[0],self.h_size[1]*psi_reg.shape[2])))
            
            H_new = []
            
#             for i in xrange(0,self.K):
                
#                 train_u_set = train_u_set_orig[:,(i*self.h_size[1]):((i+1)*self.h_size[1])]
#                 train_psi_set = train_psi_set_orig[:,(i*self.h_size[0]):((i+1)*self.h_size[0])]

#                 # For each row of H.
#                 final_weights = np.zeros((self.h_size[0],self.h_size[1]))
#                 for ii in xrange(train_psi_set.shape[1]):
#                     rg = BayesRegressor(w0=np.zeros(self.h_size[1]), V0=np.eye(self.h_size[1])*1.0, sigma=1.0)
#                     rg.fit(train_u_set,train_psi_set[:,ii])
#                     # Sample the weights.
#                     final_weights[ii] = rg.sample_w_posterior()

#                 # Build full H. 
#                 #print final_weights
#                 #H_new[(i*self.h_size[0]):((i+1)*self.h_size[0]),(i*self.h_size[1]):((i+1)*self.h_size[1])] = np.copy(final_weights)
#                 H_new.append(np.copy(final_weights))

            
            train_u_set = np.concatenate([train_u_set_orig[:,(i*self.h_size[1]):((i+1)*self.h_size[1])] for i in xrange(0,self.K)],axis=0)
            train_psi_set = np.concatenate([train_psi_set_orig[:,(i*self.h_size[0]):((i+1)*self.h_size[0])] for i in xrange(0,self.K)],axis=0) 
            
            # For each row of H.
            final_weights = np.zeros((self.h_size[0],self.h_size[1]))
            for ii in xrange(train_psi_set.shape[1]):
                rg = BayesRegressor(w0=np.zeros(self.h_size[1]), V0=np.eye(self.h_size[1])*1.0, sigma=1.0)
                rg.fit(train_u_set,train_psi_set[:,ii])
                # Sample the weights.
                final_weights[ii] = rg.sample_w_posterior()

            # Build full H. 
            #print final_weights
            #H_new[(i*self.h_size[0]):((i+1)*self.h_size[0]),(i*self.h_size[1]):((i+1)*self.h_size[1])] = np.copy(final_weights)
            for i in xrange(0,self.K):
                H_new.append(np.copy(final_weights))
                
            '''
            END
            '''

            # Return a learned H.
            return H_new

        ####
        # Main section of resample_lds_params.
        ####

        # Learn H.
        self.H_mat = learn_H(self.psi, self.u) 
        
        return


'''
Bayesian Regression
'''

class BayesRegressor(object):
    '''Bayesian Linear Regressor.'''
    
    def __init__(self, w0, V0, sigma):
        self.w0 = w0  # prior weights mean
        self.V0 = V0  # prior weights cov
        self.sigma = sigma  # sigma^2 is noise variance
        
    def fit(self,X,y):
        self.X = X
        self.y = y    
        VN_inv = np.linalg.inv(self.V0) + np.dot(X.T,X)/(self.sigma**2)
        self.VN = np.linalg.inv(VN_inv)
        self.wN = np.dot(np.dot(self.VN, np.linalg.inv(self.V0)),self.w0) + \
            np.dot(np.dot(self.VN,X.T),y)/(self.sigma**2)
        
    def update_prior(self):
        self.V0 = self.VN
        self.w0 = self.wN
        
    def sample_w_posterior(self):
        return np.random.multivariate_normal(self.wN.reshape((len(self.wN),)), self.VN)
    
    def sample_w_prior(self):
        return np.random.multivariate_normal(self.w0.reshape((len(self.w0),)), self.V0)
    
    def central_credible_interval(self,x,a):
        sigma_sq_N = self.sigma**2 + np.dot(x.T,np.dot(self.VN,x))
        cci_mean = np.dot(self.wN.T,x)
        lower = norm.ppf(a/2, loc=cci_mean, scale=np.sqrt(sigma_sq_N))
        upper = norm.ppf(1 - a/2, loc=cci_mean, scale=np.sqrt(sigma_sq_N))
        return [lower, upper]
    
    def marginal_likelihood(self):
        ml_mean = np.dot(self.X,self.wN)
        ml_cov = (self.sigma**2) * np.eye(self.X.shape[0]) + \
            np.dot(self.X,np.dot(self.VN,self.X.T))
        return multivariate_normal.pdf(self.y, mean=ml_mean, cov=ml_cov)


'''
SOTU data
'''

def fetch_sotu():
    baseurl = 'http://stateoftheunion.onetwothree.net/texts/'
    path = 'data/sotu/sotus.pkl'

    def download_text(datestr):
        pagetext = urlopen(baseurl + datestr + '.html').read().replace('\n', ' ')
        paragraphs = re.findall(r'<p>(.*?)</p>', pagetext, re.DOTALL)
        return ' '.join(paragraph.strip() for paragraph in paragraphs)

    if not os.path.isfile(path):
        response = urlopen(baseurl + 'index.html')
        dates = re.findall(r'<li><a href="([0-9]+)\.html">', response.read())

        print 'Downloading SOTU data...'
        sotus = {date:download_text(date) for date in progprint(dates)}
        print '...done!'

        mkdir(os.path.dirname(path))
        with open(path, 'w') as outfile:
            pickle.dump(sotus, outfile, protocol=-1)
    else:
        with open(path, 'r') as infile:
            sotus = pickle.load(infile)

    return sotus

def load_sotu_data(V, sort_data=True):
    sotus = fetch_sotu()
    datestrs, texts = zip(*sorted(sotus.items(), key=operator.itemgetter(0)))
    return datestrs_to_timestamps(datestrs), get_sparse_repr(texts, V, sort_data)

def datestrs_to_timestamps(datestrs):
    return [dateutil.parser.parse(datestr).year for datestr in datestrs]