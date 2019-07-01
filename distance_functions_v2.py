#! /usr/bin/env python

## This code is written by Davide Albanese <albanese@fbk.eu> and 
## Giuseppe Jurman <jurman.fbk.eu>.
## (C) 2010 Fondazione Bruno Kessler - Via Santa Croce 77, 38100 Trento, ITALY.

## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.

## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.

## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import scipy as sp
from scipy.integrate import quad
import scipy.optimize
import itertools as it

LIMIT = 1000

def max_ipsen_minus_one(g,n):
        return(np.sqrt( 1/(np.pi*g) + 1/(2*g*(np.arctan(np.sqrt(n)/g)+np.pi/2)**2)*(np.pi/2+ (np.sqrt(n)/g)/(1+(np.sqrt(n)/g)**2)+np.arctan(np.sqrt(n)/g)) -4*(np.pi-(g/np.sqrt(n))*np.log(1/(1+(np.sqrt(n)/g)**2))+np.arctan(np.sqrt(n)/g))/ (np.pi*g*(4+(np.sqrt(n)/g)**2)*(np.arctan(np.sqrt(n)/g)+np.pi/2))) -1)

def optimal_gamma(n):
    return(scipy.optimize.fsolve(max_ipsen_minus_one, 0.4, args=(n))[0])

def mcc(G,H):
  tp = 0.0
  tn = 0.0
  fp = 0.0
  fn = 0.0
  n = G.shape[0]

  for i in range(n-1):
      for j in range((i+1),n):
               if((G[i,j]==1) and (H[i,j]==1)): 
                   tp = tp+1
               if((G[i,j]==1) and (H[i,j]==0)): 
                   fn = fn+1
               if((G[i,j]==0) and (H[i,j]==0)): 
                   tn = tn+1
               if((G[i,j]==0) and (H[i,j]==1)): 
                   fp = fp+1
  
  num = tp*tn-fp*fn
  den = np.sqrt((tp+fn)*(tp+fp)*(tn+fn)*(tn+fp))
  if(den==0.0):
      mcc = 0.0
  else:
      mcc = 1.0*num/den
   
  return mcc 

def hamming(G,H):
    return 1.0*np.sum(abs(G-H))/(1.0*G.shape[0]*(G.shape[0]-1))

def rho_unnormalized(omega, omega_k, gamma):
    tmp = gamma  / ((omega - omega_k)**2 + gamma**2)
    return np.sum(tmp)


def rho_unnormalized_int(omega_k, gamma):
    return quad(rho_unnormalized, 0, np.Inf, args=(omega_k, gamma), limit=LIMIT)[0]


def density_diff(omega, omega_k_one, omega_k_two, K_one, K_two, gamma):

    rho_one = K_one * rho_unnormalized(omega, omega_k_one, gamma)
    rho_two = K_two * rho_unnormalized(omega, omega_k_two, gamma)

    return (rho_one - rho_two)**2


def density_diff_int(omega_k_one, omega_k_two, K_one, K_two, gamma):
    
    return quad(density_diff, 0, np.Inf, args=(omega_k_one, omega_k_two, 
               K_one, K_two, gamma), limit=LIMIT)[0]

def laplacian(adj_mat):
 
    return degree(adj_mat)  - adj_mat

def degree(adj_mat):

    dim = adj_mat.shape[0]
    deg_mat = np.zeros((dim,dim))
    for i in range(dim):
        deg_mat[i,i] = sum(adj_mat[i,])

    return deg_mat
   

def epsilon(a_one,a_two,gamma): 

    # laplacian    
    l_one = laplacian(a_one)   
    l_two = laplacian(a_two)   
 
    # eigenvalues
    e_one = np.real_if_close(np.linalg.eigvals(l_one))
    e_two = np.real_if_close(np.linalg.eigvals(l_two))

    # sorting
    e_one.sort()
    e_two.sort()
   
    # excluding smallest value (zero)
    ee_one = e_one[1:]
    ee_two = e_two[1:]
 
    # omega 
    omega_one = np.sqrt([round(l,10) for l in ee_one])
    omega_two= np.sqrt([round(l,10) for l in ee_two])

    # K
    K_one = 1 / rho_unnormalized_int(omega_k=omega_one, gamma=gamma)
    K_two = 1 / rho_unnormalized_int(omega_k=omega_two, gamma=gamma)
    
    # epsilon
    return np.sqrt(density_diff_int(omega_one, omega_two, K_one, K_two, gamma))

def epsilon_with_eigens(a_one,a_two,gamma,e_one,e_two): 

    # sorting
    e_one.sort()
    e_two.sort()
   
    # excluding smallest value (zero)
    ee_one = e_one[1:]
    ee_two = e_two[1:]
 
    # omega 
    omega_one = np.sqrt([round(l,10) for l in ee_one])
    omega_two= np.sqrt([round(l,10) for l in ee_two])

    # K
    K_one = 1 / rho_unnormalized_int(omega_k=omega_one, gamma=gamma)
    K_two = 1 / rho_unnormalized_int(omega_k=omega_two, gamma=gamma)
    
    # epsilon
    return np.sqrt(density_diff_int(omega_one, omega_two, K_one, K_two, gamma))


def E(n):
    return np.zeros((n,n))

def F(n):    
    return np.ones((n,n))-np.diag(np.ones(n))
    
def normalization(n, gamma):
    return epsilon(E(n),F(n),gamma)

def epsilon_hat(a_one,a_two,gamma):
    return epsilon(a_one,a_two,gamma)/normalization(a_one.shape[0],gamma)

    
def thresholder(matrix,threshold):
    binary = matrix.copy()
    binary[binary>=threshold]=1
    binary[binary<threshold]=0
    return(binary)
    
def glocal(G,H):
    ed=hamming(G,H)
    ip=epsilon(G,H,optimal_gamma(G.shape[0]))
    gl=np.sqrt(0.5*(ed**2+ip**2))
    return(ed,ip,gl)    

def glocal_with_eigens(G,H,EG,EH):
    ed=hamming(G,H)
    ip=epsilon_with_eigens(G,H,optimal_gamma(G.shape[0]),EG,EH)
    gl=np.sqrt(0.5*(ed**2+ip**2))
    return(ed,ip,gl)    

