import torch
import numpy as np
import wincnn.wincnn as wincnn
from sympy import Rational

# Helper in generating polynomial interpolation points.
def helpInterpolate(n,k):
    if k==0:
        return n
    elif k==1:
        return -n
    elif k==2:
        return Rational(1,n)
    else:
        return -Rational(1,n)

# Get N polynomial interpolation points.
def getInterpolate(N):
    a=[]
    for i in range(0,N):
        if i==0:
            a.append(0)
        elif i==1:
            a.append(1)
        elif i==2:
            a.append(-1)
        else:
            n=int((i+5)/4)
            a.append(helpInterpolate(n,(i-3)%4))
    return tuple(a)

# Winograd Accelerated Convolution.
# - M_w(input): rxr kernel.
# - M_in(input): (m+r−1)x(m+r−1) inputs.
# - M_out(output): mxm outputs.
def Winograd(M_in,M_w):
    r=list(M_w.shape)[0]
    m=list(M_in.shape)[0]-r+1

    N=m+r-2
    a=getInterpolate(10)
    M_A,M_G,M_D=wincnn.getCookToomConvolution(a,m,r)

    M_A=np.array(M_A).astype(np.float64)
    M_G=np.array(M_G).astype(np.float64)
    M_D=np.array(M_D).astype(np.float64)
    
    tmp_wG=M_G.dot(M_w).dot(M_G.T)
    tmp_inD=M_D.T.dot(M_in).dot(M_D)
    M_out=M_A.T.dot(tmp_wG*tmp_inD).dot(M_A)

    return M_out


# Example: 
M_w=torch.tensor([[1,2],[5,6]])
M_in=torch.tensor([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,25]])
print(Winograd(M_in,M_w))

# Result: 
# [[ 77.  91. 105. 119.]
#  [147. 161. 175. 189.]
#  [217. 231. 245. 259.]
#  [287. 301. 315. 329.]]