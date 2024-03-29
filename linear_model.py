from __future__ import division
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from numpy.linalg import multi_dot
import math



def runScalenet(trainD, testD, yTrain, yTest, k, eta1, eta2, MaxIter=20, subMaxIter=6):
    error = 1e-3  

    
    scaler = StandardScaler()
    trainD = scaler.fit_transform(trainD)
    testD = scaler.transform(testD)
    
    trainD = np.column_stack((trainD,np.ones((trainD.shape[0])))).T  #add a feature for bias term 
    testD = np.column_stack((testD,np.ones((testD.shape[0])))).T
    
    MTr = np.isnan(trainD)
    MTr = np.concatenate([~MTr,MTr], axis=0).astype(float)
    MTe = np.isnan(testD)
    MTe = np.concatenate([~MTe,MTe], axis=0).astype(float)
    trainD = np.nan_to_num(trainD)
    testD = np.nan_to_num(testD) 
    dTrain = trainD.shape[0]
    nTrain = trainD.shape[1]
    PTr = MTr[:dTrain,:]


    
    
    '''
    initialize U and V
        
    '''
    U = np.zeros((k,dTrain))
    V = np.random.random((k,2*dTrain))
    

    
    
    I = np.identity(2*dTrain)

    
  
    sv = getSupportV(yTrain, MTr, U, V, trainD)     ###   get support vectors
    
    obj0 = getobj(trainD, U, V, MTr, I, eta1, eta2, yTrain, sv)   ### get obj values
    
    
    for maxIter in range(MaxIter):
        
        C = getC(V, eta1)
        L_phi = getL_phi(V, MTr, eta1, I, nTrain, PTr)
        L_h = getL_h(V, eta2, nTrain, MTr, trainD)        
        gamma = max((8 * L_phi)**0.5, 8 * L_h)        
        eps = obj0
        
            
        '''
        Calculate step and T(U), if T(U) > 50, we set 50 to the iteration number
        
        '''    
        step = eps / (gamma*(eta2**0.5)) 

        if eta2<1:
            T_u = max (((8 * L_phi)**0.5 * C * gamma)/ (3 * eps ), ((8 * L_h) * C * gamma)/ eps)
        else:
            T_u = max (((8 * L_phi)**0.5 * eta2 * C * gamma)/ (3 * eps ), ((8 * L_h) * eta2**0.5 * C * gamma)/ eps)
        
        
        '''
        update U
        
        '''       
        
        obj_best = obj0
        best_U = U
        best_V = V
        best_sv = sv        
        
        for stage1 in range(subMaxIter):
            
            '''
            find best U
            '''   
            
            T_u = min(T_u, 50)      
               
            for t1 in range(int(math.ceil(T_u))):
                
                gU = multi_dot([2*V, eta1 * I, np.dot(V.T, U)]) - (eta2 / nTrain) * multi_dot([V, MTr.T[sv].T, np.diag(yTrain[sv]), trainD.T[sv]]) + (2/ nTrain) * multi_dot([V, MTr, np.multiply(PTr.T, multi_dot([MTr.T,V.T,U]))])
                U = U - step * gU / (np.linalg.norm(gU, ord='fro') )
                               
                sv = getSupportV(yTrain, MTr, U, V, trainD)                
                obj_new = getobj(trainD, U, V, MTr, I, eta1, eta2, yTrain, sv)
                
                if obj_new <= obj_best:
                    
                    obj_best = getobj(trainD, U, V, MTr, I, eta1, eta2, yTrain, sv)
                    best_U = U
                    best_sv = sv
            
            step = step/2.
            T_u = T_u * 2
            U = best_U
            sv = best_sv
            
        '''
        update V
        '''            
        eps = obj_best
        C = getC(U, eta1)
        L_phi = getL_phi(U, MTr, eta1, I, nTrain, PTr)
        L_h = getL_h(U, eta2, nTrain, MTr, trainD)
        
        gamma = max((8 * L_phi)**0.5, 8 * L_h)
        
        step = eps / (gamma*(eta2**0.5)) 

        if eta2<1:
            T_v = max (((8 * L_phi)**0.5 * C * gamma)/ (3 * eps ), ((8 * L_h) * C * gamma)/ eps)
        else:
            T_v = max (((8 * L_phi)**0.5 * eta2 * C * gamma)/ (3 * eps ), ((8 * L_h) * eta2**0.5 * C * gamma)/ eps)        

                
        for stage2 in range(subMaxIter):
            
            '''
            find best V
            '''
            T_v = min(T_v, 50)
            
            for t2 in range(int(math.ceil(T_v))):
                
                gV = multi_dot([multi_dot([2*U, U.T, V]), eta1 * I]) - (eta2 / nTrain) * multi_dot([U, trainD.T[sv].T, np.diag(yTrain[sv]), MTr.T[sv]]) + (2/ nTrain) * multi_dot([U, np.multiply(PTr, multi_dot([U.T,V,MTr])), MTr.T])
                V = V - step * gV / (np.linalg.norm(gV, ord='fro'))
                
                sv = getSupportV(yTrain, MTr, U, V, trainD)
                
                obj_new = getobj(trainD, U, V, MTr, I, eta1, eta2, yTrain, sv)
                
                if obj_new <= obj_best:
                    obj_best = getobj(trainD, U, V, MTr, I, eta1, eta2, yTrain, sv)
                    best_V = V
                    best_sv = sv  
            
            step = step/2.
            T_v = T_v * 2
            V = best_V
            sv = best_sv
                
        obj = getobj(trainD, U, V, MTr, I, eta1, eta2, yTrain, sv)
                    
        if abs(obj0- obj) / (obj0) < error:
            break
        
        obj0 = obj
    

    '''
    test
    '''
    
    predict = np.sum(np.multiply(multi_dot([U.T, V, MTe]), testD), axis = 0)
    predict[predict < 0] = -1
    predict[predict >= 0] = 1       
        
    accuray = accuracy_score(yTest, predict)    
        
    return accuray      


def getobj(trainD, U, V, MTr, I, eta1, eta2, y0, sv):
    
    dp = trainD.shape[0]
    PTr = MTr[:dp,:]
    n = trainD.shape[1]
    I1 = np.identity(np.count_nonzero(sv))  
    L = np.linalg.norm(np.multiply(PTr, multi_dot([U.T,V,MTr])))**2/n + eta1 * np.linalg.norm(np.dot(U.T,V)) + eta2/n * np.trace(I1 - multi_dot([np.diag(y0[sv]), trainD.T[sv], U.T, V, MTr.T[sv].T]))
    
    return L
  
def getSupportV(y0, MTr, U, V, trainD):    

    v1 = np.sum(np.multiply(multi_dot([U.T, V, MTr]), trainD), axis = 0)    
    v2 = 1 - np.multiply(y0, v1)    
    sv = (v2 > 0)
       
    return sv

def getC(X, eta1_c):

    s = np.linalg.svd(X,full_matrices=1,compute_uv=0)    
    sigma = s[s>0][-1]
    C = 1/ (sigma**2 * eta1_c)
    
    return C

def getL_phi (X, MTr, eta1, I, nTrain, PTr):
    
    if X.shape[1] == MTr.shape[0]:
        L = []
        for i in range(PTr.shape[0]):
            
            L.append(np.linalg.norm(multi_dot([X, multi_dot([MTr, np.diag(PTr.T[:,i]), MTr.T])/nTrain + eta1 * I, X.T]), ord='fro'))
        
        L_phi = 2*np.max(L)
    else:
        L = []
        for i in range(PTr.shape[1]):
            
            X_U = np.multiply(PTr[:,i], X)
            L.append(np.linalg.norm(np.dot(X_U, X_U.T) ,ord='fro'))
            
        L_phi = MTr.shape[0]*np.sum(L)/nTrain + 2*eta1 * np.linalg.norm(np.dot(X, X.T), ord='fro')
      
    return L_phi

def getL_h (X, eat2, nTrain, MTr, trainD):
    
    if X.shape[1] == MTr.shape[0]:

        L= sum([np.linalg.norm(np.outer(np.dot(X, MTr[:, i]), trainD[:,i]) , ord = 'fro') for i in range(nTrain)])        
    else:
        
        L= sum([np.linalg.norm(np.outer(np.dot(X, trainD[:, i]), MTr[:,i].T) , ord = 'fro') for i in range(nTrain)])
    
    L_h = eta2*L/nTrain
    
    return L_h



if __name__=="__main__":

    for d_fdir in ['bands','hepatitis','horse','mammographics','pima']:
        
        np.random.seed(2)
        
        fdir = './linear_data/'+d_fdir +'/'
        
        #==============load data============================        
        #load original data
        data0 = np.genfromtxt(fdir + 'data.txt', dtype='f8', delimiter=',')
        y = np.load(fdir+'labels.npy')
        #load data with additional missing entries
        data = np.load(fdir + 'data.npy')
        
        test_index = np.load(fdir + 'test_id.npy')
        train_index = np.setdiff1d(range(len(y)), test_index)
    
        trainD = data[train_index]
        testD = data0[test_index]
        
        yTrain = y[train_index]
        yTest = y[test_index]
        #===================================================
        
        
        #==============parameters===========================
        para = np.load(fdir + 'para.npy')
        k = int(para[0])
        eta1 = 10**-6
        eta2 = para[1]
        #===================================================
        
        acc=[]        
        for i in range(5):
            missing_mask = np.random.choice(a = [False, True], size = testD.shape, p = [0.7,0.3])
            test_D = np.copy(testD)
            test_D[missing_mask]=np.nan
            acc.append(runScalenet(trainD, test_D, yTrain, yTest, k=k, eta1=eta1, eta2=eta2))
        print(d_fdir+"   Mean Accuracy: {0:3.3f}, Standard Deviation: {1:3.3f}".format(np.mean(acc),np.std(acc)))





    
    
    
    
    
