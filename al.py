import tensorflow as tf
import numpy as np
import cvxpy as cp


# -------------------- Input data from matlab .mat file ---------------------
# Df 601x400
#Df = np.loadtxt(open("/home/ethanshan/Codes/TF114_AI/data/Df.csv", "rb"), delimiter=",")
Df = np.loadtxt(open("/home/toybrick/Codes/TF114_AI/data/Df.csv", "rb"), delimiter=",")
print("Df shape: " + str(Df.shape))

# exp_Input 400x1
#exp_Input = np.loadtxt(open("/home/ethanshan/Codes/TF114_AI/data/exp_Input.csv", "rb"), delimiter=",")
exp_Input = np.loadtxt(open("/home/toybrick/Codes/TF114_AI/data/exp_Input.csv", "rb"), delimiter=",")
print("exp_Input shape: " + str(exp_Input.shape))

# MAT_obs = 400x601
#MAT_obs = np.loadtxt(open("/home/ethanshan/Codes/TF114_AI/data/MAT_obs.csv", "rb"), delimiter=",")
MAT_obs = np.loadtxt(open("/home/toybrick/Codes/TF114_AI/data/MAT_obs.csv", "rb"), delimiter=",")
print("MAT_obs shape: " + str(MAT_obs.shape))

# ------------------- Input data from program parameters ----------------------------
Alpha = 20

# limit is a 1dim array, value from 0...600
limit = np.arange(0, 600, step=1, dtype=np.int)
print("limit shape: " + str(limit.shape))

h1 = 1
h2 = 20
l1 = 1
l2 = 20



# ------------------- Algorithm -------------------------------------
def calc_result(Df, MAT_obs, expv, Alpha, limit, h1, h2, l1, l2):
    # -------------------- Input data process ----------------------------
    MAT_20_20 = np.arange(0, 400, step=1, dtype=np.int)
    print("MAT_20_20 shape: " + str(MAT_20_20.shape))
    MAT_20_20 = np.array(MAT_20_20).reshape(20, 20)
    print("MAT_20_20 shape: " + str(MAT_20_20.shape))
    pick = MAT_20_20.flatten('F')
    print("pick shape: " + str(pick.shape))
    
    exp_Input = expv[pick]
    print("exp_Input shape: " + str(exp_Input.shape))
    
    MAT_here = MAT_obs[pick, :]
    print("MAT_here shape: " + str(MAT_here.shape))

    N_dim = Df.shape[1]
    print("Df column len: " + str(N_dim))

    Tf = MAT_here.dot(Df)
    print("Tf shape: " + str(Tf.shape))

    arr_1_400 = np.arange(0, 400, step=1, dtype=np.int)
    arr_1_400 = np.array(arr_1_400).reshape(1, 400)

    I = cp.Variable(N_dim)
    M = Df*I
    constraints = [M>=0]
    #objective = cp.Minimize(arr_1_400*I)
    objective = cp.Minimize(cp.norm(I, 1) + Alpha*cp.norm( (expv-Tf*I)))
    
    prob = cp.Problem(objective, constraints)
    result = prob.solve(verbose=True)

    #print(I.value)
    print("status: ", prob.status)
    print("optimal value: ", prob.value)
    #print("optimal value: " + prob.value)

    # convx problem resolve process
    #m = 30
    #n = 20
    #np.random.seed(1)
    #A = np.random.randn(m, n)
    #b = np.random.randn(m)
    
    # Construct the problem.
    #x = cp.Variable(n)
    #objective = cp.Minimize(cp.sum_squares(A*x - b))
    #constraints = [0 <= x, x <= 1]
    #prob = cp.Problem(objective, constraints)
    
    # The optimal objective value is returned by `prob.solve()`.
    #result = prob.solve()
    # The optimal value for x is stored in `x.value`.
    #print(x.value)
    # The optimal Lagrange multiplier for a constraint is stored in
    # `constraint.dual_value`.
    #print(constraints[0].dual_value)
    return h1 + h2


calc_result(Df, MAT_obs, exp_Input, Alpha, limit, h1, h2, l1, l2)

#val = np.dot(Df, exp_Input)

#val2 = calc_result(Df, MAT_obs, exp_Input, Alpha, limit, h1, h2, l1, l2)

#print(val);

#print(val2)

init_val = tf.random_normal((1, 5), 0.5, 1)

#print(init_val)

np.get_printoptions()
np.set_printoptions(precision=12)
print(np.get_printoptions())
#val3 = np.loadtxt(open("/home/ethanshan/Codes/TF114_AI/data/Df.csv", "rb"), dtype=np.double, delimiter=",")
#print("Rows" + str(len(val3)))
#print("Shape: " + str(val3.shape))
#print(val3)

