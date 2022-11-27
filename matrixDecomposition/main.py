'''
要求完成课堂上讲的关于矩阵分解的LU、QR（Gram-Schmidt）、Orthogonal Reduction (Householder reduction 和Givens reduction)和 URV程序实现，要求如下：
        1、一个综合程序，根据选择参数的不同，实现不同的矩阵分解；在此基础上，实现Ax=b方程组的求解，以及计算A的行列式；
         2、可以用matlab、Python等编写程序，需附上简单的程序说明，比如参数代表什么意思，输入什么，输出什么等等，附上相应的例子；
         3、一定是可执行文件，例如 .m文件等,不能是word或者txt文档。附上源代码，不能为直接调用matlab等函数库;
'''
import numpy as np
from math import sqrt

def count_det(A, m, n):
    if m != n:
        print("该矩阵不是方阵，无法计算行列式！")
        return float("-inf")
    det = 1
    for i in range(n - 1):
        for j in range(i + 1, n):
            if A[i, i] == 0:
                print("ERROR: 使用朴素高斯消去法时主元素为0，无法进行LU分解！")
                return
            fac = A[j, i] / A[i, i]
            A[j] = A[j] - fac * A[i]
    for i in range(n):
        det = det * A[i, i]
    return det

def decomposition(A, b, n, operation_index = 1):
    print("A: ", A , '\n')
    if operation_index == 1:                           # LU分解
        L = np.identity(n, dtype = float)
        #for i in range(n):
        #    L[n][n] = 1
        U = A.copy()
        for i in range(n-1):
            #print(U[i,i-1])
            for j in range(i+1,n):
                if U[i, i] == 0:
                    print("ERROR: 使用朴素高斯消去法时主元素为0，无法进行LU分解！")
                    return
                fac = U[j, i] / U[i, i]                 # 求倍数
                #print("fac = ",fac)
                temp = np.identity(n, dtype = float)
                temp[j,i] = fac                         # G_i的逆
                L = np.dot(L, temp)                     # 更新L
                #print(U[i-1])
                U[j] = U[j] - fac*U[i]                  # 高斯消去法更新U
                #print(L)
                #print(U)
                #print("---")
        print(">>>LU分解已完成:")
        print("L: ", L)
        print("U: ", U)
        print("求解: ")
        b = b.T
        print(b)
        y = np.zeros((n, 1), dtype=float)
        for i in range(n):
            sum = b[0][i]
            #print(sum)
            for j in range(i):
                #print(j)
                sum = sum - y[j] * L[i][j]
            y[i] = sum
        print("y(Ly = b): ", y)
        # print("RX[0]: ", RX[0])
        x = np.zeros((n, 1), dtype=float)
        for i in range(n - 1, -1, -1):
            sum = y[i]
            for j in range(n - 1, i, -1):
                sum = sum - x[j] * U[i][j]
            x[i] = sum / U[i][i]
        print("x(Ux = y): ", x, "\n")
    elif operation_index == 2:                          # QR（Gram-Schmidt）分解
        A = A.T
        Q = np.zeros((n, n), dtype = float)
        R = np.zeros((n, n), dtype = float)
        print(">>>开始QR（Gram-Schmidt）分解：")
        for i in range(n):
            print("No. ",i+1,": ")
            for j in range(i):
                R[j,i] = round(np.dot(Q[j], A[i].T),4)
                print("q_",j," = ",Q[j],sep='')
                print("a_",i," = ",A[i],sep='')
                print('r', j, i, " = ", "q_",j,"^Ta_",i," = ",R[j,i],sep='')
                for k in range(3):
                    print(A[i,k],'-',round(R[j, i]*Q[j, k], 4),'=',end='')
                    A[i,k] = round(A[i,k],4) - round(R[j, i]*Q[j, k], 4)
                    A[i,k] = round(A[i,k],4)
                    print(A[i,k])
                print("a",i,'-r',j,i,"*q",j," = ",A[i],sep='')
            R[i, i] = round(sqrt(np.dot(A[i], A[i].T)),4)
            Q[i] = 1/R[i,i] * A[i]
            print('r', i, i, ": ", R[i, i], sep='')
        #Q = Q.T
        print(">>>QR（Gram-Schmidt）分解已完成:")
        print("Q: ", Q.T)
        print("R: ", R)
        print('------------------')
        print("求解: ")
        b = b.T
        #print(b)
        RX = np.zeros((n, 1), dtype = float)
        for i in range(n):
            RX[i] = round(np.dot(Q[i],b[0]),4)
        print("Rx = Q^Tb: ",RX)
        #print("RX[0]: ", RX[0])
        x = np.zeros((n, 1), dtype = float)
        for i in range(n-1,-1,-1):
            sum = RX[i]
            for j in range(n - 1, i, -1):
                sum = sum - x[j]*R[i][j]
            x[i] = sum / R[i][i]
        print("x: ",x, "\n")
    elif operation_index == 3: # Orthogonal Reduction (Householder reduction)
        Q = np.eye(n, dtype=float)
        R = A.copy()
        # den = 1.0
        for i in range(n-1):
            RT = R.T
            # print("RT : ", RT)
            u = RT[i][i:n].copy()
            norm = np.linalg.norm(u)                                               # 求向量模长|u|
            u[0] = u[0] - norm                                                     # u = u-e1，后面分式中上下模长平方可抵消
            # print("u: ", u)
            # print("uu^T : ", round(np.dot(u,u),4))
            # print("u^Tu : ", np.outer(u, u))
            den_temp = round(np.dot(u,u),4)                                        # uu^T，作分母
            R_temp = np.eye(n, dtype=float) * den_temp                             # 每次的Ri，左上角都为单位矩阵
            R_block = den_temp * np.eye(n-i, dtype=float) - 2 * np.outer(u, u)     # 右下角的矩阵块，I-2uu^T
            for j in range(n-i):
                for k in range(n-i):
                    R_temp[j+i][k+i] = R_block[j][k]                               # 矩阵块填入Ri
            #den = den * den_temp
            # print("R_temp : ", 1/den_temp * R_temp)
            # print("R : " , R)
            R = 1/den_temp * np.matmul(R_temp, R)                                  # 累乘
            Q = 1/den_temp * np.matmul(Q, R_temp.T)                                # 累乘
            # print("R = R_temp * R : ", R)
            # print("Q : ", Q)
            # print('>>>><<<<')

        print(">>>正交（Householder reduction）分解已完成:")
        print("Q: ", Q)
        print("R: ", R)
        # print(np.matmul(Q, R))                                                   # 验证：QR = A
        print('------------------')
        print("求解: ")
        RX = np.matmul(Q.T, b)
        print("Rx = Q^Tb: ", RX)
        x = np.zeros((n, 1), dtype=float)
        for i in range(n - 1, -1, -1):
            sum = RX[i]
            for j in range(n - 1, i, -1):
                sum = sum - x[j] * R[i][j]
            x[i] = sum / R[i][i]
        print("x: ", x, "\n")
    elif operation_index == 4:  # Orthogonal Reduction (Givens reduction)
        Q = np.eye(n, dtype=float)
        R = A.copy()
        P = np.eye(n, dtype=float)
        for i in range(n - 1):
            P_temp = np.eye(n, dtype=float)
            non_zero_index = i+1
            while R[i][i] == 0 and non_zero_index < n:  # 保证主元素非零
                if R[non_zero_index][i] != 0:
                    if R[i][non_zero_index] < 0:
                        P_temp[i][i] = -1
                    row_temp = P_temp[non_zero_index].copy()   # 两行对换的矩阵P
                    P_temp[non_zero_index] = P_temp[i].copy()
                    P_temp[i] = row_temp.copy()
                    break
                non_zero_index = non_zero_index + 1
            #print("P_temp : ", P_temp)
            P = np.matmul(P_temp, P)
            R = np.matmul(P_temp, R)
            #print("A : ", R)
            for j in range(i+1,n):
                if R[j][i] == 0:
                    continue
                P_temp = np.eye(n, dtype=float)
                #print("R[", i + 1, "][", i + 1, "] = ", R[i][i], sep='')
                #print("R[", j + 1, "][", i + 1, "] = ", R[j][i], sep='')
                den_temp = round(sqrt(round(R[i][i],4)**2 + round(R[j][i],4)**2), 4)  # 旋转所需要的分母
                #print("den = ", den_temp)
                c = round(R[i][i] / den_temp, 4)
                s = round(R[j][i] / den_temp, 4)
                P_temp[i][i] = P_temp[j][j] = c                                       # 更新P_ij矩阵
                P_temp[i][j] = s
                P_temp[j][i] = -s
                P = np.matmul(P_temp, P)                                              # 矩阵乘法
                R = np.matmul(P_temp, R)
                for index_i in range(n):                                              # 近似
                    for index_j in range(n):
                        P[index_i][index_j] = round(P[index_i][index_j], 3)
                        R[index_i][index_j] = round(R[index_i][index_j], 3)
                #print("P_",i+1,j+1," : ",P_temp,sep='')
                #print("P : ", P)
                #print("R : ", R)
                #print(">>>------<<<")

        Q = P.T
        print(">>>正交（Givens reduction）分解已完成:")
        print("Q: ", Q)
        print("R: ", R)
        print('------------------')
        print("求解: ")
        RX = np.matmul(Q.T, b)
        print("Rx = Q^Tb: ", RX)
        x = np.zeros((n, 1), dtype=float)
        for i in range(n - 1, -1, -1):
            sum = RX[i]
            for j in range(n - 1, i, -1):
                sum = sum - x[j] * R[i][j]
            x[i] = sum / R[i][i]
        print("x: ", x, "\n")
    elif operation_index == 5:  # URV
        pass
    #print(np.dot(L, U))
    #print(A)
    pass


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    A0 = np.array([[2,2,2],[4,7,7],[6,18,22]])
    # b0 = np.array([[0],[0],[0]])
    A1 = np.array([[1,2,3],[2,2,8],[-3,-10,-2]])
    b1 = np.array([[6],[12],[-15]])
    n = 3
    # print(count_det(A0, 3, 3))
    # decomposition(A1, b1, n, 1)

    A2 = np.array([[0.0,-20.0,-14.0],[3.0,27.0,-4.0],[4.0,11.0,-2.0]])
    b2 = np.array([[-20.0],[30.0],[15.0]])
    # decomposition(A2, b2, n, 2)
    A3 = np.array([[1.0, 19.0, -34.0], [-2.0, -5.0, 20.0], [2.0, 8.0, 37.0]])
    b3 = np.array([[6.0], [6.0], [57.0]])
    # decomposition(A3, b3, n, 3)
    A4 = np.array([[0.0, -20.0, -14.0], [3.0, 27.0, -4.0], [4.0, 11.0, -2.0]])
    b4 = np.array([[-14.0], [-1.0], [2.0]])
    decomposition(A4, b4, n, 4)

    #decomposition(A3, b3, n, 5)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
