"""
    File name: Channel_Coding.py
    Description: This file provides functions for Channel Coding developed in Python3 
                 for research purposes.
    Author: Visuttha Manthamkarn
    Study: Yosita Tinjunchay 13 July 2024
    Affiliation : Department of Electrical Engineering, Kasetsart University
    Email: fengvtm@ku.ac.th
    Version: 1
    Date: 4 Aug 2022
"""

import numpy as np

""" Matrix Functions """
"""
    Function Name: MatrixDec_to_MatrixBinary
    Description: To convert Decial Matrix to Binary Matrix
    Input Function:
        1. Decimal Matrix: A matrix of decimal numbers.
                Example: np.array([32,10,7,9,63],dtype = np.uint64)
                Note: dtype should be np.uint64 (64 bits unsigned integer)
        2. Number_bit: Bit length
    Output Function:
        1. Binary Matrix:
                [[MSB, ..., LSB],
                 [MSB, ..., LSB],
                 [MSB, ..., LSB],
                        ...
                 ,[MSB ... LSB]]         
"""
def MatrixDec_to_MatrixBinary(Decimal_matrix,Number_bit):
    Number_bit = "#0"+str(Number_bit+2)+"b" #Format of Bit length 
    Decimal_to_Binary = lambda Decimal,Bits : np.fromiter(format(Decimal,Bits)[2:],dtype = int)
    return np.array([Decimal_to_Binary(Decimal,Number_bit) for Decimal in Decimal_matrix],dtype = int)

"""
    Function Name: MatrixBinary_to_MatrixDec
    Description: To convert Binary Matrix to Decimal Matrix
    Input Function:
        1. Binary Matrix:
                [[MSB, ..., LSB],
                 [MSB, ..., LSB],
                 [MSB, ..., LSB],
                        ...
                 ,[MSB ... LSB]] 
    Output Function:
        1. Decimal Matrix    
"""     
def MatrixBinary_to_MatrixDec(Binary_matrix):
    Binary_to_Decimal = lambda Binary : Binary.dot(np.uint64(1) << np.arange(Binary.size,dtype = np.uint64)[::-1])
    return np.array([Binary_to_Decimal(Binary) for Binary in Binary_matrix],dtype=np.uint64)

"""
    Function Name: Compute_Reduced_Row_Echelon_Form
    Description: To change Binary Matrix to its Row Echelon Form
    Input Function:
        1. Binary Matrix
    Output Function:
        1. Binary Matrix in Row Echelon Form
""" 
def Compute_Reduced_Row_Echelon_Form(Received_Matrix):
    Matrix_Binary = np.array(Received_Matrix,dtype=int)
    index_operation_row = 0
    row = 0
    while row < np.shape(Matrix_Binary)[0]:
        columns_one = np.where(Matrix_Binary[row,:] == 1)[0]
        if index_operation_row in columns_one:
            for sub_row in range( row+1,np.shape(Matrix_Binary)[0]):
                sub_columns_one = np.where(Matrix_Binary[sub_row,:] == 1)[0]
                if index_operation_row in sub_columns_one:
                    Matrix_Binary[sub_row] = Matrix_Binary[sub_row]^Matrix_Binary[row]
            index_operation_row = index_operation_row + 1
            row = row + 1
        else:
            if row == index_operation_row:
                Flag = 0
                for sub_row in range(row+1,np.shape(Matrix_Binary)[0]):
                    sub_columns_one = np.where(Matrix_Binary[sub_row,:] == 1)[0]
                    if index_operation_row in sub_columns_one:
                        operation_row = np.array(Matrix_Binary[row])
                        current_row = np.array(Matrix_Binary[sub_row])
                        Matrix_Binary[row] = current_row
                        Matrix_Binary[sub_row] = operation_row
                        Flag = 1
                        break
                if Flag == 0:
                    return Matrix_Binary
    index_operation_row = np.shape(Matrix_Binary)[0] - 1
    for row in range(np.shape(Matrix_Binary)[0]-2,-1,-1):
        columns_one = np.where(Matrix_Binary[row,:] == 1)[0]
        columns_one = columns_one[columns_one<=index_operation_row]
        columns_one = columns_one[columns_one>row]
        for sub_row in columns_one:
            Matrix_Binary[row] = Matrix_Binary[row]^Matrix_Binary[sub_row]
    return Matrix_Binary

"""
    Function Name: Compute_Inverse_Binary_Matrix
    Description: To Inverse Binary Matrix using Row Operation ([X|I] -> [I|X'])
    Input Function:
        1. Binary Matrix
                Note: Binary Matrix must be a Square Matrix and has Full Rank
    Output Function:
        1. Inversed Binary Matrix 
"""
def Compute_Inverse_Binary_Matrix(Original_matrix):
    Binary_matrix = np.array(Original_matrix,dtype=int)
    Inverse_Binary_matrix = np.identity(Binary_matrix.shape[0],dtype = np.uint64)
    index_operation_col = 0
    rows = 0
    while rows < Binary_matrix.shape[0]:
        columns_one = np.where(Binary_matrix[rows,:] == 1)[0]
        if len(columns_one) > 0:
            if index_operation_col in columns_one:
                other_index = np.where(columns_one != index_operation_col)[0]
                for row_modulo in range(np.shape(Binary_matrix)[0]):
                    for col_modulo in other_index:
                        Binary_matrix[row_modulo][columns_one[col_modulo]] = (Binary_matrix[row_modulo][columns_one[col_modulo]] + Binary_matrix[row_modulo][index_operation_col])%2
                        Inverse_Binary_matrix[row_modulo][columns_one[col_modulo]] = (Inverse_Binary_matrix[row_modulo][columns_one[col_modulo]] + Inverse_Binary_matrix[row_modulo][index_operation_col])%2
                index_operation_col = index_operation_col + 1
                rows = rows + 1
            else:
                change_index_col = np.where(columns_one > index_operation_col)[0]
                if len(change_index_col) > 0:
                    Binary_matrix[:,[index_operation_col,columns_one[change_index_col[0]]]] = Binary_matrix[:,[columns_one[change_index_col[0]],index_operation_col]]
                    Inverse_Binary_matrix[:,[index_operation_col,columns_one[change_index_col[0]]]] = Inverse_Binary_matrix[:,[columns_one[change_index_col[0]],index_operation_col]]
                else:
                    rows = rows + 1
        else:
            rows = rows + 1
    if index_operation_col != Binary_matrix.shape[0]:
        raise ValueError(
            "Compute_Inverse_Binary_Matrix: matrix is singular over GF(2) "
            f"(rank {index_operation_col} < {Binary_matrix.shape[0]})"
        )
    return Inverse_Binary_matrix

"""
    Function Name: Compute_Gauss_Jordan_Redection
    Description: To change Binary Matrix to its Row Echelon Form
                 using Gauss-Jordan reduction with Column Operation
    Input Function:
        1. Binary Matrix
    Output Function:
        1. Binary Matrix in Reduced Row Echelon Form
        2. Rank of Matrix
        3. Independent Row Position
"""
def Compute_Gauss_Jordan_Reduction(Original_matrix):
    Binary_matrix = np.array(Original_matrix,dtype = int)
    index_operation_col = 0
    rows = 0
    Index_Rows = []
    while rows < np.shape(Binary_matrix)[0]:
        columns_one = np.where(Binary_matrix[rows,:] == 1)[0]
        if len(columns_one) > 0:
            if index_operation_col in columns_one:
                other_index = np.where(columns_one != index_operation_col)[0]
                for row_modulo in range(np.shape(Binary_matrix)[0]):
                    for col_modulo in other_index:
                        Binary_matrix[row_modulo][columns_one[col_modulo]] = (Binary_matrix[row_modulo][columns_one[col_modulo]] + Binary_matrix[row_modulo][index_operation_col])%2
                Index_Rows.append(rows)
                index_operation_col = index_operation_col + 1
                rows = rows + 1
                if index_operation_col == np.shape(Binary_matrix)[1]:
                    break
            else:
                change_index_col = np.where(columns_one > index_operation_col)[0]
                if len(change_index_col) > 0:
                    Binary_matrix[:,[index_operation_col,columns_one[change_index_col[0]]]] = Binary_matrix[:,[columns_one[change_index_col[0]],index_operation_col]]
                else:
                    rows = rows + 1
        else:
            rows = rows + 1
    return Binary_matrix,index_operation_col,Index_Rows

""" Channel Models """
"""
    Function Name: q_ary_error_channel
    Description: To random Error position using q-ary Symmetric Channel model
    Input Function:
        1. Probability of Error
        2. The number of symbols
    Output Function:
        1. An array of Error position
                Example: [0, 0, 1, 0, 0, 1, ..., 0]
                Note: 1 = Error position
"""
def q_ary_error_channel(prob_error,num):
    rng = np.random.default_rng()
    Probability = [1-prob_error, prob_error]
    channel = rng.choice(2, num, p=Probability)
    return channel

"""
    Function Name: q_ary_erasure_channel
    Description: To random Error position using q-ary Erasure Channel model
    Input Function:
        1. Probability of Erasure
        2. The number of symbols
    Output Function:
        1. An array of Erasure position
                Example: [0, 0, 1, 0, 0, 1, ..., 0]
                Note: 1 = Erasure position
"""
def q_ary_erasure_channel(prob_erasure,num):
    rng = np.random.default_rng()
    Probability = [1-prob_erasure, prob_erasure]
    channel = rng.choice(2, num, p=Probability)
    return channel

"""
    Function Name: q_ary_error_erasure_channel
    Description: To random Error postion using q-ary Error-Erasure Channel model
    Input Function:
        1. Probability of Error
        2. Probability of Erasure
        3. The number of symbols
    Output Function:
        1. An array of Error and Erasure position
                Example: [2, 0, 1, 0, 2, 1, ..., 0]
                Note: 1 = Error position
                      2 = Erasure position
"""
def q_ary_error_erasure_channel(prob_error,prob_erasure,num):
    rng = np.random.default_rng()
    Probability = [1-prob_error-prob_erasure, prob_error, prob_erasure]
    channel = rng.choice(3, num, p=Probability)
    return channel

""" Functions for Block Code """
"""
    Function Name: Generate_Systematic_H
    Description: To Random Systematic Parity Check Matrix which Q is constructed using 
                 a pseudo-random number generator with a uniform distribution
                 Note: Q will be generated as a full rank matrix
    Input Function:
        1. The number of Symbols (n)
        2. The number of Data Symbols (k)
        3. Matrix Construction
                Note: 'Q|I' or 'I|Q'
        4. Random Seed
        5. Save Condition
                Note: 1 = Save Output -> Parity Check Matrix (H)
                      File Name: H(n,k).npy
    Output Function:
        1. Parity Check Matrix (H)
"""
def Generate_Systematic_H(n,k,type,seed=0,save=0):
    Num_Parity = n-k
    Num_Data = k
    rng = np.random.default_rng(seed)
    while True:
        Q = rng.choice(2, Num_Data*Num_Parity, p=[0.5,0.5]).reshape(Num_Parity,Num_Data)
        _,Q_rank,_ = Compute_Gauss_Jordan_Reduction(Q)
        if Q_rank == np.shape(Q)[0]:
            break
    Identity_Matrix = np.identity(Q.shape[0], dtype=int)
    if type == 'Q|I':
        H = np.concatenate((Q,Identity_Matrix), axis=1)
    elif type == 'I|Q':
        H = np.concatenate((Identity_Matrix,Q), axis=1)
    else:
        exit("Please Select [Q|I] or [I|Q]")
    if save:
        np.save(f'H({n},{k})',H)
    return H

"""
    Function Name: Encode_Systematic_H
    Description: To Encode Data by using Generator Matrix constructed by Systematic Parity Check Matrix 
    Input Function:
        1. Decimal Data
        2. Systematic Parity Check Matrix (H)
        3. Matrix Construction
                Note: 'Q|I' or 'I|Q'
    Output Function:
        1. Codeword
"""
def Encode_Systematic_H(Data,H,type):
    if len(Data) != np.shape(H)[1]-np.shape(H)[0]:
        exit("Wrong length of Data")
    if type == 'Q|I':
        Q = H[:,:np.shape(H)[1]-np.shape(H)[0]]
        Q_T = np.transpose(Q)
        Identity_Matrix = np.identity(Q_T.shape[0],dtype=int)
        #G = [I|Q^T]
        G = np.concatenate((Identity_Matrix,Q_T),axis=1)
    elif type == 'I|Q':
        Q = H[:,np.shape(H)[1]-np.shape(H)[0]:]
        Q_T = np.transpose(Q)
        Identity_Matrix = np.identity(Q_T.shape[0],dtype=int)
        #G = [Q^T|I]
        G = np.concatenate((Q_T,Identity_Matrix),axis=1)
    else:
        exit("Please Select [Q|I] or [I|Q]")
    Codeword = np.zeros(np.shape(G)[1],dtype=np.uint64)
    for i in range(0,G.shape[1]):
        for j in range(0,G.shape[0]):
            if G[j][i] == 1:
                Codeword[i] = int(Codeword[i])^int(Data[j])
    return Codeword

""" hMP Functions"""
"""
    Function Name: Compute_Check_Nodes_Matrix
    Description: To Compute Check Node values for hMP decoder
    Input Function:
        1. Parity Check Matrix (H)
        2. Decimal Symbol Matrix (Y)
    Output Function:
        1. Check Node values
"""
def Compute_Check_Nodes_Matrix(H_matrix,Y_matrix):
    #Initial Check Node Matrix
    Check_Node_matrix = np.zeros(np.shape(H_matrix)[0],dtype = np.uint64)
    #Find Corresponding H For Each Check Node
    Find_H_Sub = lambda x : np.where(x==1)[0]
    H_Sub_matrix = [Find_H_Sub(h) for h in H_matrix]
    #Compute Each Check Node
    for i in range(len(H_Sub_matrix)):
        for postion in H_Sub_matrix[i]:
            Check_Node_matrix[i] = Check_Node_matrix[i]^Y_matrix[postion]
    return Check_Node_matrix

"""
    Function Name: Compute_Verified_Symbols_Matrix
    Description: To Verify symbols for hMP decoder
    Input Function:
        1. Parity Check Matrix (H)
        2. Check Node values
    Output Function:
        1. Verified Symbol Positions
                Note: 0 = Unverified Symbol
                      1 = Verified Symbol
"""
def Compute_Verified_Symbols_Matrix(H_matrix,Check_Node_matrix):
    #Find Check Node = 0
    Check_Node_Zero = np.where(Check_Node_matrix==0)[0]
    #Find Corresponding H For Check Node = 0
    Selected_H = H_matrix[Check_Node_Zero,:]
    #Compute Verified Symbol Positions
    Verified_matrix = Selected_H.sum(axis=0)
    Verified_matrix[Verified_matrix>1] = 1
    return Verified_matrix

"""
    Function Name: hMP
    Description: hMP Decoder
    Input Function:
        1. Parity Check Matrix (H)
        2. Decimal Symbol Matrix (Y)
    Output Function:
        1. Decoded Codeword
        2. Flag Complete
                Note: 0 = Decoding Failed
                      1 = Decoding Success
        3. Verified Symbol Positions
                Note: 0 = Unverified Symbol
                      1 = Verified Symbo
"""
def hMP(H,Received_Matrix):
    Y = np.array(Received_Matrix,dtype=np.uint64)
    #Compute Check Node matrix
    Total_Check_Node = Compute_Check_Nodes_Matrix(H,Y)
    #Compute Verified Symbols
    Verified = Compute_Verified_Symbols_Matrix(H,Total_Check_Node)
    #if No Error : return Decoding Success
    if len(np.where(Verified == 0)[0]) == 0:
        return Y,1,Verified
    list_position_check_node = [np.where(check_node == 1)[0] for check_node in H]
    while True:
        position_Unverified = np.where(Verified == 0)[0]
        list_position_Common = [list(set(position_check_node).intersection(position_Unverified)) for position_check_node in list_position_check_node]
        one_position_Common = np.where(np.array([len(position_Common) for position_Common in list_position_Common],dtype = int) == 1)[0]
        #Have One unverified symbol
        if len(one_position_Common) != 0:
            position_Common = list_position_Common[one_position_Common[0]][0]
            Y[position_Common] = Y[position_Common]^Total_Check_Node[one_position_Common[0]]
            Total_Check_Node = Compute_Check_Nodes_Matrix(H,Y)
            Verified = Compute_Verified_Symbols_Matrix(H,Total_Check_Node)
            continue
        #Check more than one unverified symbols connected to a check node        
        if np.count_nonzero(Verified == 0) > 0:
            position_Unverified = np.where(Verified == 0)[0]
            Check_Node_Value = {}
            for value in Total_Check_Node:
                Check_Node_Value[value] = Check_Node_Value.get(value,0) + 1
            Check_Node_Value = dict(filter(lambda x:x[1]>=2 and x[0]!=0,Check_Node_Value.items()))
            Flag_Finish = 1
            for key,value in Check_Node_Value.items():
                Check_Node = np.where(Total_Check_Node == key)[0]
                Common_Y_in_H = np.ones(np.shape(H)[1],dtype = int)
                for index in Check_Node:
                    Common_Y_in_H = Common_Y_in_H & H[index]
                position_Common_Y_in_H = np.where(Common_Y_in_H == 1)[0]
                position_Common = list(set(position_Common_Y_in_H).intersection(position_Unverified))
                if len(position_Common) == 1:
                    Y[position_Common] = Y[position_Common]^key
                    Total_Check_Node = Compute_Check_Nodes_Matrix(H,Y)
                    Verified = Compute_Verified_Symbols_Matrix(H,Total_Check_Node)
                    Flag_Finish = 0
                    if len(np.where(Verified == 0)[0]) == 0:
                        return Y,1,Verified
                    break
            if Flag_Finish == 1:
                if len(np.where(Verified == 0)[0]) == 0:
                    return Y,1,Verified
                else:
                    return Y,0,Verified
        else:
            return Y,1,Verified

"""
    Function Name: hMP_Group_Common
    Description: improved hMP Decoder (able to detect group of unverified symbols)
    Input Function:
        1. Parity Check Matrix (H)
        2. Decimal Symbol Matrix (Y)
    Output Function:
        1. Decoded Codeword
        2. Flag Complete
                Note: 0 = Decoding Failed
                      1 = Decoding Success
        3. Verified Symbol Positions
                Note: 0 = Unverified Symbol
                      1 = Verified Symbo
"""
def hMP_Group_Common(H,Received_Matrix):
    Y = np.array(Received_Matrix,dtype=np.uint64)
    #Compute Check Node matrix
    Total_Check_Node = Compute_Check_Nodes_Matrix(H,Y)
    #Compute Verified Symbols
    Verified = Compute_Verified_Symbols_Matrix(H,Total_Check_Node)
    #if No Error : return Decoding Success
    if len(np.where(Verified == 0)[0]) == 0:
        return Y,1,Verified
    list_position_check_node = [np.where(check_node == 1)[0] for check_node in H]
    Extra_Verified = np.zeros(len(Verified),dtype=int)
    while True:
        position_Unverified = np.where(Verified == 0)[0]
        list_position_Common = [list(set(position_check_node).intersection(position_Unverified)) for position_check_node in list_position_check_node]
        one_position_Common = np.where(np.array([len(position_Common) for position_Common in list_position_Common],dtype = int) == 1)[0]
        #Have One unverified symbol
        if len(one_position_Common) != 0:
            position_Common = list_position_Common[one_position_Common[0]][0]
            Y[position_Common] = Y[position_Common]^Total_Check_Node[one_position_Common[0]]
            Total_Check_Node = Compute_Check_Nodes_Matrix(H,Y)
            Verified = Compute_Verified_Symbols_Matrix(H,Total_Check_Node)
            Verified = Verified|Extra_Verified
            continue
        #Check more than one unverified symbols connected to a check node        
        if np.count_nonzero(Verified == 0) > 0:
            position_Unverified = np.where(Verified == 0)[0]
            Check_Node_Value = {}
            for value in Total_Check_Node:
                Check_Node_Value[value] = Check_Node_Value.get(value,0) + 1
            Check_Node_Value = dict(filter(lambda x:x[1]>=2 and x[0]!=0,Check_Node_Value.items()))
            Flag_Finish = 1
            for key,value in Check_Node_Value.items():
                Check_Node = np.where(Total_Check_Node == key)[0]
                Common_Y_in_H = np.ones(np.shape(H)[1],dtype = int)
                for index in Check_Node:
                    Common_Y_in_H = Common_Y_in_H & H[index]
                position_Common_Y_in_H = np.where(Common_Y_in_H == 1)[0]
                position_Common = list(set(position_Common_Y_in_H).intersection(position_Unverified))
                if len(position_Common) == 1:
                    Y[position_Common] = Y[position_Common]^key
                    Total_Check_Node = Compute_Check_Nodes_Matrix(H,Y)
                    Verified = Compute_Verified_Symbols_Matrix(H,Total_Check_Node)
                    Verified = Verified|Extra_Verified
                    Flag_Finish = 0
                    if len(np.where(Verified == 0)[0]) == 0:
                        return Y,1,Verified
                    break
                #Group Unverified Symbol
                else:
                    Check_Extra = np.array(Verified,dtype=int)
                    for index in Check_Node:
                        Extra_Verified = Extra_Verified | H[index]
                    for common in position_Common:
                        Extra_Verified[common] = 0
                    Verified = Verified|Extra_Verified
                    if np.array_equal(Check_Extra,Verified,equal_nan=True)==False:
                        Flag_Finish = 0
                    continue
            if Flag_Finish == 1:
                if len(np.where(Verified == 0)[0]) == 0:
                    return Y,1,Verified
                else:
                    return Y,0,Verified
        else:
            return Y,1,Verified

""" VSD Functions"""
"""
    Function Name: Compute_Error_Locating_Vector
    Description: To Compute Error Locating Vector for VSD
    Input Function:
        1. Syndrome (Binary matrix) in Reduced Row Echelon Form
                Note: Output of Function Compute_Gauss_Jordan_Reduction
        2. Independent Row Position
                Note: Output of Function Compute_Gauss_Jordan_Reduction
        3. Parity Check Matrix (H)
    Output Function:
        1. Error Locating Vector
                Note: 0 = Erroneous Symbol
                      1 = Correct Symbol
"""
def Compute_Error_Locating_Vector(Syndrome_Binary_matrix,Index_Rows,H_matrix):
    Error_Locating_Vector = np.zeros(np.shape(H_matrix)[1], dtype = int)
    Select_Index_of_Rows = np.array([i for i in range(np.shape(Syndrome_Binary_matrix)[0]) if i not in Index_Rows], dtype = int)
    for rows in Select_Index_of_Rows:
        if sum(Syndrome_Binary_matrix[rows]) > len(Index_Rows):
            pass
        else:
            Temp_Vector = np.zeros(np.shape(H_matrix)[1], dtype = int)
            Temp_Vector = Temp_Vector^H_matrix[rows]
            Col_One = np.where(Syndrome_Binary_matrix[rows] == 1)[0]
            for col in Col_One:
                Temp_Vector = Temp_Vector^H_matrix[Index_Rows[col]]
            Error_Locating_Vector = Error_Locating_Vector|Temp_Vector
    return Error_Locating_Vector

"""
    Function Name: VSD
    Description: Traditional VSD
    Input Function:
        1. Parity Check Matrix (H)
        2. Decimal Symbol Matrix (Y)
    Output Function:
        1. Decoded Codeword
        2. Flag Complete
                Note: 0 = Decoding Failed
                      1 = Decoding Success
"""
def VSD(H,Y):
    Y_Binary = np.array(Y,dtype = int)
    #Compute Symdrome Matrix
    S_Binary = np.floor(np.mod(np.dot(H,Y_Binary),2)).astype(int)
    S_Gauss,S_rank,Index_Rows = Compute_Gauss_Jordan_Reduction(S_Binary)
    #Compute Error Locating Vector
    Error_Locating_Vector = Compute_Error_Locating_Vector(S_Gauss,Index_Rows,H)
    #Find Number of Erroneous Symbol
    Number_Error = np.count_nonzero(Error_Locating_Vector == 0)
    if Number_Error == 0:
        return MatrixBinary_to_MatrixDec(Y),1
    #Rank of S = Number of Erroneous Symbol
    if S_rank == Number_Error:
        S_Sub = S_Binary[Index_Rows,:]
        Position_Error = np.where(Error_Locating_Vector==0)[0]
        H_Sub = H[np.ix_(Index_Rows,Position_Error)]
        H_Sub_inv = Compute_Inverse_Binary_Matrix(H_Sub)
        Error_Binary = np.floor(np.mod(np.dot(H_Sub_inv,S_Sub),2)).astype(int)
        for index in range(len(Position_Error)):    
            Y_Binary[Position_Error[index]] = Y_Binary[Position_Error[index]]^Error_Binary[index]
        Y_decode = MatrixBinary_to_MatrixDec(Y_Binary)
        return Y_decode,1
    else:
        return MatrixBinary_to_MatrixDec(Y),0

"""
    Function Name: BVSD
    Description: Limited Complexity VSD by limiting the number of correctable erroneous symbols
    Input Function:
        1. Parity Check Matrix (H)
        2. Decimal Symbol Matrix (Y)
        3. Limit
    Output Function:
        1. Decoded Codeword
        2. Flag Complete
                Note: 0 = Decoding Failed
                      1 = Decoding Success
        3. Error Locating Vector
"""
def BVSD(H,Y,limit):
    Y_Binary = np.array(Y,dtype = int)
    #Compute Symdrome Matrix
    S_Binary = np.floor(np.mod(np.dot(H,Y_Binary),2)).astype(int)
    S_Gauss,S_rank,Index_Rows = Compute_Gauss_Jordan_Reduction(S_Binary)
    #Compute Error Locating Vector
    Error_Locating_Vector = Compute_Error_Locating_Vector(S_Gauss,Index_Rows,H)
    #Find Number of Erroneous Symbol
    Number_Error = np.count_nonzero(Error_Locating_Vector == 0)
    if Number_Error == 0:
        return MatrixBinary_to_MatrixDec(Y),1,Error_Locating_Vector
    #Rank of S = Number of Erroneous Symbol
    if S_rank == Number_Error and Number_Error <= limit:
        S_Sub = S_Binary[Index_Rows,:]
        Position_Error = np.where(Error_Locating_Vector==0)[0]
        H_Sub = H[np.ix_(Index_Rows,Position_Error)]
        H_Sub_inv = Compute_Inverse_Binary_Matrix(H_Sub)
        Error_Binary = np.floor(np.mod(np.dot(H_Sub_inv,S_Sub),2)).astype(int)
        for index in range(len(Position_Error)):    
            Y_Binary[Position_Error[index]] = Y_Binary[Position_Error[index]]^Error_Binary[index]
        Y_decode = MatrixBinary_to_MatrixDec(Y_Binary)
        return Y_decode,1
    else:
        return MatrixBinary_to_MatrixDec(Y),0,Error_Locating_Vector



"""
    Function Name: VSD_Data
    Description: Improved VSD for Systematic Parity Check Matrix by correcting only erroneous data symbols (Pretty Patent 2022)
    Input Function:
        1. Systematic Parity Check Matrix (H)
                Note: H must be constructed as 'Q|I'
        2. Decimal Symbol Matrix (Y)
    Output Function:
        1. Decoded Codeword
        2. Flag Complete
                Note: 0 = Decoding Failed
                      1 = Decoding Success
"""
def VSD_Data(H,Y):
    Y_Binary = np.array(Y,dtype = int)
    S_Binary = np.floor(np.mod(np.dot(H,Y_Binary),2)).astype(int)
    S_Gauss,S_rank,Index_Rows = Compute_Gauss_Jordan_Reduction(S_Binary)
    Error_Locating_Vector = Compute_Error_Locating_Vector(S_Gauss,Index_Rows,H)
    Number_Error = np.count_nonzero(Error_Locating_Vector == 0)
    if Number_Error == 0:
        return MatrixBinary_to_MatrixDec(Y),1
    if S_rank == Number_Error:
        S_Sub = S_Binary[Index_Rows,:]
        Position_Error = np.where(Error_Locating_Vector==0)[0]
        Position_Data_Error = Position_Error[np.where(Position_Error<(H.shape[1]-H.shape[0]))[0]]
        Position_Parity_Error = Position_Error[np.where(Position_Error>=(H.shape[1]-H.shape[0]))[0]]
        H_Sub = H[np.ix_(Index_Rows,Position_Error)]
        Select_row = np.where(np.sum(H_Sub[:,range(len(Position_Data_Error),len(H_Sub))],axis=1) == 0)[0]
        H_Sub_data = H_Sub[Select_row,:]
        H_Sub_data = H_Sub_data[:,range(len(Position_Data_Error))]
        S_Sub_data = S_Sub[Select_row,:]
        H_Sub_data_inv = Compute_Inverse_Binary_Matrix(H_Sub_data)
        Error_Binary = np.floor(np.mod(np.dot(H_Sub_data_inv,S_Sub_data),2)).astype(int)
        for index in range(len(Position_Data_Error)):    
            Y_Binary[Position_Data_Error[index]] = Y_Binary[Position_Data_Error[index]]^Error_Binary[index]
        Y_decode = MatrixBinary_to_MatrixDec(Y_Binary)
        return Y_decode,1
    else:
        return MatrixBinary_to_MatrixDec(Y),0

"""
    Function Name: BVSD_data
    Description: Limited Complexity VSD by limiting the number of correctable erroneous symbols and improved VSD by correcting only erroneous data symbols
    Input Function:
        1. Parity Check Matrix (H)
        2. Decimal Symbol Matrix (Y)
        3. Limit
    Output Function:
        1. Decoded Codeword
        2. Flag Complete
                Note: 0 = Decoding Failed
                      1 = Decoding Success
        3. Error Locating Vector
"""

def BVSD_data(H,Y,limit):
    Y_Binary = np.array(Y,dtype = int)
    #Compute Symdrome Matrix
    S_Binary = np.floor(np.mod(np.dot(H,Y_Binary),2)).astype(int)
    S_Gauss,S_rank,Index_Rows = Compute_Gauss_Jordan_Reduction(S_Binary)
    #Compute Error Locating Vector
    Error_Locating_Vector = Compute_Error_Locating_Vector(S_Gauss,Index_Rows,H)
    Position_Error = np.where(Error_Locating_Vector==0)[0]
    Position_Data_Error = Position_Error[np.where(Position_Error<(H.shape[1]-H.shape[0]))[0]]
    Position_Parity_Error = Position_Error[np.where(Position_Error>=(H.shape[1]-H.shape[0]))[0]]
      #Find Number of Erroneous Symbol
    Number_Error1 = np.count_nonzero(Error_Locating_Vector == 0)
    Number_Error2 = len(Position_Data_Error)
    if Number_Error2 == 0:
        return MatrixBinary_to_MatrixDec(Y),1,Error_Locating_Vector
    #Rank of S = Number of Erroneous Symbol
    if S_rank == Number_Error1 and Number_Error2 <= limit:
        S_Sub = S_Binary[Index_Rows,:]
        H_Sub = H[np.ix_(Index_Rows,Position_Error)]
        Select_row = np.where(np.sum(H_Sub[:,range(len(Position_Data_Error),len(H_Sub))],axis=1) == 0)[0]
        H_Sub_data = H_Sub[Select_row,:]
        H_Sub_data = H_Sub_data[:,range(len(Position_Data_Error))]
        S_Sub_data = S_Sub[Select_row,:]
        H_Sub_data_inv = Compute_Inverse_Binary_Matrix(H_Sub_data)
        Error_Binary = np.floor(np.mod(np.dot(H_Sub_data_inv,S_Sub_data),2)).astype(int)
        for index in range(len(Position_Data_Error)):    
            Y_Binary[Position_Data_Error[index]] = Y_Binary[Position_Data_Error[index]]^Error_Binary[index]
        Y_decode = MatrixBinary_to_MatrixDec(Y_Binary)
        return Y_decode,1
    else:
        return MatrixBinary_to_MatrixDec(Y),0,Error_Locating_Vector

""" Rateless Code Functions"""
"""
    Function Name: Generate_Parity
    Description: To Random Parity Check Equation constructed using 
                 a pseudo-random number generator with a uniform distribution
    Input Function:
        1. The number of Parity Check Symbols (n)
        2. The number of Data Symbols (m)
        3. Random Seed
        4. Save Condition
                Note: 1 = Save Output -> Parity Check Equation (Q)
                      File Name: Q(n,k).npy
    Output Function:
        1. Parity Check Equation Matrix (Q)
"""
def Generate_Parity(n,m,seed=0,save=0):
    rng = np.random.default_rng(seed)
    Q = rng.choice(2, n*m, p=[0.5,0.5]).reshape(n,m)
    if save:
        np.save(f'Q({n},{m})',Q)
    return Q

"""
    Function Name: Generate_Data
    Description: To Random Decimal Data 
    Input Function:
        1. The number of Data Symbols (m)
        2. Random Seed
        3. Save Condition
                Note: 1 = Save Output -> Data
                      File Name: Data(m).npy
    Output Function:
        1. Decimal Data 
"""
def Generate_Data(m, Number_bit,seed=0,save=0):
    rng = np.random.default_rng(seed)
    Data_Binary = rng.choice(2, Number_bit*m, p=[0.5,0.5]).astype(int).reshape(m,Number_bit)
    Data = MatrixBinary_to_MatrixDec(Data_Binary)
    if save:
        np.save(f'Data({m})',Data)
    return Data

"""
    Function Name: Compute_Parity_Value
    Description: To Calculate Parity Check Values
    Input Function:
        1. Parity Check Equation Matrix (Q)
        2. Decimal Data 
        3. Save Condition
                Note: 1 = Save Output -> Parity Values
                      File Name: Parity_Value(n,m).npy
    Output Function:
        1. Parity Check Values
"""
def Compute_Parity_Value(Q,Data,save=0):
    Parity_Value = np.zeros(Q.shape[0],dtype=np.uint64)
    for index in range(len(Q)):
        for index_data in np.where(Q[index]==1)[0]:
            Parity_Value[index] = Parity_Value[index]^Data[index_data]
    if save:
        np.save(f'Parity_Value({Q.shape[0]},{Q.shape[1]})',Parity_Value)
    return Parity_Value

"""
    Function Name: Erasure_Channel_Converter
"""
def Erasure_Channel_Converter(Q,Data_Error,Data_Erasure_Position,Parity_Value_Error,Parity_Erasure_Position,Pull_Buffer):
    
    #Maximum Number of Parity
    limit_parity = len(Parity_Value_Error)

    Position_Buffer = 0
    Number_bit = 32
    Additional_bit = 2

    #Initial Variable
    Selected_Parity_Matrix = np.empty((0, len(Data_Error)), int)
    Selected_Parity_Value = np.array([],dtype=np.uint64)

    while True:

        if limit_parity != -1:
            if Position_Buffer > limit_parity:
                # print('Reach Maximum Number of Parity')
                # exit()
                return None  ## - Added by Nopparuj

        Number_bit = 32
        for Erasure in Data_Erasure_Position:
            Data_Error[Erasure] = np.random.randint(low = 2**(Number_bit), high = 2**(Number_bit+Additional_bit), dtype = np.uint64)
        Number_bit = Number_bit + Additional_bit
        
        #Pull Parity from Buffer
        Pull_Parity_Matrix = np.array(Q[Position_Buffer:Position_Buffer+Pull_Buffer])    
        Pull_Parity_Value  = np.array(Parity_Value_Error[Position_Buffer:Position_Buffer+Pull_Buffer])
        #Delete Erasure Parity Symbols
        Erasure_Pull_Parity = Parity_Erasure_Position[(Parity_Erasure_Position>=Position_Buffer)&(Parity_Erasure_Position<Position_Buffer + Pull_Buffer)] - Position_Buffer
        Pull_Parity_Matrix = np.delete(Pull_Parity_Matrix,Erasure_Pull_Parity,axis=0)
        Pull_Parity_Value  = np.delete(Pull_Parity_Value,Erasure_Pull_Parity)
        #Shift Buffer Position
        Position_Buffer = Position_Buffer + Pull_Buffer
        #Add New Parity
        Selected_Parity_Matrix = np.append(Selected_Parity_Matrix,Pull_Parity_Matrix,axis=0)
        Selected_Parity_Value = np.append(Selected_Parity_Value,Pull_Parity_Value)
        #Create H
        Identity_Matrix = np.identity(Selected_Parity_Matrix.shape[0],dtype=int)
        Selected_H = np.concatenate((Selected_Parity_Matrix,Identity_Matrix),axis=1)
        #Add Parity Symbols
        Y = np.append(Data_Error,Selected_Parity_Value)
        Y_Binary = MatrixDec_to_MatrixBinary(Y,Number_bit)
        S_Binary = np.floor(np.mod(np.dot(Selected_H,Y_Binary),2)).astype(int)
        #Error Locating Vector
        S_Gauss,S_rank,Index_Rows = Compute_Gauss_Jordan_Reduction(S_Binary)
        Error_Locating_Vector = Compute_Error_Locating_Vector(S_Gauss,Index_Rows,Selected_H)
        Number_Error = np.count_nonzero(Error_Locating_Vector == 0)
        if Number_Error == 0:
            return Data_Error
        #Rank of S = The Number of Error
        if S_rank == Number_Error:
            #Seperate Index
            Index_Error = np.where(Error_Locating_Vector==0)[0]
            Data_Erasure_Position = np.array(Index_Error[Index_Error<len(Data_Error)])
            Index_Error_Parity = Index_Error[Index_Error>=len(Data_Error)] - len(Data_Error)
            if len(Data_Erasure_Position)>0:
                for Erasure in Data_Erasure_Position:
                    Data_Error[Erasure] = 0
            if len(Index_Error_Parity)> 0 :
                Selected_Parity_Matrix = np.delete(Selected_Parity_Matrix,Index_Error_Parity,axis=0)
                Selected_Parity_Value = np.delete(Selected_Parity_Value,Index_Error_Parity)
            #Fill In Erasure Data
            if len(Data_Erasure_Position):
                while(len(Data_Erasure_Position)):
                    Flag_Fill_In = 1
                    Fill_Data_Index = np.array([],dtype=int)
                    Partial_Parity_Matrix = Selected_Parity_Matrix[:,Data_Erasure_Position]
                    Case_Erasure = np.identity(len(Data_Erasure_Position),dtype=int)
                    for Index_Case in range(len(Case_Erasure)):
                        Case = Case_Erasure[Index_Case]
                        Position_Case = np.where(np.all(Partial_Parity_Matrix == Case, axis=1) == True)[0]
                        if len(Position_Case) >= 1:
                            Fill_Parity_Matrix = Selected_Parity_Matrix[Position_Case[0]]
                            Erasure_Value = Selected_Parity_Value[Position_Case[0]]
                            Position_Data_Partial_H = np.where(Fill_Parity_Matrix == 1)[0]
                            for index in Position_Data_Partial_H:
                                Erasure_Value = Erasure_Value^Data_Error[index]
                            Data_Error[Data_Erasure_Position[Index_Case]] = Erasure_Value
                            Fill_Data_Index = np.append(Fill_Data_Index,Index_Case)
                            Flag_Fill_In = 0    
                    Data_Erasure_Position = np.delete(Data_Erasure_Position,Fill_Data_Index)
                    if Flag_Fill_In:
                        break
            
            if len(Data_Erasure_Position) == 0:
                Decoded_Data = np.array(Data_Error,dtype = np.uint64)
                break

    return Decoded_Data

def Example_Erasure_Channel_Converter(Q,Data,Parity_Value):
    
    #Number of Pulled Parity in each round
    Pull_Buffer = 10
    #Maximum Number of Parity
    limit_parity = len(Parity_Value)
    #Probability 
    Prob_error = 0.05
    Prob_erasure = 0.05

    #Start Simulation
    Position_Buffer = 0
    Number_bit = 32
    Additional_bit = 2

    Data_Error = np.array(Data,dtype = np.uint64)
    Channel_Data = q_ary_error_erasure_channel(Prob_error,Prob_erasure,len(Data))
    Data_Error_Posittion = np.where(Channel_Data==1)[0]
    Data_Erasure_Position = np.where(Channel_Data==2)[0]
    for Error in Data_Error_Posittion:
        Data_Error[Error] = Data_Error[Error]^np.random.randint(low = 1, high = 2**Number_bit, dtype = np.uint64)
    Parity_Value_Error = np.array(Parity_Value,dtype = np.uint64)
    Channel_Parity = q_ary_error_erasure_channel(Prob_error,Prob_erasure,len(Parity_Value))
    Parity_Error_Posittion = np.where(Channel_Parity==1)[0]
    Parity_Erasure_Position = np.where(Channel_Parity==2)[0]
    for Error in Parity_Error_Posittion:
        Parity_Value_Error[Error] = Parity_Value_Error[Error]^np.random.randint(low = 1, high = 2**Number_bit, dtype = np.uint64)
    for Erasure in Parity_Erasure_Position:
        Parity_Value_Error[Erasure] = 0
    
    #Initial Variable
    Selected_Parity_Matrix = np.empty((0, len(Data)), int)
    Selected_Parity_Value = np.array([],dtype=np.uint64)

    while True:

        if limit_parity != -1:
            if Position_Buffer > limit_parity:
                raise RuntimeError('Reach Maximum Number of Parity')

        Number_bit = 32
        for Erasure in Data_Erasure_Position:
            Data_Error[Erasure] = np.random.randint(low = 2**(Number_bit), high = 2**(Number_bit+Additional_bit), dtype = np.uint64)
        Number_bit = Number_bit + Additional_bit
        
        #Pull Parity from Buffer
        Pull_Parity_Matrix = np.array(Q[Position_Buffer:Position_Buffer+Pull_Buffer])    
        Pull_Parity_Value  = np.array(Parity_Value_Error[Position_Buffer:Position_Buffer+Pull_Buffer])
        #Delete Erasure Parity Symbols
        Erasure_Pull_Parity = Parity_Erasure_Position[(Parity_Erasure_Position>=Position_Buffer)&(Parity_Erasure_Position<Position_Buffer + Pull_Buffer)] - Position_Buffer
        Pull_Parity_Matrix = np.delete(Pull_Parity_Matrix,Erasure_Pull_Parity,axis=0)
        Pull_Parity_Value  = np.delete(Pull_Parity_Value,Erasure_Pull_Parity)
        #Shift Buffer Position
        Position_Buffer = Position_Buffer + Pull_Buffer
        #Add New Parity
        Selected_Parity_Matrix = np.append(Selected_Parity_Matrix,Pull_Parity_Matrix,axis=0)
        Selected_Parity_Value = np.append(Selected_Parity_Value,Pull_Parity_Value)
        #Create H
        Identity_Matrix = np.identity(Selected_Parity_Matrix.shape[0],dtype=int)
        Selected_H = np.concatenate((Selected_Parity_Matrix,Identity_Matrix),axis=1)
        #Add Parity Symbols
        Y = np.append(Data_Error,Selected_Parity_Value)
        Y_Binary = MatrixDec_to_MatrixBinary(Y,Number_bit)
        S_Binary = np.floor(np.mod(np.dot(Selected_H,Y_Binary),2)).astype(int)
        #Error Locating Vector
        S_Gauss,S_rank,Index_Rows = Compute_Gauss_Jordan_Reduction(S_Binary)
        Error_Locating_Vector = Compute_Error_Locating_Vector(S_Gauss,Index_Rows,Selected_H)
        Number_Error = np.count_nonzero(Error_Locating_Vector == 0)
        if Number_Error == 0:
            return Data_Error
        #Rank of S = The Number of Error
        if S_rank == Number_Error:
            #Seperate Index
            Index_Error = np.where(Error_Locating_Vector==0)[0]
            Data_Erasure_Position = np.array(Index_Error[Index_Error<len(Data)])
            Index_Error_Parity = Index_Error[Index_Error>=len(Data)] - len(Data)
            if len(Data_Erasure_Position)>0:
                for Erasure in Data_Erasure_Position:
                    Data_Error[Erasure] = 0
            if len(Index_Error_Parity)> 0 :
                Selected_Parity_Matrix = np.delete(Selected_Parity_Matrix,Index_Error_Parity,axis=0)
                Selected_Parity_Value = np.delete(Selected_Parity_Value,Index_Error_Parity)
            #Fill In Erasure Data
            if len(Data_Erasure_Position):
                while(len(Data_Erasure_Position)):
                    Flag_Fill_In = 1
                    Fill_Data_Index = np.array([],dtype=int)
                    Partial_Parity_Matrix = Selected_Parity_Matrix[:,Data_Erasure_Position]
                    Case_Erasure = np.identity(len(Data_Erasure_Position),dtype=int)
                    for Index_Case in range(len(Case_Erasure)):
                        Case = Case_Erasure[Index_Case]
                        Position_Case = np.where(np.all(Partial_Parity_Matrix == Case, axis=1) == True)[0]
                        if len(Position_Case) >= 1:
                            Fill_Parity_Matrix = Selected_Parity_Matrix[Position_Case[0]]
                            Erasure_Value = Selected_Parity_Value[Position_Case[0]]
                            Position_Data_Partial_H = np.where(Fill_Parity_Matrix == 1)[0]
                            for index in Position_Data_Partial_H:
                                Erasure_Value = Erasure_Value^Data_Error[index]
                            Data_Error[Data_Erasure_Position[Index_Case]] = Erasure_Value
                            Fill_Data_Index = np.append(Fill_Data_Index,Index_Case)
                            Flag_Fill_In = 0    
                    Data_Erasure_Position = np.delete(Data_Erasure_Position,Fill_Data_Index)
                    if Flag_Fill_In:
                        break
            
            if len(Data_Erasure_Position) == 0:
                Decoded_Data = np.array(Data_Error,dtype = np.uint64)
                break

    print(Decoded_Data)

