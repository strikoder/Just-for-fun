import math
# I will jump (sqrt of N) until I reach the first True, 
# then will do linear search of sqrt of N to reach the first true

def TCB(breaks,low,high):
    jumpAmount=math.floor(math.sqrt(len(breaks)))
    for i in range(jumpAmount,len(arr),jumpAmount):
        if breaks[i]:
            idx=i
            break
    if idx==-1:
        return -1
    
    # Linear search in the previous block
    start = max(idx - jumpAmount, 0) #ensuring not start from minus

    for i in range(start,min(idx+1,len(breaks))):
        if breaks[i]:
            return i
    
    return -1
        

    
    

arr=[False,False,False,False,False,False,False,False,True,True,True,True,True,True]
print(TCB(arr,0,len(arr)))