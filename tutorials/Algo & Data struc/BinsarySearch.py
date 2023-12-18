import math
def binary_search(arr,low,high,n) ->bool:
    
    while high>low:
        mid=math.floor(low+(high-low)/2)
        if arr[mid]==n:
            return True
        elif arr[mid]>n:
            high = mid
        else:
            low=mid+1
    return False
        





arr=[2,10,5,3,9]
arr.sort()
print(arr)

print(binary_search(arr,0,len(arr),1))