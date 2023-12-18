def linear_search(arr:list, num) ->bool:
    for i in range(len(arr)):
        if arr[i]==num:
            return True
    
    return False


arr=[1,2,3]
print(linear_search(arr,3))