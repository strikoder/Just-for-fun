def bubble_sort(arr:list)->list:
    
    for i in range(len(arr)):
        for j in range(len(arr)-1-i):
            if arr[j]>arr[j+1]:
                arr[j], arr[j+1]=arr[j+1],arr[j]
        
    return arr



arr=[1,4,2,5,3,18,9]
print(bubble_sort(arr=arr))