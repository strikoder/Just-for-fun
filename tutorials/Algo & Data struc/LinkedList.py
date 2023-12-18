class Node:
    def __init__(self,data) -> None:
        self.data=data
        self.next=None
    

class LinkedList:
    def insert_at_begin(self,data):
        new_node=Node(data)
        if self.head is None:
            self.head=new_node
            return
        else:
            new_node.next=self.head
            self.head=new_node
    def insert_at_index(self,data,idx):
        new_node=Node(data)
        current_node=self.head
        pos=0
        if pos==idx:
            self.insert_at_begin(data)
        else:
            while(current_node!=None and pos+1!=idx):
                pos+=1
                current_node=current_node.next
            if current_node!=None:
                new_node.next=current_node.next
                current_node.next=new_node
            else:
                print("Idx not found")