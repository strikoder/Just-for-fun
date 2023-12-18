# I will pin the head and add through the tail
class Node():
    def __init__(self,data) -> None:
       self.data=data
       self.next=None


class Queue():
    def __init__(self) -> None:
        self.head=None
        self.tail=None
    def enqueue(self,data):
        new_node=Node(data)
        if self.head is None:
            self.head=self.tail=new_node
            return None
        self.tail.next=new_node
        self.tail=new_node
    def dequeue(self):
        if self.head is None:
            return None
        self.temp = self.head
        self.head=self.temp.next
        if self.head is None:
            self.tail = None
        return self.temp.data
    


qu=Queue()
qu.enqueue(5)
qu.enqueue(4)
res=qu.dequeue()
print(res)
res=qu.dequeue()
print(res)