# I will add tail and leave it on the last one, then add the heads and start adding through prev
class Node():
    def __init__(self,data) -> None:
       self.data=data
       self.prev=None

class Stack():
    def __init__(self) -> None:
        self.tail=None
    def push(self,data):
        new_node=Node(data)
        if self is None:
            self.tail=new_node
            return

        new_node.prev=self.tail
        self.tail=new_node 

    def pull(self):
        if self.tail is None:
            return None
        temp = self.tail
        self.tail=temp.prev
        return temp.data 

st=Stack()
st.push(4)
st.push(5)
res=st.pull()
print(res)
res=st.pull()
print(res)