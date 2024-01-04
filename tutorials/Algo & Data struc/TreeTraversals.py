#Tree types: pre-order(7,23,5,4,3,18,21), in-order(5,23,4,7,3,18,21),post-order( 5,4,23,18,21,3,7)
#         7
#      /     \
#    23        3
#  /   \     /   \
# 5     4  18     21

class TreeNode:
    def __init__(self,value):
        self.value=value
        self.left=None
        self.right=None

def insert_level_order(arr, curr, len):
    """
    This function is build so that if the arr was identfied line by line from left to right
    """

    if curr<len:
        node = TreeNode(arr[curr])
        node.left=insert_level_order(arr, 2*curr+1, len)
        node.right=insert_level_order(arr, 2*curr+2, len)
        return node 
    return None


def preorder_traversal(root):
    if root is None:
        return 
    print(root.value, end=" ")
    preorder_traversal(root.left)
    preorder_traversal(root.right)

def inorder_traversal(root):
    if root is None:
        return
    inorder_traversal(root.left)
    print(root.value, end=" ")
    inorder_traversal(root.right)

def postorder_traversal(root):
    if root is None:
        return
    postorder_traversal(root.left)
    postorder_traversal(root.right)
    print(root.value,end=" ")




# Construct the tree
tree = [7, 23, 3, 5, 4, 18, 21]
root = insert_level_order(tree, 0, len(tree))

# Perform and print the traversals
print("Pre-order Traversal:")
preorder_traversal(root)
print("\nIn-order Traversal:")
inorder_traversal(root)
print("\nPost-order Traversal:")
postorder_traversal(root)





