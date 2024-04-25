
# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# Basic modules
import numpy as np
from copy import deepcopy

# Import files.
import _mathExpressions

# -------------------------------------------------------------------------- #
# --------------------------- Tree Data Structure -------------------------- #

class treeNodeClass:

    def __init__(self, numericValue, stringValue, parentNode, maxChildren = 0):
        # Structural references in the tree.
        self.maxChildren = maxChildren
        self.parentNode = parentNode
        self.children = []

        # References to the real equation.
        self.numericValue = numericValue    # The numerical value of the node: the feature colummn index or the transformer method.
        self.stringValue = stringValue      # The readable value of the node: feature or transformer name.
        
    def replaceChildNode(self, oldChildNode, newChildNodes):
        for childIndex in range(len(self.children)):
            if self.children[childIndex] == oldChildNode:
                if isinstance(newChildNodes, list):
                    self.children = self.children[:childIndex] + newChildNodes + self.children[childIndex + 1:]
                else:
                    self.children[childIndex] = newChildNodes
    
    def addChildNodes(self, childNodes):
        if isinstance(childNodes, list):
            self.children.extend(childNodes)
        else:
            self.children.append(childNodes)
    
    def removeChildNode(self, childNode):
        try:
            self.children.remove(childNode)
        except:
            exit("Cannot remove a childNode that isnt there.")

    def isChildPresent(self, childNode):
        for node in self.children:
            if node == childNode.stringValue:
                return True

        return False
     
    def prettyPrint(self, prefix=''):
        if self.parentNode != None:
            print(prefix + '|__ ' + self.stringValue)
        else:
            print(self.stringValue)
        for child in self.children:
            if child is not None:
                child.prettyPrint(prefix + '    ')
                                
# -------------------------------------------------------------------------- #
# -------------------------- Recommendation Model -------------------------- #

class expressionTreeModel(_mathExpressions.mathExpressions):        
    
    def __init__(self):   
        # Initialize mathmatical expressions.
        super().__init__()
        
        # Special node values.
        self.rootNodeValue = "Root"

    # ---------------------------------------------------------------------- #
    # ------------------------- Tree Modifications ------------------------- #
    
    def insertChildNode(self, parentNode, childNode):
        # Link the parentNode to the child.
        parentNode.addChildNodes(childNode)
        # Link the childNode to the parent.
        childNode.parentNode = parentNode
        
    def insertParentNode(self, childNode, insertionNode):    
        # Replace the old child node with the inserted node.
        childNode.parentNode.replaceChildNode(childNode, insertionNode)
        
        # Add the old childNode as the insertionNode's child.
        insertionNode.addChildNodes(childNode)
        
        
        # Fix parent pointers
        insertionNode.parentNode = childNode.parentNode
        childNode.parentNode = insertionNode
    
    def removeNode(self, removingNode):
        # Replace the removing node with the children (in order).
        removingNode.parentNode.replaceChildNode(removingNode, removingNode.children)
        
        # Reset the parent of the moved children.
        for childNode in removingNode.children:
            childNode.parentNode = removingNode.parentNode
            
        # Remove all links of the node.
        removingNode.parentNode = None
        removingNode.children = []
        
    def isFirstChild(self, childNode):
        children = childNode.parentNode.children
        assert len(children) != 0
        
        return childNode.parentNode.children[0] == childNode
        
    def changeNodeValues(self, node, numericValue, stringValue):
        node.numericValue = numericValue
        node.stringValue = stringValue
        
    def scaleNode(self, node, scalarValue, byAddition = False):
        parentNode = node.parentNode
        
        # If we have a multiplication node as the parent or currentNode.
        if parentNode.stringValue == "*" or node.stringValue == "*":
            multiplicationNode = parentNode if parentNode.stringValue == "*"  else node
            # The equation can be easily scaled.
            for childNode in multiplicationNode.children:
                if self.isScalarNode(childNode):
                    scalarValue = scalarValue + childNode.numericValue if byAddition else scalarValue*childNode.numericValue
                    self.changeNodeValues(childNode, scalarValue, str(scalarValue))
                    return True
            
        else:
            # Create a multiplication node to add to the tree.
            multiplicationNode = treeNodeClass(self.multiply, stringValue = "*", parentNode = None, maxChildren = np.inf)
            # Add the multiplication node above the node to scale.
            self.insertParentNode(node, multiplicationNode)
            
        # Create a scalar node.
        scalarValue = 1 + scalarValue if byAddition else scalarValue*1
        scalarNode = treeNodeClass(scalarValue, stringValue = str(scalarValue), parentNode = None, maxChildren = 0)
        # Multiply the whole equation by the scalar node.
        self.insertChildNode(multiplicationNode, scalarNode)
        
    def normalizeEquation(self, expressionTree, predictedVals):
        # Verify the integrity of the expression tree.
        assert self.beginsAtRoot(expressionTree), "We should not normalize partial equations."
        
        # Normalize the tree.
        normalizationFactor = 1/np.linalg.norm(predictedVals) # Calculate the length of the expression vector.
        self.scaleNode(expressionTree.children[0], normalizationFactor, byAddition = False)
    
    # ---------------------------------------------------------------------- #
    # --------------------------- Tree Searching --------------------------- #

    def findFirstAncestorNode(self, ancestorValues, currentNode, returnChild = False):
        """ Return ancestor node or start node if None """
        # Base case, we are at the root node.
        if currentNode.stringValue == self.rootNodeValue:
            exit("Searching past the root node.")
        # Base case, we are at the start node.
        elif currentNode.parentNode.stringValue == self.rootNodeValue:
            return currentNode
        
        # Look at the ancestor node.
        if returnChild: ancestorChildNode = currentNode
        ancestorNode = currentNode.parentNode
        # Check if the parent node is the ancestor.
        while ancestorNode.stringValue not in ancestorValues:
            # If you can, walk up the tree.
            if returnChild: ancestorChildNode = ancestorNode
            ancestorNode = ancestorNode.parentNode
            # Use the start node if no acestor.
            if ancestorNode.stringValue == self.rootNodeValue:
                break
        
        if returnChild:
            return ancestorChildNode
        return ancestorNode
    
    def isStubTree(self, expressionTree):
        # If not a tree, it is not a stub.
        if not isinstance(expressionTree, treeNodeClass):
            return False
        
        # Return if the tree is a stub.
        return len(expressionTree.children) == 0 and expressionTree.stringValue == self.rootNodeValue
    
    def beginsAtRoot(self, expressionTree):
        return expressionTree.stringValue == self.rootNodeValue and len(expressionTree.children) == 1
    
    def getDegreesOfFreedom(self, expressionTree):
        # If not a tree, it is not a stub.
        if not isinstance(expressionTree, treeNodeClass):
            return 0, 0
        
        expressionTree = deepcopy(expressionTree)
        # Setup breadth first search.
        queuedNodes = expressionTree.children.copy()
        featureSet = set()
        
        numFeatures = 0
        numTransformers = 0
        # While there are unexplored nodes.
        while len(queuedNodes) != 0:
            # Explore a new child node.
            activeNode = queuedNodes.pop(0)
            
            if self.isFeatureNode(activeNode) and activeNode.stringValue not in featureSet:
                featureSet.add(activeNode.stringValue)
                numFeatures += 1
            elif self.isTransformerNode(activeNode):
                numTransformers += 1
            
            # Store the actionNode's children.
            queuedNodes.extend(activeNode.children.copy())
        
        return numFeatures, numTransformers
    
    # ---------------------------------------------------------------------- #
    # -------------------------- String Searching -------------------------- # 
    
    def findNextSymbolInd(self, currentSymbolInd, stringEquation):
        symbol = stringEquation[currentSymbolInd]
        
        # For each successive symbol in the equation.
        for symbolInd in range(1, len(stringEquation) - currentSymbolInd):
            symbolInd += currentSymbolInd
    
            # If the symbols match, return the symbol index.
            if stringEquation[symbolInd] == symbol:
                return symbolInd
    
        # If no symbol found, return None
        return None
    

    def findClosingSymbolInd(self, currentSymbolInd, counterSymbol, stringEquation):
        symbol = stringEquation[currentSymbolInd]
        
        numExtraSymbols = 0
        # For each successive symbol in the equation.
        for symbolInd in range(currentSymbolInd+1, len(stringEquation)):  
            # If the symbols match, return the symbol index.
            if stringEquation[symbolInd] == counterSymbol:
                if numExtraSymbols == 0:
                    return symbolInd
                numExtraSymbols -= 1
            elif stringEquation[symbolInd] == symbol:
                numExtraSymbols += 1
    
        # If no symbol found, return None
        return None
    
    def findNextMatch(self, currentSymbolInd, stringEquation, matchCriteria):        
        # For each successive symbol in the equation.
        for symbolInd in range(1, len(stringEquation) - currentSymbolInd):
            symbolInd += currentSymbolInd
    
            # If the symbol fullfills the requirements, return index.
            if matchCriteria(stringEquation[symbolInd]):
                return symbolInd - 1
    
        # Return the last index.
        return len(stringEquation) - 1
    
    
    # ---------------------------------------------------------------------- #
    # ------------------------- Tree Data Structure ------------------------ #
    
    def simplifyExpressionTree_org(self, initialExpressionTree):
        """
        Original code to simplify expression tree. (I don't think it works after 
        all of the changes to equationInterface())
        """
        expressionTree = deepcopy(initialExpressionTree)
        # Setup breadth first search.
        queuedNodes = expressionTree.children.copy()

        # While there are unexplored nodes.
        while len(queuedNodes) != 0:
            # Explore a new child node.
            activeNode = queuedNodes.pop(0)
            parentNode = activeNode.parentNode
            # Get the values of the nodes.
            activeValue = activeNode.numericValue
            parentValue = parentNode.numericValue
            
            # If we have chaining addition.
            if activeValue == parentValue and activeValue in [self.add, self.multiply]:
                # Remove the extra add node.
                self.removeNode(activeNode)
            # If we are adding similar nodes.
            if activeValue == self.add:
                childrenSet = []
                # Search the children for similar nodes.
                for childNode in activeNode.children:                    
                    if childNode not in childrenSet:
                        childrenSet.append(childNode)
                    else:
                        # For each previously node seen
                        for node in childrenSet:
                            if node == childNode:
                                self.removeNode(childNode)
                                self.scaleNode(node, 1, byAddition = True)
            
            # Store the actionNode's children.
            queuedNodes.extend(activeNode.children.copy())
        
        return expressionTree
    
    def simplifyExpressionTree(self, initialExpressionTree):
        # TODO
        return initialExpressionTree
    
    # Function to validate if the given expression tree is valid.
    def isValidExpressionTree(self, expressionTree):
        
        # If we are at the root node
        if expressionTree.stringValue == self.rootNodeValue:
            # We an only have one child of root node. Return false if root does not have exactly 1 child
            if len(expressionTree.children) != 1:
                return False
            else:
                # Otherwise, recursively validate that child
                return self.isValidExpressionTree(expressionTree.children[0])
        
        # If we are at a scalar node
        elif self.isScalarNode(expressionTree):
            # This must be a leaf node. Return false if the node has any children
            if len(expressionTree.children) != 0:
                return False
        
        # If we are at a transformer node (function or operator)
        elif self.isTransformerNode(expressionTree):
            # The number of children cannot exceed the specified maximum limit of children for the given type of node. Return false if this is the case.
            if len(expressionTree.children) > expressionTree.maxChildren:
                return False
            else:
                # Otherwise, recursively validate each child of the transformer node
                for child in expressionTree.children:
                    if not self.isValidExpressionTree(child):
                        return False
        return True
    
    def expressionTreeInterface(self, expressionTree, inputData = []):
        """
        An interface to convert the tree data structure to a string equation
        or a numerical value.

        Solve the equation
        -------------------
        If inputData is given, we will treat any integer numericValue as a column 
        index of the input data. This column should represent the features. We
        will then plug and chug the features through the equation to get a final result.
        The dimensions of the output is (# of data points x 1)
                                         
        Get the equation
        -------------------
        If inputData is not given, we have no references to the real data. Hence,
        we will output the string equation without replacing the variables.                      

        """
        
        # Base case: we are at the root node.
        if expressionTree.stringValue == self.rootNodeValue:
            assert len(expressionTree.children) == 1, expressionTree.prettyPrint()
            stringEquation = self.expressionTreeInterface(expressionTree.children[0], inputData)
            if stringEquation[0] == "(":
                return stringEquation[1:-1]
            return stringEquation
        
        # If the node is a scalar.
        elif self.isScalarNode(expressionTree):
            # If we are solving a discrete equation.
            if len(inputData) != 0:
                return expressionTree.numericValue
            else:
                return expressionTree.stringValue
    
        # If the node is a feature.
        elif self.isFeatureNode(expressionTree):
            # If we are solving a discrete equation.
            if len(inputData) != 0:
                return inputData[:, expressionTree.numericValue]
            else:
                return expressionTree.stringValue
        
        # If the node is a transformer.
        elif self.isTransformerNode(expressionTree):
            # If we are solving a discrete equation.
            if len(inputData) != 0:
                # Apply the tranformer to the variables.
                if len(expressionTree.children) == 1: 
                    return expressionTree.numericValue(self.expressionTreeInterface(expressionTree.children[0], inputData))
                return expressionTree.numericValue([self.expressionTreeInterface(subtree, inputData) for subtree in expressionTree.children])
            else:
                # Seperate the transformer from the variables
                transformer = expressionTree.stringValue
                variableList = [self.expressionTreeInterface(subtree, inputData) for subtree in expressionTree.children]
                # print(variableList, transformer)
                # Assert that we have a transformer.
                assert transformer in self.mapString_toTransformer, "Why is the First Element NOT an transformer. Equation: " + transformer
                # If the transformer is a function.
                if self.isFunctionNode(expressionTree) and not transformer in ["-", "/", "^"]:
                    # Rewrite the equation in this format: sin(x)
                    equationString = f"{transformer}({variableList[0]})"
                elif transformer in ["-", "/"]:
                    if len(variableList) > 1:
                        equationString = f"{transformer}({expressionTree.numericValue([self.expressionTreeInterface(subtree, inputData) for subtree in expressionTree.children])})"
                    else:
                        equationString = f"{transformer}{variableList[0]}"
                #     expressionTreeInterface
                # Else the transformer is an operator.
                else:
                    # Rewrite the equation in this format: x + y
                    operatorString = transformer
                    equationString = variableList[0]
                    
                    for expression in variableList[1:]:
                        if not ((expression[0] == "-" and operatorString in ["+"]) or (expression[0] == "/" and operatorString in ["*"])):
                            equationString += operatorString
                        equationString += expression
                    

                # Return formatted string with operator and variables.
                if len(variableList) > 1:
                    return f"({equationString})"
                return equationString                    
        else:
            assert False, "Unknown numerical type " + str(type(expressionTree.numericValue)) + " with value: " + str(expressionTree.numericValue)
    
    # ---------------------------------------------------------------------- #
    # ----------------------- Equation Data Structure ---------------------- #      
    
    def isValidEquation(self):
    
        pass
    
    def equationInterface(self, stringEquation, featureNames): 
        """ Return  the Equation """
        # Initialize root node to create the tree structure.
        rootNode = treeNodeClass(numericValue = None, stringValue = self.rootNodeValue, parentNode = None, maxChildren = np.inf)
        activeNode = rootNode
        
        symbolInd = 0
        # For each symbol in the equation.
        while symbolInd < len(stringEquation):
            symbol = stringEquation[symbolInd] 
            
            # --------------------------------------------------------------- #
            
            # If we have a feature
            if symbol == "'":
                # Find the end of the feature name. If no end quote, there is an error.
                closingSymbolInd = self.findNextSymbolInd(symbolInd, stringEquation)
                if closingSymbolInd == None:
                    return None
                # Get the feature information.
                featureName = stringEquation[symbolInd:closingSymbolInd+1] # Leaving on the wrapping quotes "'"
                featureInd = np.where(featureNames == featureName[1:-1])[0]
                if len(featureInd) != 1: exit("Unknown feature:", featureName)
                # Don't reannalyze the feature.                
                symbolInd = closingSymbolInd + 1
                
                # Add the feature to the tree.
                featureNode = treeNodeClass(int(featureInd), stringValue = featureName, parentNode = activeNode, maxChildren = 0)
                activeNode.addChildNodes(featureNode)
                # The featureNode is now the active node.
                activeNode = featureNode 
                
            # --------------------------------------------------------------- #
            
            # If we have a function.
            elif self.isFunctionChar(symbol):
                # Get the function name.
                closingSymbolInd = symbolInd
                if symbol not in ["-", "/"]:
                    closingSymbolInd = self.findNextMatch(symbolInd, stringEquation, matchCriteria = lambda x: not x.isalpha()) 
                functionName = stringEquation[symbolInd:closingSymbolInd+1]
                # Don't reannalyze the function name.                
                symbolInd = closingSymbolInd + 1
                
                # Add the function to the tree.
                functionMethod = self.mapString_toTransformer[functionName]
                if not self.isStubTree(activeNode) and symbol in ["-", "/"]:
                    if functionName == "-":
                        symbol = "+"
                        symbols = ["+"]
                    elif "/":
                        symbol = "*"
                        symbols = ["+", "*", "-"]
                    else:
                        assert False
                        
                    activeNode = self.findFirstAncestorNode(symbols, activeNode, returnChild = True)
                    # If we are chaining.
                    if activeNode.parentNode.stringValue == symbol:
                        # The active node is the old operator.
                        activeNode = activeNode.parentNode
                        
                    # If we have no add/multiply operator present.
                    if activeNode.stringValue != symbol:
                        # Create an add/multiply node.
                        operatorMethod = self.mapString_toTransformer[symbol]
                        operatorNode = treeNodeClass(operatorMethod, stringValue = symbol, parentNode = None, maxChildren = np.inf)
                        # Insert the node below the root.
                        # activeNode.prettyPrint()
                        # assert activeNode.parentNode.stringValue == self.rootNodeValue
                        self.insertParentNode(activeNode, operatorNode)
                        activeNode = operatorNode
                
                # Create and add the function node.
                functionNode = treeNodeClass(functionMethod, stringValue = functionName, parentNode = activeNode, maxChildren = 1)
                activeNode.addChildNodes(functionNode)
                # The functionNode is now the active node.
                activeNode = functionNode
                
            # --------------------------------------------------------------- #

            # If we have a scalar
            elif symbol.isnumeric():
                # Find the number
                closingSymbolInd = self.findNextMatch(symbolInd, stringEquation, matchCriteria = lambda x: not (x.isnumeric() or x in ",."))
                scalar = stringEquation[symbolInd:closingSymbolInd+1]
                # Don't reannalyze the feature.                
                symbolInd = closingSymbolInd + 1
                
                # Add the feature to the tree.
                featureNode = treeNodeClass(float(scalar), stringValue = scalar, parentNode = activeNode, maxChildren = 0)
                activeNode.addChildNodes(featureNode)
                # The featureNode is now the active node.
                activeNode = featureNode 
                
            # --------------------------------------------------------------- #
            
            # If we have an operator.
            elif symbol in self.mapString_toTransformer.keys():
                allowChaining = False
                                        
                # Thats addition.
                if symbol == "+":
                    maxChildren = np.inf
                    # Addition has the lowest priority. Therefore, the active node is most recent appearance of the add operator (or start node if None).
                    activeNode = self.findFirstAncestorNode(["+"], activeNode, returnChild = True) # Returns either an add operator or start node.
                    # If we are chaining addition.
                    if activeNode.parentNode.stringValue == "+":
                        # The active node is the old mult operator.
                        activeNode = activeNode.parentNode

                # Thats multiplication.
                elif symbol == "*":
                    maxChildren = np.inf
                    # Multiplication has high priority than addition. Therefore, the active node is most recent appearance of the add or mult operator (or start node if None).
                    activeNode = self.findFirstAncestorNode(["*", "+", "-"], activeNode, returnChild = True) 
                    
                    # If we are chaining multiplication.
                    if activeNode.parentNode.stringValue == "*":
                        # The active node is the old mult operator.
                        activeNode = activeNode.parentNode
                    
                # Thats raising.
                elif symbol == "^":
                    # print("HERE", activeNode.stringValue, symbol)
                    maxChildren = 2
                    allowChaining = True
                    # if activeNode.stringValue == symbol:
                    #     activeNode = activeNode.children[-1]
                    
                # ----------------------------------------------------------- #
                                    
                # If not chaining operators.
                if activeNode.stringValue != symbol or allowChaining:
                    # Create the operator node.
                    operatorMethod = self.mapString_toTransformer[symbol]
                    operatorNode = treeNodeClass(operatorMethod, stringValue = symbol, parentNode = None, maxChildren = maxChildren)
                                        
                    # Insert the operator between the active node and the parent.
                    self.insertParentNode(activeNode, operatorNode)
                    # The active node is now the operator.
                    activeNode = operatorNode
                    
                # ----------------------------------------------------------- #
                    
                # Iterate the symbol index
                symbolInd += 1
                
            # --------------------------------------------------------------- #
            
            # If we have a parenthesis.
            elif symbol in "([":
                # Find the associated closing parenthesis.
                counterSymbol = ")" if symbol == "(" else "]"
                closingSymbolInd = self.findClosingSymbolInd(symbolInd, counterSymbol, stringEquation)
                sectionEquation = stringEquation[symbolInd+1:closingSymbolInd] # Removing parentheses.
                                
                # Recursively create this section's tree.
                sectionTree = self.equationInterface(sectionEquation, featureNames)

                # Snip the root node
                rootChildren = sectionTree.children
                assert len(rootChildren) == 1
                sectionTree = rootChildren[0]
                sectionTree.parentNode = None
                # Add the section to the tree
                self.insertChildNode(activeNode, sectionTree)
                if activeNode.stringValue == self.rootNodeValue:
                    activeNode = sectionTree
                # activeNode = sectionTree
                
                # Don't reannalyze the section.                
                symbolInd = closingSymbolInd + 1
            
            elif symbol not in " ":
                exit("Invalid symbol:", symbol)
            else:
                symbolInd += 1
                
        return rootNode
    
# ---------------------------------------------------------------------------#

if __name__ == "__main__":    
    numPoints = 10000
    # Specify input features.
    x = np.random.uniform(1, 4, numPoints)
    y = np.random.uniform(2, 5, numPoints)
    z = np.random.uniform(3, 6, numPoints)
    a = np.random.uniform(4, 7, numPoints)
    b = np.random.uniform(5, 8, numPoints)
    c = np.random.uniform(6, 9, numPoints)
    
    # Compile feature data.
    featureData = np.array([x, y, z, a, b, c]).T
    featureNames_Tree = np.array(['x', 'y', 'z', 'a', 'b', 'c'])
    featureDict = dict(zip(featureNames_Tree, featureData))

    # Instantiate classes.
    treeClass = expressionTreeModel()

    stringEquation = "sin('x'*'y'-'z') + 2.01^'x'/ 3 *'y' - 'a'*'x'^('y'/'z')"
    expressionTree = treeClass.equationInterface(stringEquation, featureNames_Tree)
    
    # Print out the results.
    expressionTree.prettyPrint()
    print(treeClass.expressionTreeInterface(expressionTree, []))
    print(treeClass.expressionTreeInterface(expressionTree, featureData))
    
    # Assert the validity of the class.
    # assert all(treeClass.expressionTreeInterface(expressionTree, featureData) == np.sin(x*y - z) + 2.01**x / 3 * y - a*x**(y/z))

    
    
