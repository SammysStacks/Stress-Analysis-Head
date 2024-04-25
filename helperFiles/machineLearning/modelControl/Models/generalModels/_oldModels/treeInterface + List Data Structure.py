
# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# Basic modules
import numpy as np

# Import Files
import equationGenerator    # The equation generator model.

# -------------------------------------------------------------------------- #
# --------------------------- Tree Data Structure -------------------------- #

class treeNode:

    def __init__(self, numericValue, stringValue, parentNode, children = []):
        # Structural references in the tree.
        self.parentNode = parentNode
        self.children = children
        
        # References to the real equation.
        self.numericValue = numericValue    # The numerical value of the node: the feature colummn index or the operator/transformer function.
        self.stringValue = stringValue      # The readable value of the node: feature name, operator, or transformer name.
        
    def addChildNode(self, childNode):
        self.children.append(childNode)
    
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

class equationTreeInterface:        
    
    def __init__(self, featureData, featureNames, operatorMap, functionMap): 
        # Equation information
        self.featureData = featureData
        self.featureNames = featureNames
        
        # Symbolic expression information
        self.operatorMap = operatorMap
        self.functionMap = functionMap
        
        self.rootNodeValue = "Root"
        
        
    # ---------------------------------------------------------------------- #
    # --------------------------- Tree Searching --------------------------- #  
    
    def insertNodeBetween(self, childNode, insertionNode, parentNode):
        # Replace the old child node with the inserted node.
        parentNode.addChildNode(insertionNode)
        parentNode.removeChildNode(childNode)
        
        # Add the old childNode as the insertionNode's child.
        insertionNode.addChildNode(childNode)
        
        # Fix parent pointers
        childNode.parentNode = insertionNode
        insertionNode.parentNode = parentNode
    
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
        for symbolInd in range(1, len(stringEquation) - currentSymbolInd):
            symbolInd += currentSymbolInd
    
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
    
    def simplifyTree(self, equation):
        return equation
    
    def isValidTree(self):
        pass
    
    def treeInterface(self, treeEquation, inputData = []):
        
        # Base case: we are at the root node.
        if treeEquation.stringValue == self.rootNodeValue:
            assert len(treeEquation.children) == 1
            return self.treeInterface(treeEquation.children[0], inputData)
        
        # If the node is a scalar.
        elif isinstance((treeEquation.numericValue), float):
            # If we are solving a discrete equation.
            if len(inputData) != 0:
                return treeEquation.numericValue
            else:
                return treeEquation.stringValue
    
        # If the node is a feature.
        elif isinstance((treeEquation.numericValue), int):
            # If we are solving a discrete equation.
            if len(inputData) != 0:
                return inputData[:, treeEquation.numericValue]
            else:
                return treeEquation.stringValue
        
        # If the node is an operator or transformer.
        elif isinstance((treeEquation.numericValue), type(lambda x: x)):
            # If we are solving a discrete equation.
            if len(inputData) != 0:
                if len(treeEquation.children) == 1: 
                    return treeEquation.numericValue(self.treeInterface(treeEquation.children[0], inputData))
                # print([self.treeInterface(subtree, inputData) for subtree in treeEquation.children])
                return treeEquation.numericValue([self.treeInterface(subtree, inputData) for subtree in treeEquation.children])
            else:
                expression = treeEquation.stringValue
                operands = [self.treeInterface(subtree, inputData) for subtree in treeEquation.children]
                                
                # If the operator is a function.
                if expression in self.functionMap:
                    # Rewrite the equation in this format: sin(x)
                    functionString = expression
                    arg_str = "".join(operands)
                    equationString = f"{functionString}({arg_str})"
                    
                # If the operator is an operator.
                elif expression in self.operatorMap:
                    # Rewrite the equation in this format: x + y
                    operatorString = expression
                    equationString = operatorString.join(operands)
                    # If using paranthesis, add closing term: x^(y*z)
                    if operatorString[-1] == "(":
                        equationString += ")"
                    if treeEquation.parentNode.stringValue in ["^", "*"] and not expression == self.operatorMap["^"]:
                        equationString = "(" + equationString + ")"
                else:
                    exit("Why is the First Element NOT an Expression. Equation: " + expression)
                    
                # Return formatted string with operator and operands.
                return equationString                    
        else:
            exit(treeEquation.numericValue)
    
    # ---------------------------------------------------------------------- #
    # ----------------------- Equation Data Structure ---------------------- #      
    
    def equationInterface(self, stringEquation): 
        """ Solves or Stringifies the Equation """
        # Initialize root node to create the tree structure.
        rootNode = treeNode(numericValue = None, stringValue = self.rootNodeValue, parentNode = None, children = [])
        activeNode = rootNode
        
        symbolInd = 0
        # For each symbol in the equation.
        while symbolInd < len(stringEquation):
            symbol = stringEquation[symbolInd] 
            
            # If we have a quote
            if symbol == "'":
                # Find the end of the feature name. If no end quote, there is an error.
                closingSymbolInd = self.findNextSymbolInd(symbolInd, stringEquation)
                if closingSymbolInd == None:
                    return None
                # Get the feature information.
                featureName = stringEquation[symbolInd:closingSymbolInd+1] # Leaving on the wrapping quotes "'"
                featureInd = np.where(self.featureNames == featureName[1:-1])[0]
                if len(featureInd) != 1: exit("Unknown feature:", featureName)
                # Don't reannalyze the feature.                
                symbolInd = closingSymbolInd + 1
                
                # Add the feature to the tree.
                featureNode = treeNode(int(featureInd), stringValue = featureName, parentNode = activeNode, children = [])
                activeNode.addChildNode(featureNode)
                # The featureNode is now the active node.
                activeNode = featureNode 
                
            # If we have a transformer.
            elif symbol.isalpha():
                # Get the function name.
                closingSymbolInd = self.findNextMatch(symbolInd, stringEquation, matchCriteria = lambda x: not x.isalpha()) 
                functionName = stringEquation[symbolInd:closingSymbolInd+1]
                # Don't reannalyze the function name.                
                symbolInd = closingSymbolInd + 1
                
                # Add the function to the tree.
                functionMethod = self.functionMap[functionName]
                functionNode = treeNode(functionMethod, stringValue = functionName, parentNode = activeNode, children = [])
                activeNode.addChildNode(functionNode)
                # The functionNode is now the active node.
                activeNode = functionNode
                
            # If we have a number
            elif symbol.isnumeric():
                # Find the number
                closingSymbolInd = self.findNextMatch(symbolInd, stringEquation, matchCriteria = lambda x: not (x.isnumeric() or x in ",."))
                scalar = stringEquation[symbolInd:closingSymbolInd+1]
                # Don't reannalyze the feature.                
                symbolInd = closingSymbolInd + 1
                
                # Add the feature to the tree.
                featureNode = treeNode(float(scalar), stringValue = scalar, parentNode = activeNode, children = [])
                activeNode.addChildNode(featureNode)
                # The featureNode is now the active node.
                activeNode = featureNode 
            
            # If we have an operator.
            elif symbol in self.operatorMap.keys():
                # Thats addition.
                if symbol == "+":
                    # Addition has the lowest priority. Therefore, the active node is most recent appearance of the add operator (or start node if None).
                    activeNode = self.findFirstAncestorNode(["+"], activeNode) # Returns either an add operator or start node.
                    
                # Thats multiplication.
                elif symbol == "*":
                    # Multiplication has high priority than addition. Therefore, the active node is most recent appearance of the add or mult operator (or start node if None).
                    activeNode = self.findFirstAncestorNode(["*", "+"], activeNode, returnChild = True) 
                    
                    
                    # If we are chaining multiplication.
                    if activeNode.parentNode.stringValue == "*":
                        # The active node is the old mult operator.
                        activeNode = activeNode.parentNode
                    
                # If the operator is raising.
                elif symbol == "^":
                    # Check whether we are inside a raising operator WITHOUT multiplication.
                    ancestorNode = self.findFirstAncestorNode(["^", "*"], activeNode) # Returns either an mult/raise operator or start node.
                    symbol = "*" if ancestorNode.stringValue == "^" else "^"

                # If not chaining operators.
                if activeNode.stringValue != symbol:
                    # Create the operator node.
                    operatorMethod = self.operatorMap[symbol]
                    operatorNode = treeNode(operatorMethod, stringValue = symbol, parentNode = None, children = [])
                    
                    # Insert the operator between the active node and the parent.
                    self.insertNodeBetween(activeNode, operatorNode, activeNode.parentNode)
                    # The active node is now the operator.
                    activeNode = operatorNode
                
                # Iterate the symbol index
                symbolInd += 1
            
            # If we have a parenthesis.
            elif symbol in "([":
                # Find the associated closing parenthesis.
                counterSymbol = ")" if symbol == "(" else "]"
                closingSymbolInd = self.findClosingSymbolInd(symbolInd, counterSymbol, stringEquation)
                sectionEquation = stringEquation[symbolInd+1:closingSymbolInd] # Removing parentheses.
                                
                # Recursively create this section's tree.
                sectionTree = self.equationInterface(sectionEquation)
                # Snip the root node
                rootChildren = sectionTree.children
                assert len(rootChildren) == 1
                sectionTree = rootChildren[0]
                # Add the section to the tree
                activeNode.addChildNode(sectionTree)
                sectionTree.parentNode = activeNode
                
                # Don't reannalyze the section.                
                symbolInd = closingSymbolInd + 1
            
            elif symbol not in " ":
                exit("Invalid symbol:", symbol)
            else:
                symbolInd += 1
                
        return rootNode
            
    
    # def equationInterface(self, equation, inputData, solveEquation = True, nested = False): 
    #     """ Solves or Stringifies the Equation """
        
    #     # Special case: we have a single input: 'x' or 0.
    #     if not isinstance(equation, list):
    #         # An integer would be the column index of the feature.
    #         if isinstance(equation, int):
    #             # Replace the index with the feature name or feature input column.
    #             return inputData[:, equation].T if solveEquation else self.finalFeatureNames[equation]

    #         # Else, we have an expression.
    #         return str(equation)
        
    #     # Special case: we hav no equation.
    #     elif len(equation) == 0:
    #         # Return an empty result.
    #         return None if solveEquation else ""
        
    #     # Base case: we have a list of numbers.
    #     elif isinstance(equation[0], (int, float)):
    #         # Assert the integrity of the equation.
    #         assert all([isinstance(operand, (int, float)) for operand in equation]), "All operands must be integers IF the first operand is NOT an expression. Equation: " + str(equation)
            
    #         # Replace the indices with the feature names.
    #         returnValue = inputData[:, equation].T if solveEquation else self.finalFeatureNames[equation]
    #         # Special case: we only have one item
    #         if len(equation) == 1:
    #             return returnValue[0]
    #         return returnValue
    
    #     # Base case: we have a list of lists.
    #     elif isinstance(equation[0], list):
    #         # Evaluate each entry in the list.    
    #         return [self.equationInterface(operand, inputData, solveEquation, nested = True) for operand in equation]
        
    #     # Recursive case: we have an operator
    #     elif isinstance(equation[0], type(lambda x: x)):
    #         # Assert the integrity of the input equation
    #         assert len(equation) == 2 # At this point, we have an expression and its inputs
                        
    #         # Seperate the expression from the inputs.
    #         expression, inputs = equation
    #         # Convert the inputs into a stringified equation.
    #         operands = self.equationInterface(inputs, inputData, solveEquation, nested = False)
            
    #         # if solving the equation, plug and chug.
    #         if solveEquation:
    #             return expression(operands)
            
    #         # Else, we have to form the string equation.
    #         else:
    #             # If the operator is a function.
    #             if expression in self.function_hash:
    #                 # Rewrite the equation in this format: sin(x)
    #                 functionString = self.function_hash[expression]
    #                 arg_str = "".join(operands)
    #                 equationString = f"{functionString}({arg_str})"
                    
    #             # If the operator is an operator.
    #             elif expression in self.operator_hash:
    #                 # Rewrite the equation in this format: x + y
    #                 operatorString = self.operator_hash[expression]
    #                 equationString = operatorString.join(operands)
    #                 # If using paranthesis, add closing term: x^(y*z)
    #                 if operatorString[-1] == "(":
    #                     equationString += ")"
    #                 if nested and not expression == self.raiseXY:
    #                     equationString = "(" + equationString + ")"
    #             else:
    #                 print(expression)
    #                 exit("Why is the First Element NOT an Expression. Equation: " + str(equation))
        
    #             # Return formatted string with operator and operands.
    #             return equationString
                    
    #     # Unknown case
    #     else:
    #         exit("IDK What Happened. Equation: " + str(equation))
    
    # ---------------------------------------------------------------------- #
    # --------------------------- Update Equation -------------------------- #    
    
    
# ---------------------------------------------------------------------------#

if __name__ == "__main__":
    # Set model paramters
    modelPath = "./EGM.pkl"
    modelType = "EGM"
    allFeatureNames = [""]
    overwriteModel = True
    
    numPoints = 100000
    # Specify input features.
    x = np.random.uniform(0, 100, numPoints)
    y = np.random.uniform(2, 10, numPoints)
    z = np.random.uniform(2, 10, numPoints)
    a = np.random.uniform(2, 10, numPoints)
    b = np.random.uniform(2, 10, numPoints)
    c = np.random.uniform(2, 10, numPoints)
    
    # Compile feature data.
    featureData = np.array([x, y, z, a, b, c]).T
    featureNames_Tree = np.array(['x', 'y', 'z', 'a', 'b', 'c'])

    # Instantiate classes.
    modelClass = equationGenerator.equationGenerator(modelPath, modelType, allFeatureNames, overwriteModel)
    treeClass = equationTreeInterface(featureData, featureNames_Tree, modelClass.operatorMap, modelClass.functionMap)

    stringEquation = "sin('x'*'y'*'z') + 2.01^'x'*3*'y' + 'a'*'x'^('y'*'z')"
    treeEquation = treeClass.equationInterface(stringEquation)
    
    treeEquation.prettyPrint()
    
    print(treeClass.treeInterface(treeEquation, []))
    print(treeClass.treeInterface(treeEquation, featureData))
    2.01**x*3*y + np.sin(x*y*z) + a*x**(y*z)

    
    print(all(treeClass.treeInterface(treeEquation, featureData) == 2.01**x*3*y + np.sin(x*y*z) + a*x**(y*z)))
