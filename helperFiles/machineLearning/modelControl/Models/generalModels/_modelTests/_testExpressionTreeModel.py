
# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# Basic modules
import os
import sys
import numpy as np

# Import files
sys.path.append(os.path.dirname(__file__) + "/../Model Helpers/Expression Tree/")
import expressionTreeModel # Global model class
import _generateDataset

# Libary for simplification of string equations
import sympy
from sympy.parsing.sympy_parser import parse_expr

# -------------------------------------------------------------------------- #
# -------------------------- Recommendation Model -------------------------- #

class testExpressionTreeModel:
    
    def __init__(self):
        # Set random seed
        np.random.seed(1234)
        
        numPoints = 100
        # Specify input features.
        x = np.random.uniform(0, 1, numPoints)
        y = np.random.uniform(0, 1, numPoints)
        z = np.random.uniform(0, 1, numPoints)
        a = np.random.uniform(0, 1, numPoints)
        b = np.random.uniform(0, 1, numPoints)
        c = np.random.uniform(0, 1, numPoints)
        
        # Compile feature data.
        self.featureData = np.array([x, y, z, a, b, c]).T
        self.featureNames_Tree = np.array(['x', 'y', 'z', 'a', 'b', 'c'])
        
        # Initialize classes.
        self.datasetGenerator = _generateDataset.datasetGenerator()
        
        # Reset the model.
        self.resetModel()
            
    def resetModel(self):
        # Set random seed
        np.random.seed(1234)
        
        # Instantiate model class.
        self.expressionTreeModel = expressionTreeModel.expressionTreeModel()
        
    def assertTreeStructure(self, expressionTree, nodeOrder):
        # Setup breadth first search.
        queuedNodes = expressionTree.children.copy()
        
        nodeOrderInd = 0
        # While there are unexplored nodes.
        while len(queuedNodes) != 0:
            # Explore a new child node.
            activeNode = queuedNodes.pop(0)
            
            assert activeNode.stringValue == nodeOrder[nodeOrderInd], f"Bad node at index {nodeOrderInd}. Should be {activeNode.stringValue}, not {nodeOrder[nodeOrderInd]}"
            
            nodeOrderInd += 1
            # Store the actionNode's children.
            queuedNodes.extend(activeNode.children.copy())
    
    # ---------------------------------------------------------------------- #
    
    """
    Helper functions to check original input string equation is equal to the string 
    equation created from the expression tree, using sympy library.
    """
    
    def simplify_equation(self, equation_string):
        equation_string = equation_string.replace("'", "").replace("^", "**").replace(" ", "")
        equation_string = parse_expr(equation_string)
        simplified_equation = sympy.simplify(equation_string)
        return simplified_equation

    def check_equation_equality(self, equation1_string, equation2_string):
        simplified_equation1 = self.simplify_equation(equation1_string)
        simplified_equation2 = self.simplify_equation(equation2_string)
    
        return simplified_equation1 == simplified_equation2
    
    # ---------------------------------------------------------------------- #
    
    def assertEquation2Tree(self, stringEquation, nodeOrder):
        print("\n-----------------------------------------\n", stringEquation)
        # Build the equation tree from the string equation input.
        expressionTree = self.expressionTreeModel.equationInterface(stringEquation, self.featureNames_Tree)
        assert self.expressionTreeModel.isValidExpressionTree(expressionTree), "Invalid Tree" # Assert tree is valid
        
        expressionTree.prettyPrint() # Print generated tree
        
        if len(nodeOrder) != 0:
            self.assertTreeStructure(expressionTree, nodeOrder) # Assert that tree is correct.
        
        # Build the string equation from the expression tree
        stringOutput = self.expressionTreeModel.expressionTreeInterface(expressionTree)
        # Assert this string output matches the original input we used to build the tree
        assert self.check_equation_equality(stringOutput, stringEquation), f"Expression Tree to String, {stringOutput}, is not original String Equation {stringEquation}."
        
        # TODO: Stricter tree simplification assertions/tests.
        # Right now, we make sure that if we give the variables real values, the tree produces the same output before and after the simplification
        numExpressionTree = self.expressionTreeModel.expressionTreeInterface(expressionTree, self.featureData) # Generate output values from expression tree using featureData array
        simplifiedExpressionTree = self.expressionTreeModel.simplifyExpressionTree(expressionTree) # Simplify Tree
        assert self.expressionTreeModel.isValidExpressionTree(simplifiedExpressionTree) # Assert tree is valid

        numSimplifiedExpressionTree = self.expressionTreeModel.expressionTreeInterface(simplifiedExpressionTree, self.featureData) # Generate output values from simplified tree featureData array
        assert np.array_equal(numExpressionTree, numSimplifiedExpressionTree), "Tree produces different outputs after simplification"
        # print(numExpressionTree)
            
    def testEquationInterface(self):  
        self.resetModel()
        
        # -------------------------------------------------------------------#
        
        # Build a simple addition equation
        simpleAdd_Equation = "'x'+'y'"
        simpleAdd_nodeOrder = ["+", "'x'", "'y'"]
        # Assert that simple addition is valid.
        self.assertEquation2Tree(simpleAdd_Equation, simpleAdd_nodeOrder)
        
        
        # Build a simple subtraction equation
        simpleSubtract_Equation = "'x'-'y'"
        simpleSubtract_nodeOrder = ["+", "'x'", "-", "'y'"]
        # Assert that simple subtraction is valid.
        self.assertEquation2Tree(simpleSubtract_Equation, simpleSubtract_nodeOrder)
        
        # Build a simple multiplication equation
        simpleMultiply_Equation = "'x'*'y'"
        simpleMultiply_nodeOrder = ["*", "'x'", "'y'"]
        # Assert that simple multiplication is valid.
        self.assertEquation2Tree(simpleMultiply_Equation, simpleMultiply_nodeOrder)
        
        # Build a simple division equation
        simpleDivide_Equation = "'x'/'y'"
        simpleDivide_nodeOrder = ["*", "'x'", "/", "'y'"]
        # Assert that simple division is valid.
        self.assertEquation2Tree(simpleDivide_Equation, simpleDivide_nodeOrder)
        
        # Build a simple raising equation
        simpleRaise_Equation = "'x'^'y'"
        simpleRaise_nodeOrder = ["^", "'x'", "'y'"]
        # Assert that simple raising is valid.
        self.assertEquation2Tree(simpleRaise_Equation, simpleRaise_nodeOrder)
        
        simpleLog_Equation = "ln('x')"
        simpleLog_nodeOrder = ["ln", "'x'"]
        self.assertEquation2Tree(simpleLog_Equation, simpleLog_nodeOrder)
        
        simpleSin_Equation = "sin('x')"
        simpleSin_nodeOrder = ["sin", "'x'"]
        self.assertEquation2Tree(simpleSin_Equation, simpleSin_nodeOrder)
        
        simpleNeg_Equation = "-'x'"
        simpleNeg_nodeOrder = ["-", "'x'"]
        self.assertEquation2Tree(simpleNeg_Equation, simpleNeg_nodeOrder)
        
        
        # -------------------------------------------------------------------#
        
        # Build a chained addition equation
        chainedAddition_Equation = "'x'+'y'+'z'+'a'+'b'+'c'"
        chainedAddition_nodeOrder = ["+", "'x'", "'y'", "'z'", "'a'", "'b'", "'c'"]
        # Assert that chained addition is valid.
        self.assertEquation2Tree(chainedAddition_Equation, chainedAddition_nodeOrder)

        # Build a chained subtraction equation
        chainedSubtraction_Equation = "'x'-'y'-'z'-'a'-'b'-'c'"
        chainedSubtraction_nodeOrder = ["+", "'x'", "-", "-", "-", "-", "-", "'y'", "'z'", "'a'", "'b'", "'c'"]
        chainedSubtractionTree = self.expressionTreeModel.equationInterface(chainedSubtraction_Equation, self.featureNames_Tree)
        # Assert that chained subtraction is valid.
        self.assertEquation2Tree(chainedSubtraction_Equation, chainedSubtraction_nodeOrder)
        
        # Build a chained multiplication equation
        chainedMultiplication_Equation = "'x'*'y'*'z'*'a'*'b'*'c'"
        chainedMultiplicationTree = self.expressionTreeModel.equationInterface(chainedMultiplication_Equation, self.featureNames_Tree)
        # Assert that chained multiplication is valid.
        chainedMultiplication_nodeOrder = ["*", "'x'", "'y'", "'z'", "'a'", "'b'", "'c'"]
        self.assertEquation2Tree(chainedMultiplication_Equation, chainedMultiplication_nodeOrder)
        
        # Build a chained division equation
        chainedDivision_Equation = "'x'/'y'/'z'/'a'/'b'/'c'"
        chainedDivisionTree = self.expressionTreeModel.equationInterface(chainedDivision_Equation, self.featureNames_Tree)
        # Assert that chained division is valid.
        chainedDivison_nodeOrder = ["*", "'x'", "/", "/", "/", "/", "/", "'y'", "'z'", "'a'", "'b'", "'c'"]
        self.assertEquation2Tree(chainedDivision_Equation, chainedDivison_nodeOrder)
        
        # Build a chained raising equation
        chainedRaise_Equation = "'x'^('y'^('z'^('a'^('b'^'c'))))"
        chainedRaisingTree = self.expressionTreeModel.equationInterface(chainedRaise_Equation, self.featureNames_Tree)
        
        # Assert that chained raising is valid.
        chainedRaising_nodeOrder = ["^", "'x'", "^", "'y'", "^", "'z'", "^", "'a'", "^", "'b'", "'c'"]
        self.assertEquation2Tree(chainedRaise_Equation, chainedRaising_nodeOrder)
        
        chainedSin_Equation = "-sin(sin(sin(sin(sin(sin('x'))))))"
        chainedSin_nodeOrder = ["-", "sin", "sin", "sin", "sin", "sin", "sin", "'x'"]
        self.assertEquation2Tree(chainedSin_Equation, chainedSin_nodeOrder)
        
        # -------------------------------------------------------------------#
        
        # Build an equation with spaces.
        stringEquation = "  'x'    ^ 'y'   "
        simpleRaisingTree_Spaces = self.expressionTreeModel.equationInterface(stringEquation, self.featureNames_Tree)
        # Assert that spaces are fine.
        simpleRaisingTree_nodeOrder = ["^", "'x'", "'y'"]
        self.assertEquation2Tree(stringEquation, simpleRaisingTree_nodeOrder)
        
        # Build an equation with parenthesis.
        stringEquation = "'x'*('x'*'z')*('a')*('b'*'c')"
        parenthesisEquation_Raise = self.expressionTreeModel.equationInterface(stringEquation, self.featureNames_Tree)
        # Assert that equation with parenthesis are fine.
        simpleRaisingTree_nodeOrder = ["*", "'x'", "*", "'a'", "*", "'x'", "'z'", "'b'", "'c'"]
        self.assertEquation2Tree(stringEquation, simpleRaisingTree_nodeOrder)
        
        # Build an equation with parenthesis.
        stringEquation = "'x'^(('y'^'z')^('a'^('b'^'c')))"
        parenthesisEquation_Raise = self.expressionTreeModel.equationInterface(stringEquation, self.featureNames_Tree)
        # Assert that equation with parenthesis are fine.
        simpleRaisingTree_nodeOrder = ["^", "'x'", "^", "^", "^", "'y'", "'z'", "'a'", "^", "'b'", "'c'"]
        self.assertEquation2Tree(stringEquation, simpleRaisingTree_nodeOrder)
        
        # Build an equation with parenthesis.
        stringEquation = "'x'-('y'-'z')-'a'-('b'-'c')"
        parenthesisEquation_Subtract = self.expressionTreeModel.equationInterface(stringEquation, self.featureNames_Tree)
        # Assert that equation with parenthesis are fine.
        simpleSubtract_nodeOrder = ["+", "'x'", "-", "-", "-", "+", "'a'", "+", "'y'", "-", "'b'", "-", "'z'", "'c'"]
        self.assertEquation2Tree(stringEquation, simpleSubtract_nodeOrder)
        
        # Build an equation with parenthesis.
        stringEquation = "'x'/('y'/'z')/('a')/('b'/'c')"
        parenthesisEquation_Divide = self.expressionTreeModel.equationInterface(stringEquation, self.featureNames_Tree)
        parenthesisDivide_nodeOrder = ["*", "'x'", "/", "/", "/", "*", "'a'", "*", "'y'", "/", "'b'", "/", "'z'", "'c'"]
        self.assertEquation2Tree(stringEquation, parenthesisDivide_nodeOrder)
        # Assert that equation with parenthesis are fine.
        
        stringEquation = "sin(('x' * sin('y' * sin('z'))) * sin('a') * sin('b' + sin('c')))"
        sinEquation_Multiply = self.expressionTreeModel.equationInterface(stringEquation, self.featureNames_Tree)
        sinEquation_Multiply_nodeOrder = ["sin", "*", "'x'", "sin", "sin", "sin", "*", "'a'", "+", "'y'", "sin", "'b'", "sin", "'z'", "'c'"]
        self.assertEquation2Tree(stringEquation, sinEquation_Multiply_nodeOrder)
        
        stringEquation = "sin('x' * 'y') - 'z'"
        simpleFunction_nodeOrder = ["+", "sin", "-", "*", "'z'", "'x'", "'y'"]
        self.assertEquation2Tree(stringEquation, simpleFunction_nodeOrder)
        
        stringEquation = "-sin('x' * 'y') - 'z'"
        simpleFunction_nodeOrder = ["+", "-", "-", "sin", "'z'", "*", "'x'", "'y'"]
        self.assertEquation2Tree(stringEquation, simpleFunction_nodeOrder)
        
        # -------------------------------------------------------------------#
        
        # More complex equations
        
        stringEquation = "sin('x' * 'y') - 'z' + 'a' / cos('b' ^ (-'c'))"
        complexFunction_nodeOrder = ["+", "sin", "-", "*", "*", "'z'", "'a'", "/",
                                     "'x'" , "'y'", "cos", "^", "'b'", "-", "'c'"]
        self.assertEquation2Tree(stringEquation, complexFunction_nodeOrder)
        
        stringEquation = "('a' ^ 'b' * 'c' - sin('x')) + ('a' ^ 'b' * 'c' - sin('x'))"
        complexFunction_nodeOrder = ["+", "*", "-", "+", "^", "'c'", "sin", "*", "-",
                                     "'a'", "'b'", "'x'" , "^", "'c'", "sin",
                                     "'a'", "'b'", "'x'"]
        self.assertEquation2Tree(stringEquation, complexFunction_nodeOrder)
        
        stringEquation = "(('a' ^ 'b') / ('a' ^ 'b')) + ('x' * ln('y'))"
        complexFunction_nodeOrder = ["+", "*", "*", "^", "/", "'x'", "ln", "'a'", "'b'",
                                     "^", "'y'", "'a'", "'b'"]
        self.assertEquation2Tree(stringEquation, complexFunction_nodeOrder)
        
        stringEquation = "(('a' + 'b') * ('x' + 'y')) + ((sin('a') + sin('b')) * (ln('x') +ln('y')))"
        complexFunction_nodeOrder = ["+", "*", "*", "+", "+", "+", "+", "'a'", "'b'",
                                     "'x'", "'y'", "sin", "sin", "ln", "ln",
                                     "'a'", "'b'", "'x'", "'y'"]
        self.assertEquation2Tree(stringEquation, complexFunction_nodeOrder)
        
        stringEquation = "(('x' * 'y') + 'z' / 'a') * (-(('x' * 'y') + 'z' / 'a'))"
        complexFunction_nodeOrder = ["*", "+", "-", "*", "*", "+", "'x'" , "'y'", "'z'", "/", "*", "*", "'a'",
                                     "'x'" , "'y'", "'z'", "/", "'a'"]
        self.assertEquation2Tree(stringEquation, complexFunction_nodeOrder)
        
        
        stringEquation = "ln(sin('x' * 'y')) - 'z' * 'a'"
        simpleFunction_nodeOrder = ["+", "ln", "-", "sin", "*", "*", "'z'", "'a'", "'x'", "'y'"]
        self.assertEquation2Tree(stringEquation, simpleFunction_nodeOrder)
        
        
     
        # -------------------------------------------------------------------#
        
        # Original complex equation.
        stringEquation = "sin('x'*'y'-'z') + 2.01^'x'/ 3 *'y' - 'a'*'x'^('y'/'z')"
        complex_nodeOrder = ["+", "sin", "*", "-", "+", "^", "/", "'y'", "*", "*", "-", "2.01", "'x'", "3", "'a'", "^", "'x'", "'y'", "'z'", "'x'", "*", "'y'", "/", "'z'"]
        self.assertEquation2Tree(stringEquation, complex_nodeOrder)
        
    
    def assertRandomEquations(self, numEquations = 1000):
        # Generate random equations.
        equationList = self.datasetGenerator.generateEquations(numEquations)
        
        for stringEquation in equationList:
            print(stringEquation)
            self.assertEquation2Tree(stringEquation, [])    

# ---------------------------------------------------------------------------#

if __name__ == "__main__":
    # Instantiate unit test class.
    unitTester_ExpressionTreeModel = testExpressionTreeModel()
        
    # Test machine learning models
    unitTester_ExpressionTreeModel.testEquationInterface()
    # unitTester_ExpressionTreeModel.assertRandomEquations()

    
    
    