
# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# Basic modules
import math
import numpy as np
                
# -------------------------------------------------------------------------- #
# ---------------------------- Math Expressions ---------------------------- #

class mathExpressions:        
    
    def __init__(self): 
        # Initialize scalars and transformations: operators and functions.
        self._createTransformers()
        self._createScalars()
    
    # ---------------------------------------------------------------------- #
    # ----------------- Organize Transformers and Variables ---------------- #  
        
    def _createTransformers(self):
        # Initialize all transformations
        self._createOperators()
        self._createFunctions()
        self._organizeTransformers()
        
        # Map the transformers to readable expressions.
        self._transformationDecoder()
    
    def _createOperators(self):
        # Create commutative (order-independant) operators: f(x, y) = f(y, x).
        self.add = lambda x: sum(x) #np.asarray(x).sum(axis=0)         # Summation. Assume the inputs is [[f1], [f2]] -> f1 + f2
        self.multiply = lambda x: self.multiplyListElems(x)    # Multiplication. Assume the inputs is [[f1], [f2]] -> f1*f2
        # Store commutative operators
        self.operators = [self.add, self.multiply]
        self.numOpsCommute = len(self.operators)

        # Create non-commutative (order-dependant) operators: f(x, y) != f(y, x).
        self.derivXY = lambda x: np.gradient(x[0], x[1])  # Derivation: dx[0]/dx[1].
        self.raiseXY = lambda x: x[0] ** x[1]  # Exponent: x^y.
        # self.integrateXY lambda x: scipy.integrate.simpson(x[0], xData[1]) OR int_f = lambda f, a, b: quad(f, a, b)[0]
        # Store non-commutative operators.
        self.operators.extend([self.raiseXY])
        self.numOperators = len(self.operators) 
        self.numOpsNonCommute = self.numOperators - self.numOpsCommute
        
    def _createFunctions(self):
        # Once applied transformation methods.
        unitary = lambda x: np.asarray(x)  # Unit scale
        
        # Non-commutative operators, formatted as functions.
        self.negate = lambda x: - np.asarray(x)   # Negative scale
        self.invert = lambda x: 1 / np.asarray(x) # Invert values
        
        # Create functions.
        self.sin = lambda x: np.sin(np.asarray(x, dtype=float)) # Input is radians.
        self.cos = lambda x: np.cos(np.asarray(x, dtype=float)) # Input is radians.
        self.exp = lambda x: np.exp(np.asarray(x)) # Raising e to the x: e^(x).
        self.ln = lambda x: np.log(np.asarray(x))  # Natural Logarithm: Ln(x).
        
        # Save all transformation methods.
        self.functions_onceApplied = [unitary]
        self.functions = [self.negate, self.invert, self.sin, self.cos, self.exp, self.ln]
        self.numFunctions = len(self.functions)
                
    def _organizeTransformers(self):
        # Organize all the transformers
        self.transformers = self.operators.copy()
        self.transformers.extend(self.functions.copy())
        # Save the number of transformers.
        self.numTransformers = len(self.transformers)
        
        # Convert operators to numpy arrays
        self.operators = np.array(self.operators)
        self.functions = np.array(self.functions)
        self.transformers = np.array(self.transformers)
        self.functions_onceApplied = np.array(self.functions_onceApplied)

    def _createScalars(self):        
        # General constants.
        pi = math.pi    # Pi = 3.14159.
        e = math.exp(1) # e = 2.718281.
        # Kee track of the constants.
        self.scalars = [0.5, 1, 1.5, 2, e, pi, pi/8]
        self.scalars.extend(self.scalars*-1)
        
        # Physics constants.
        gravity = 6.6743015 * 10**-11           # Newton * meters^2 / KiloGram^2.
        speedOfLight = 299792458                # meters / second.
        planksConstant = 6.62607015 * 10**-34   # Joules / Hertz.
        earthGravitationalAccel = 9.8           # meters / second^2.
        # Kee track of the constants.
        self.scalars.extend([gravity, speedOfLight, planksConstant, earthGravitationalAccel])
        
    def _transformationDecoder(self):
        # Map transformers to their strings.
        self.mapTransformer_toString  = {
            # Commutative operators.
            self.add: " + ",
            self.multiply: "*",
            # Non-commutative operators.
            self.raiseXY: "^(",
            
            # Non-commutative functions.
            self.invert: " / (",
            self.negate: " - (",
            # Functions.
            self.sin: "sin(",
            self.cos: "cos(",
            self.exp: "exp(",
            self.ln: "ln(",
            }
        # Map strings to their transformers.
        self.mapString_toTransformer = {
            # Commutative operators.
            "+": self.add,
            "*": self.multiply,
            # Non-commutative operators.
            "^": self.raiseXY,

            # Non-commutative functions.
            "-": self.negate,
            "/": self.invert,
            # Functions.
            "sin": self.sin,
            "cos": self.cos,
            "exp": self.exp,
            "ln": self.ln,
        }
        
    def multiplyListElems(self, itemList):
        result = 1
        for elem in itemList:
            result *= elem
        return result
    
    # ---------------------------------------------------------------------- #
    # ------------------------ Transformer Interface ----------------------- #
    
    def assertValidTransformerIndex(self, transformerIndex):
        assert  0 <= transformerIndex < self.numTransformers, "Transformer index is out of range: " + str(transformerIndex)
        # assert isinstance(transformerIndex, int), "Transformer index must be an integer: " + str(transformerIndex)
    
    def findMaxChildren_Transformer(self, transformerIndex):
        # Assert that this is a valid index.
        self.assertValidTransformerIndex(transformerIndex)
        
        # Commutative operator have infinite children (min 2)
        if transformerIndex < self.numOpsCommute:
            return np.inf
        # Non-commutative operators have 2 children
        elif transformerIndex < self.numOperators:
            return 2
        # Function have 1 child.
        elif transformerIndex < self.numTransformers:
            return 1
        else:
            assert False
        
    def getTransformer(self, transformerIndex):
        # Assert that this is a valid index.
        self.assertValidTransformerIndex(transformerIndex)
        
        # Return the transformer.
        return self.transformers[transformerIndex]

    def getTransformerIndex(self, transformer):
        # Try to find the transformer index.
        transformerIndexes = np.where(self.transformers == transformer)
        assert len(transformerIndexes) == 1, "Something is wrong with this transformer"
        
        # Return the transformer index.
        return transformerIndexes[0][0]
    
    def isFunctionIndex(self, transformerIndex):
        return self.numOperators <= transformerIndex
    
    def isFunctionNode(self, node):
        return node.numericValue in self.functions
    
    def isFunctionChar(self, char):
        return char.isalpha() or char in ["-", "/"]
    
    def isOperatorNode(self, node):
        return node.numericValue in self.operators
    
    def isTransformerNode(self, node):
        return self.isOperatorNode(node) or self.isFunctionNode(node)
            
    # ---------------------------------------------------------------------- #
    # ------------------------- Variable Interface ------------------------- #
        
    def isScalarNode(self, node):
        return isinstance(node.numericValue, float)
    
    def isFeatureNode(self, node):
        return isinstance(node.numericValue, (int, np.int64))
    
    def isVariableNode(self, node):
        return self.isScalarNode(node) or self.isFeatureNode(node)


