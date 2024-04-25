
# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# Basic modules
import random
import numpy as np

# Libary for simplification of string equations
import sympy
from sympy.parsing.sympy_parser import parse_expr
                
# -------------------------------------------------------------------------- #
# ---------------------------- Dataset Generator --------------------------- #

class datasetGenerator:        
    
    def __init__(self): 
        # Initialize mathmatical terms.
        self.variables = np.array(["'x'", "'y'", "'z'", "'a'", "'b'", "'c'"])
        self.operators = np.array(['+', '-', '*', '/', "^"])
        self.functions = np.array(['sin', 'cos', "ln", "exp"])
        
        # Set the random parameters
        self.probSelectVariable = 0.333
        self.probSelectOperator = 0.333
        
    # ---------------------------------------------------------------------- #
    # -------------------------- Excel Processing -------------------------- #  
    
    def saveEquationList(self, equationList, file_path):
        # Save the equations to a text file with comma as the delimiter
        np.savetxt(file_path, equationList, fmt="%s", delimiter=',')
        
    def getEquationList(self, file_path):
        # Load the equations from the text file using comma as the delimiter
        return np.loadtxt(file_path, dtype=str, delimiter=',')        
    
    # ---------------------------------------------------------------------- #
    # ----------------------- Generate New Equations ----------------------- #  
    
    # def generateEquation(self, depth=0, max_depth=5):        
    #     randomFloat = random.random()
        
    #     if 0 <= randomFloat < 0.333333:
    #         randomVarable = random.choice(self.variables)
    #         return randomVarable + self.generateEquation(depth - 1)
    #     elif 0.333333 <= randomFloat < 0.6666666:
    #         randomOperator = random.choice(self.operators)
    #         return randomOperator + self.generateEquation(depth)
    #     else:
    #         randomFunction = random.choice(self.functions)
    #         # Create the function wit a random equation.
    #         function = ""
    #         randomDepth = random.randint(0, randomDepth)
    #         if randomDepth != 0:
    #             operand = self.generateEquation(randomDepth)
    #             function = f" {randomFunction}({operand})"
    #             depth -= randomDepth
            
    #         nextOperand = ""
    #         if depth != 0:
    #             randomOperator = random.choice(self.operators)
            
    #         # Keep adding to the equation
    #         return f" randomFunction({})"

    def generateEquation(self, depth=0, max_depth=5):
        randomFloat = random.random()
    
        if depth >= max_depth or randomFloat < 0.25:
            randomVariable = random.choice(self.variables)
            return randomVariable
    
        elif 0.25 <= randomFloat < 0.5:
            randomOperator = random.choice(self.operators)
            left_operand = self.generateEquation(depth + 1)
            right_operand = self.generateEquation(depth + 1)
            return f"({left_operand} {randomOperator} {right_operand})"
    
        elif 0.5 <= randomFloat < 0.75:
            randomFunction = random.choice(self.functions)
            operand = self.generateEquation(depth + 1)
            return f"{randomFunction}({operand})"
    
        else:
            operand = self.generateEquation(depth + 1)
            return f"({operand})"

    def generateEquation2(self, depth=0, max_depth=5):
        # Check if the depth has reached the maximum depth or a random number is less than 0.3
        if depth >= max_depth or random.random() < 0.3:
            # Randomly add variale or function.
            if random.random() < self.probSelectVariable:
                return random.choice(self.variables)  # Return a random variable
            else:
                function = random.choice(self.functions)  # Randomly choose a function
                equation = self.generateEquation(depth + 1)  # Generate a random equation to be inside the function
                return f"{function}({equation})"  # Return the function with the equation as an argument

        else:
            operator = random.choice(self.operators)  # Randomly choose an operator
            # Recursively generate the left and right operands
            left_operand = self.generateEquation(depth + 1)
            right_operand = self.generateEquation(depth + 1)
            # Return the equation with the left and right operands and the operator enclosed in parentheses
            return f"({left_operand} {operator} {right_operand})" * 32

    
    def generateEquations(self, numEquations, depth = 0, max_depth = 5):
        # Initialize an empty holder for all equations.
        equationList = np.zeros(numEquations, dtype=object)
        
        # For each new equation index.
        for equationInd in range(numEquations):
            # Generate and store a random equation.
            equation = self.generateEquation(depth, max_depth)
            equationList[equationInd] = equation
            
        return equationList
    
    def simplify_equation(self, equation_string):
        equation_string = equation_string.replace("'", "").replace("^", "**").replace(" ", "")
        equation_string = parse_expr(equation_string)
        simplified_equation = sympy.simplify(equation_string)
        return simplified_equation


if __name__ == "__main__":
    equationGeneratorClass = datasetGenerator()
    
    numEquations = 5000
    # Generate random equations.
    equationList = equationGeneratorClass.generateEquations(numEquations)

    saveEquationFile = "./metadata/generatedEquations.txt"
    # Save and retrieve the equation list.
    equationGeneratorClass.saveEquationList(equationList, saveEquationFile)
    retrievedEquations = equationGeneratorClass.getEquationList(saveEquationFile)
    # Assert that the file saved correctly
    assert all(retrievedEquations == equationList)
    
    
    