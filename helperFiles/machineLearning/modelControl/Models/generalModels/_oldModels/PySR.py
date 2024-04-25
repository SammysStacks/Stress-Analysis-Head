



import sklearn
import numpy as np
from pysr import PySRRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

X = standardizedFeatures

y = standardizedLabels[2]

Training_Data, Testing_Data, Training_Labels, Testing_Labels = train_test_split(X, y, test_size=0.2, shuffle= True)


model = PySRRegressor(
    niterations=100,  # < Increase me for better results
    binary_operators=["+", "-", "*", "/"],
    unary_operators=[
        "cos",
        "exp",
        "sin",
        "inv(x) = 1/x",
        # ^ Custom operator (julia syntax)
    ],
    extra_sympy_mappings={"inv": lambda x: 1 / x},
    # ^ Define operator for SymPy as well
    loss="loss(prediction, target) = (prediction - target)^2",
    # ^ Custom loss function (julia syntax)
    # early_stop_condition=(
    # "stop_if(loss, complexity) = loss < 1e-6 && complexity < 10"
    # # Stop early if we find a good and simple equation
    # ),
    # timeout_in_seconds=60 * 60 * 24,
    # ^ Alternatively, stop after 24 hours have passed.
    maxsize=50,
    # ^ Allow greater complexity.
    maxdepth=10,
    # ^ But, avoid deep nesting.
)



model.fit(Training_Data, Training_Labels) #, variable_names=featureNames)

predictedLabels_All = model.predict(X)
predictedLabels_Test = model.predict(Testing_Data)


plt.title('Machine Learning Predicted Results')
plt.plot(y, predictedLabels_All, 'ko', markersize=4)
plt.plot(Testing_Labels, predictedLabels_Test, 'bo', markersize=4)
plt.plot([min(y), max(y)], [min(y), max(y)], '-', c='tab:red')
plt.xlim(min(y)*0.95, max(y)*1.05)
plt.ylim(min(y)*0.95, max(y)*1.05)
plt.xlabel("STAI-Y1 Stress Scores")
plt.ylabel("Predicted Stress Scores")
plt.show()