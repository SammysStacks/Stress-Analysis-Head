# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# General
import numpy as np


# Standardize data class
class standardizeData:
    def __init__(self, X, axisDimension=0, threshold=10E-15):
        self.axisDimension = axisDimension

        X = np.asarray(X)
        self.mu_ = np.mean(X, axis=axisDimension)
        self.sigma_ = np.std(X, ddof=1, axis=axisDimension)

        if (self.sigma_ <= threshold).any():
            print(self.sigma_, np.asarray(X).shape)
            assert False, "Cannot Standardize the Data. The standard deviation is too small."

        if self.axisDimension == 1:
            self.mu_ = self.mu_.reshape(-1, 1)
            self.sigma_ = self.sigma_.reshape(-1, 1)

    def standardize(self, X):
        X = np.asarray(X)
        return (X - self.mu_) / self.sigma_

    def unStandardize(self, Xhat):
        return self.mu_ + self.sigma_ * Xhat

    def getStatistics(self):
        return self.mu_, self.sigma_


# -------------------------------------------------------------------------- #
# ------------------------------ Test Methods ------------------------------ #

def testRestandardization(data):
    # Initialize the standardization class.
    standardizationClass = standardizeData(data)
    # Standardize the data
    standardizedData = standardizationClass.standardize(data)
    restandardizedData = standardizationClass.unStandardize(standardizedData)
    # Assert that the standardization worked
    assert np.allclose(restandardizedData, data)


# -------------------------------------------------------------------------- #
# ------------------------------ User Testing ------------------------------ #

if __name__ == "__main__":
    # Specify data params
    numDataPoints = 10
    numFeatures = 5
    # Initialize random data
    featureData = np.random.uniform(0, 1, (numDataPoints, numFeatures))
    featureLabels = np.random.uniform(0, 1, (numDataPoints, 1))

    # Test forward-backward method.
    testRestandardization(featureData)
    testRestandardization(featureLabels)
