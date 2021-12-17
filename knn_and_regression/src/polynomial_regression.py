import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

class PolynomialRegression():
    def __init__(self, degree):
        """
        Implement polynomial regression from scratch.
        
        This class takes as input "degree", which is the degree of the polynomial 
        used to fit the data. For example, degree = 2 would fit a polynomial of the 
        form:

            ax^2 + bx + c
        
        Your code will be tested by comparing it with implementations inside sklearn.
        DO NOT USE THESE IMPLEMENTATIONS DIRECTLY IN YOUR CODE. You may find the 
        following documentation useful:

        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

        Here are helpful slides:

        http://interactiveaudiolab.github.io/teaching/eecs349stuff/eecs349_linear_regression.pdf
    
        The internal representation of this class is up to you. Read each function
        documentation carefully to make sure the input and output matches so you can
        pass the test cases. However, do not use the functions numpy.polyfit or numpy.polval. 
        You should implement the closed form solution of least squares as detailed in slide 10
        of the lecture slides linked above.

        Usage:
            import numpy as np
            
            x = np.random.random(100)
            y = np.random.random(100)
            learner = PolynomialRegression(degree = 1)
            learner.fit(x, y) # this should be pretty much a flat line
            predicted = learner.predict(x)

            new_data = np.random.random(100) + 10
            predicted = learner.predict(new_data)

            # confidence compares the given data with the training data
            confidence = learner.confidence(new_data)


        Args:
            degree (int): Degree of polynomial used to fit the data.
        """
        self.degree = degree
        self.confidence = None
        self.w = None  # define the weights
        self.f = None  # function
        self.X_training = None


    def fit(self, features, targets):
        """
        Fit the given data using a polynomial. The degree is given by self.degree,
        which is set in the __init__ function of this class. The goal of this
        function is fit features, a 1D numpy array, to targets, another 1D
        numpy array.
        

        Args:
            features (np.ndarray): 1D array containing real-valued inputs.
            targets (np.ndarray): 1D array containing real-valued targets.
        Returns:
            None (saves model and training data internally)
        """
        X = np.ones((features.size, self.degree + 1))
        X_row = X.shape[0]
        X_column = X.shape[1]

        for i in range(X_row):
            for j in range (1, X_column):
                X[i,j] = features[i]**j

        # formula for weights: (X^TX)X^ty
        X_inv = X.T
        #self.w = np.matmul(np.matmul((np.linalg.inv(np.matmul(X_inv, X))), X_inv), targets)
        self.w = np.linalg.inv(X.T @ X) @ X.T @ targets
        #self.w = ((np.linalg.inv((X.T).dot(X))).dot(X.T)).dot(targets)


        self.features_row = features.shape[0]
        self.f = np.zeros((self.features_row))
        for i in range(self.w.size):
            self.f += self.w[i] * np.sort(features)**i
    
        self.X_training = features  # save training data internally
        
    def predict(self, features):
        """
        Given features, a 1D numpy array, use the trained model to predict target 
        estimates. Call this after calling fit.

        Args:
            features (np.ndarray): 1D array containing real-valued inputs.
        Returns:
            predictions (np.ndarray): Output of saved model on features.
        """
        predictions = np.zeros((features.shape[0]))
        
        for i in range(self.w.size):
            predictions += self.w[i] * features**i

        return predictions


    def visualize(self, features, targets, path, title, color='g'):
        """
        This function should produce a single plot containing a scatter plot of the
        features and the targets, and the polynomial fit by the model should be
        graphed on top of the points.

        DO NOT USE plt.show() IN THIS FUNCTION. Instead, use plt.savefig().

        Args:
            features (np.ndarray): 1D array containing real-valued inputs.
            targets (np.ndarray): 1D array containing real-valued targets.
        Returns:
            None (plots to the active figure)
        """

        plt.figure()
        plt.scatter(features, targets)
        plt.plot(np.sort(self.X_training), self.f)
        plt.title(title)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.savefig(path)
