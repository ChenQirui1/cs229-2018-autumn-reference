import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    clf = GDA()
    clf.fit(x_train,y_train)
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m,n = np.shape(x)

        phi = sum(y)/m

        mu1 = np.sum(x[np.nonzero(y)])/sum(y)

        mu0 = np.sum(x[np.where(y==0)])/(m-sum(y))

        def covariance_matrix(mu0,mu1):
            #X, design matrix for feature vector x where y == 1
            x_true = x[np.nonzero(y)]
            x_true = x_true - mu1
            sum_true = np.sum(np.repeat(x_true,n,axis=1) * np.tile(x_true,n),axis=0)


            x_false = x[np.where(y==0)]
            x_false = x_false - mu0
            sum_false = np.sum(np.repeat(x_false,n,axis=1) * np.tile(x_false,n),axis=0)

            total = np.sum([sum_false,sum_true],axis=0)/m
            result = np.reshape(total,(n,n))

            return result
        print(covariance_matrix(mu0,mu1))



        # mu0 = np.n


        def joint_likelihood():
            pass



        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        # *** END CODE HERE
