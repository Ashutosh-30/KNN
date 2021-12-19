
import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY CODES ABOVE
############################################################################
# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    # F1_SCORE(from wikipedia, rearranging variables) = 2tp/(2tp+fp+fn) : tp -> true_positive
    true_positive = sum([x == 1 and y == 1 for x, y in zip(real_labels, predicted_labels)])  # hit
    false_positive = sum([x == 0 and y == 1 for x, y in zip(real_labels, predicted_labels)])  # false alarm
    false_negative = sum([x == 1 and y == 0 for x, y in zip(real_labels, predicted_labels)])  # miss

    # print(true_positive)
    # print(false_positive)
    # print(false_negative)

    denominator = float(2 * true_positive + false_positive + false_negative)
    # print(denominator)
    if denominator == 0.0:
        return 0 # to avoid division by zero
    f1_score = 2 * true_positive / denominator
    return f1_score


class Distances:
    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        p = 3
        s = 0
        for p1, p2 in zip(point1, point2):
            s = s + np.absolute(p1 - p2) ** p

        return float(np.cbrt(s))

    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        s = 0
        for p1, p2 in zip(point1, point2):
            s = s + (p1 - p2) ** 2

        return float(np.sqrt(s))

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        norm1 = 0
        norm2 = 0
        prod = 0
        for p1, p2 in zip(point1, point2):
            norm1 = norm1 + p1 ** 2
            norm2 = norm2 + p2 ** 2
            prod = prod + p1 * p2
        if norm1 == 0 or norm2 == 0:
            return 1
        return float(1 - float(prod) / float(np.sqrt(norm1) * np.sqrt(norm2)))



class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you need to try different distance functions you implemented in part 1.1 and different values of k (among 1, 3, 5, ... , 29), and find the best model with the highest f1-score on the given validation set.

        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] training labels to train your KNN model
        :param x_val:  List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), and model (an instance of KNN) and assign them to self.best_k,
        self.best_distance_function, and self.best_model respectively.
        NOTE: self.best_scaler will be None.

        NOTE: When there is a tie, choose the model based on the following priorities:
        First check the distance function:  euclidean > Minkowski > cosine_dist
		(this will also be the insertion order in "distance_funcs", to make things easier).
        For the same distance function, further break tie by prioritizing a smaller k.
        """

        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_model = None

        max_f1_score = 0
        for df in distance_funcs:
            for k in range(1, min(31, len(x_train) + 1), 2):
                nth_model = KNN(k, distance_funcs[df])
                nth_model.train(x_train, y_train)
                y_dash = nth_model.predict(x_val)
                curr_f1_score = f1_score(y_val, y_dash)
                if curr_f1_score > max_f1_score:  # if better f1 score, update variables
                    max_f1_score = curr_f1_score
                    self.best_k = k
                    self.best_distance_function = df
                    self.best_model = nth_model

    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is the same as "tuning_without_scaling", except that you also need to try two different scalers implemented in Part 1.3. More specifically, before passing the training and validation data to KNN model, apply the scalers in scaling_classes to both of them.

        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param scaling_classes: dictionary of scalers (key is the scaler name, value is the scaler class) you need to try to normalize your data
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), scaler (its name), and model (an instance of KNN), and assign them to self.best_k, self.best_distance_function, best_scaler, and self.best_model respectively.

        NOTE: When there is a tie, choose the model based on the following priorities:
        First check scaler, prioritizing "min_max_scale" over "normalize" (which will also be the insertion order of scaling_classes). Then follow the same rule as in "tuning_without_scaling".
        """

        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None
        max_f1_score = 0
        for sc in scaling_classes:
            s = scaling_classes[sc]()  # The only
            s_x_train = s.__call__(x_train)  # difference
            s_x_val = s.__call__(x_val)  # from unscaled run
            for df in distance_funcs:
                for k in range(1, min(31, len(x_train) + 1), 2):
                    nth_model = KNN(k, distance_funcs[df])
                    nth_model.train(s_x_train, y_train)
                    y_dash = nth_model.predict(s_x_val)
                    curr_f1_score = f1_score(y_val, y_dash)
                    if curr_f1_score > max_f1_score:  # if better f1 score, update variables
                        max_f1_score = curr_f1_score
                        self.best_k = k
                        self.best_distance_function = df
                        self.best_scaler = sc
                        self.best_model = nth_model


class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        features = np.array(features, dtype=float) # converting to array to use append sum sqrt etc
        n_features = [] # initialized empty array to store normalised features
        for i in range(len(features)):
            for j in range(len(features[i])):
                if features[i][j] == 0:
                    n_features.append(0)
                else:
                    n_features.append(features[i][j] / np.sqrt(np.sum(features[i] * features[i])))
        n_features = np.reshape(n_features, (features.shape[0], features.shape[1]))
        return n_features.tolist() # because they want the return to be list rather array



class MinMaxScaler:
    def __init__(self):
        pass

    # TODO: min-max normalize data
    def __call__(self, features):
        """
		For each feature, normalize it linearly so that its value is between 0 and 1 across all samples.
        For example, if the input features are [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]].
		This is because: take the first feature for example, which has values 2, -1, and 0 across the three samples.
		The minimum value of this feature is thus min=-1, while the maximum value is max=2.
		So the new feature value for each sample can be computed by: new_value = (old_value - min)/(max-min),
		leading to 1, 0, and 0.333333.
		If max happens to be same as min, set all new values to be zero for this feature.
		(For further reference, see https://en.wikipedia.org/wiki/Feature_scaling.)

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        features = np.array(features, dtype=float) # converting to array to use amin amax etc
        # if self.min is None or self.max is None: # keep track if first call or not
        # print("start")
        minimum = np.amin(features, axis=0)
        maximum = np.amax(features, axis=0)
        diff = maximum - minimum
        diff[diff == 0] = 1
        diff_2d = np.reshape(diff, (1, -1)) # reshaping to make it 1XN(from N)
        # print(minimum.shape)
        # print(maximum.shape)
        # print(diff.shape)
        # print(diff_2d.shape)
        # print("end")
        n_features = (features - minimum) / (diff_2d)
        return n_features.tolist() # because they want the return to be list rather array
