import torch
import numpy as np
import math

class FallDetecion():
    def __init__(self):
        pass

    def __call__(self, skeleton_cache):
        '''
            This __call__ function takes a cache of skeletons as input, with a shape of (M x 17 x 2), where M represents the number of skeletons.
            The value of M is constant and represents time. For example, if you have a 7 fps stream and M is equal to 7 (M = 7), it means that the cache length is 1 second.
            The number 17 represents the count of points in each skeleton (as shown in skeleton.png), and 2 represents the (x, y) coordinates.

            This function uses the cache to detect falls.

            The function will return:
                - bool: isFall (True or False)
                - float: fallScore
        '''

        # The function 'get_skeleton_cache' retrieves the 'skeleton_cache' data and computes angles.
        # It returns a list for a fixed frame containing its eight significant angles, utilizing the
        # 'angle_calculator' function. This function, in turn, relies on 'getv' to obtain vectors.
        # For a single frame, it applies this function and calculates angles for the entire cache.
        # Subsequently, it determines the argument for the sigmoid function using a specific formula.
        # This argument is then passed to the 'fallScore_calculator,' which yields the probability
        # of a fall. We use a 77% threshold to determine whether a fall has occurred or not.

        angles = self.angles_getter(skeleton_cache)
        detector = self.sigmoid_argument_getter(angles)
        fallScore = self.fallScore(detector)
        isFall = self.isFall(fallScore)
        return isFall, fallScore

    # Calculates the angles between 2 vectors
    def angle_calculator(self, vec1, vec2):
        if np.linalg.norm(vec1) != 0 and np.linalg.norm(vec2) != 0:
            return abs(
                np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))) * 180 / math.pi)
        return 0

    # gets the vector from 2 points
    def getv(self, a, b):
        c = [b[0] - a[0], b[1] - a[1]]
        return c

    # returns a list for a fixed frame with its 8 important angles by using angle calculator
    def angles_calculator(self, data, order):
        midpoint = [(data[order][5][0] + data[order][6][0]) / 2, (data[order][5][1] + data[order][6][1]) / 2]
        midpoint1 = [(data[order][11][0] + data[order][12][0]) / 2, (data[order][11][1] + data[order][12][1]) / 2]
        angles = []

        angles.append(self.angle_calculator(
            self.getv(data[order][13], data[order][11]),
            self.getv(data[order][13], data[order][15])))

        angles.append(self.angle_calculator(
            self.getv(data[order][14], data[order][12]),
            self.getv(data[order][14], data[order][16])))

        angles.append(self.angle_calculator(
            self.getv(data[order][12], data[order][11]),
            self.getv(data[order][12], data[order][14])))

        angles.append(self.angle_calculator(
            self.getv(data[order][11], data[order][13]),
            self.getv(data[order][11], data[order][12])))

        angles.append(self.angle_calculator(
            self.getv(data[order][5], data[order][7]),
            self.getv(data[order][5], data[order][11])))

        angles.append(self.angle_calculator(
            self.getv(data[order][6], data[order][8]),
            self.getv(data[order][6], data[order][12])))

        angles.append(self.angle_calculator(
            self.getv(midpoint, data[order][0]),
            self.getv(midpoint, data[order][12])))

        angles.append(self.angle_calculator(
            self.getv(midpoint1, data[order][0]),
            self.getv(midpoint1, data[order][6])))
        return angles


    # returns the (m, 8) array, where 8is the number of angles for each frame, and m is the number of frames in the cache.
    def angles_getter(self, cache):
        frame_indices = np.arange(len(cache))
        angles = [self.angles_calculator(cache, frame) for frame in frame_indices]
        return np.array(angles)

    # calculates and returns average value, which can serve as an argument for the sigmoid function.
    def sigmoid_argument_getter(self, angles):
        result = 0
        for i in range(len(angles) - 1):
            for j in range(8):
                update = abs(angles[i + 1][j] - angles[i][j])
                result += update

        result /= 8
        result /= 107
        return result

    # returns the probability of detecting fall based on the cache information,
    # that is preprocessed, got angles and found the parameter x for the sigmoid function.
    def fallScore(self, x):
        return (1 / (1 + math.e ** (-x))) * 100

    # If the probability is above 77%, it assigns Fall as True, otherwise False.

    def isFall(self, score):
        if score >= 77:
            return True
        return False


CACHE1 = np.load('C:/Users/User/lectures/data/skeleton_1.npy')[36:72]
CACHE2 = np.load('C:/Users/User/lectures/data/skeleton_2.npy')[36:72]
CACHE3 = np.load('C:/Users/User/lectures/data/skeleton_3.npy')[54:90]

fd = FallDetecion()
print(fd(CACHE1))
print(fd(CACHE2))
print(fd(CACHE3))