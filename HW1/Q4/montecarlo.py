import numpy as np

def q4(numPoints):
    # Generate random x and y values between -1 and 1
    x = np.random.uniform(-1, 1, numPoints)
    y = np.random.uniform(-1, 1, numPoints)

    # Check which points fall inside the unit circle
    insideCircle = (x**2 + y**2) <= 1

    # Count how many are inside
    countInside = np.sum(insideCircle)

    # Apply Monte Carlo formula
    piEstimate = 4 * (countInside / numPoints)

    return piEstimate


if __name__ == "__main__":
    
    for points in [100, 1000, 10000, 100000, 1000000]:
        piValue = q4(points)
        print(f"Using {points} points: π ≈ {piValue}")