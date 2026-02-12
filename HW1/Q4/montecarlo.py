import numpy as np

def approximate_pi(num_points):
    # Generate random x and y values between -1 and 1
    x = np.random.uniform(-1, 1, num_points)
    y = np.random.uniform(-1, 1, num_points)

    # Check which points fall inside the unit circle
    inside_circle = (x**2 + y**2) <= 1

    # Count how many are inside
    count_inside = np.sum(inside_circle)

    # Apply Monte Carlo formula
    pi_estimate = 4 * (count_inside / num_points)

    return pi_estimate


# Test different sample sizes
for points in [100, 1000, 10000, 100000, 1000000]:
    pi_value = approximate_pi(points)
    print(f"Using {points} points: π ≈ {pi_value}")