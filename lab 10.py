import numpy as np

# Set clean print format (2 decimal places)
np.set_printoptions(precision=2, suppress=True)

# Step 1: Simple dataset
data = np.array([
    [2, 0],
    [0, 2],
    [3, 1]
])

print("Original Data:")
for row in data:
    print(f"[{row[0]:.2f}, {row[1]:.2f}]")

# Step 2: Mean
mean = np.mean(data, axis=0)
print("\nMean:")
print(f"[{mean[0]:.2f}, {mean[1]:.2f}]")

# Step 3: Center data
centered_data = data - mean
print("\nCentered Data:")
for row in centered_data:
    print(f"[{row[0]:.2f}, {row[1]:.2f}]")

# Step 4: Covariance matrix
cov_matrix = np.cov(centered_data.T)
print("\nCovariance Matrix:")
for row in cov_matrix:
    print(f"[{row[0]:.2f}, {row[1]:.2f}]")

# Step 5: Eigenvalues & Eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

print("\nEigenvalues:")
for val in eigenvalues:
    print(f"{val:.2f}")

print("\nEigenvectors:")
for row in eigenvectors:
    print(f"[{row[0]:.2f}, {row[1]:.2f}]")

# Step 6: Sort eigenvalues (descending)
idx = np.argsort(eigenvalues)[::-1]
eigenvectors = eigenvectors[:, idx]

# Select first principal component
pc1 = eigenvectors[:, 0]
print("\nPrincipal Component (PC1):")
print(f"[{pc1[0]:.2f}, {pc1[1]:.2f}]")

# Step 7: Transform data
transformed_data = centered_data.dot(pc1)

print("\nFinal Reduced Data (1D):")
for val in transformed_data:
    print(f"{val:.2f}")