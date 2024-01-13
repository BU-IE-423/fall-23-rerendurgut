# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, expon, gamma

# Load the image
image_path = "/Users/receperendurgut/Desktop/0100.jpg"
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
pixel_values = original_image.flatten()

# Display the histogram
plt.hist(pixel_values, bins=256, range=[0, 256], density=True, color='gray', alpha=0.7)
plt.title('Pixel Value Distribution Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Probability Density')

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, original_image.size)

# Plot the fitted normal distribution
mu, sigma = norm.fit(pixel_values)
p_norm = norm.pdf(x, mu, sigma)
plt.plot(x, p_norm, 'k', linewidth=2, label='Normal')

# Plot the fitted gamma distribution
a, loc, scale = gamma.fit(pixel_values)
pdf_gamma = gamma.pdf(x, a, loc, scale)
plt.plot(x, pdf_gamma, 'r', linewidth=2, label=f'Gamma')

# Plot the fitted exponential distribution
loc_expon, scale_expon = expon.fit(pixel_values)
p_expon = expon.pdf(x, loc_expon, scale_expon)
plt.plot(x, p_expon, 'b', linewidth=2, label='Exponential')

plt.legend()

plt.show()

# Output for distributions
print(f"Normal Distribution Parameters: mu: {mu}, sigma: {sigma}")
print(f"Gamma Distribution Parameters: loc: {loc}, scale: {scale}")
print(f"Exponential Distribution Parameters: loc: {loc_expon}, scale: {scale_expon}")

# %%
a, loc, scale = gamma.fit(pixel_values)

# Define the lower and upper bounds based on the specified probability limits
lower_bound = gamma.ppf(0.001, a, loc, scale)
upper_bound = gamma.ppf(0.999, a, loc, scale)

# Create a separate array to store modified pixel values
modified_pixel_values = pixel_values.copy()

# Identify pixels outside the bounds and set their values to zero
for i in range(len(modified_pixel_values)):
    if modified_pixel_values[i] < lower_bound:
        print(modified_pixel_values[i])
        modified_pixel_values[i] = 0
    elif modified_pixel_values[i] > upper_bound:
        print(modified_pixel_values[i])
        modified_pixel_values[i] = 0
    else:
        None
                
# Reshape the modified pixel values back to the original image shape
modified_image = modified_pixel_values.reshape(original_image.shape)

# Display the original and modified images
plt.subplot(121)
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')

plt.subplot(122)
plt.imshow(modified_image, cmap='gray')
plt.title('Modified Image (Outliers set to zero)')

plt.show()

# Output observations
print(f"Gamma Distribution Parameters: a={a}, loc={loc}, scale={scale}")
print(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")

# %%
# Define window size
window_size = 51

# Function to identify and modify outlier pixels within patches
# (get help from GPT to understand what to do exactly. also the part used to change pixels didn't work for this part so had an assist to come up with an output *np.where* part)

def process_patches(image, window_size):
    modified_image = image.copy()
    a, loc, scale = gamma.fit(image.flatten())
    
    for i in range(0, image.shape[0] - window_size + 1, window_size):
        for j in range(0, image.shape[1] - window_size + 1, window_size):
            patch = image[i:i+window_size, j:j+window_size].flatten()
            lower_bound = gamma.ppf(0.001, a, loc, scale)
            upper_bound = gamma.ppf(0.999, a, loc, scale)
            outlier_pixels = np.where((patch < lower_bound) | (patch > upper_bound))[0]
            patch[outlier_pixels] = 0
            modified_image[i:i+window_size, j:j+window_size] = patch.reshape((window_size, window_size))
    
    return modified_image

# Process patches
modified_image_patches = process_patches(original_image, window_size)

# Display the original and modified images
plt.subplot(121)
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')

plt.subplot(122)
plt.imshow(modified_image_patches, cmap='gray')
plt.title('Modified Image (Outliers set to zero in patches)')

plt.show()

# %%
def control_chart_operation(image, axis='row'):
    modified_image = image.copy()

    if axis == 'row':
        statistics = np.mean(image, axis=1), np.var(image, axis=1)
    elif axis == 'column':
        statistics = np.mean(image, axis=0), np.var(image, axis=0)
    else:
        raise ValueError("Invalid axis. Use 'row' or 'column'.")

    mean_values, var_values = statistics
    sigma = 3

    for i in range(image.shape[0] if axis == 'row' else image.shape[1]):
        lower_bound = mean_values[i] - sigma * np.sqrt(var_values[i])
        upper_bound = mean_values[i] + sigma * np.sqrt(var_values[i])

        out_of_control_pixels = np.where((image[i, :] < lower_bound) | (image[i, :] > upper_bound))[0]

        # Set out-of-control pixels to zero
        modified_image[i, out_of_control_pixels] = 0

    return modified_image, mean_values, var_values

# Perform operations on rows
modified_image_rows, mean_rows, var_rows = control_chart_operation(original_image, axis='row')

# Perform operations on columns
modified_image_columns, mean_columns, var_columns = control_chart_operation(original_image, axis='column')


plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["lines.linewidth"] = 2.0  # Line thickness
plt.rcParams["lines.markersize"] = 8.0  # Marker size

# Display the original and modified images for rows
plt.subplot(331)
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')

plt.subplot(332)
plt.plot(mean_rows, label='Mean')
plt.title('Row-wise Mean')
plt.axhline(np.mean(mean_rows) + 3 * np.std(mean_rows), color='r', linestyle='--', label='Upper Control Limit')
plt.axhline(np.mean(mean_rows) - 3 * np.std(mean_rows), color='g', linestyle='--', label='Lower Control Limit')
plt.legend()

plt.subplot(333)
plt.plot(var_rows, label='Variance')
plt.title('Row-wise Variance')
plt.legend()

plt.subplot(334)
plt.imshow(modified_image_rows, cmap='gray')
plt.title('Modified Image (Rows)')

# Display the original and modified images for columns
plt.subplot(335)
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')

plt.subplot(336)
plt.plot(mean_columns, label='Mean')
plt.title('Column-wise Mean')
plt.axhline(np.mean(mean_columns) + 3 * np.std(mean_columns), color='r', linestyle='--', label='Upper Control Limit')
plt.axhline(np.mean(mean_columns) - 3 * np.std(mean_columns), color='g', linestyle='--', label='Lower Control Limit')
plt.legend()

plt.subplot(337)
plt.plot(var_columns, label='Variance')
plt.title('Column-wise Variance')
plt.legend()

plt.subplot(338)
plt.imshow(modified_image_columns, cmap='gray')
plt.title('Modified Image (Columns)')

plt.tight_layout()
plt.show()


