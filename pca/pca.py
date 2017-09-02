import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Functions:
def batching(image, image_size, window_size):
    batch_per_row = image_size - window_size + 1
    batch_per_col = image_size - window_size + 1
    number_of_batches = batch_per_col * batch_per_row
    number_of_channel = 3
    batches = np.zeros([num_total_images, number_of_batches, window_size, window_size, number_of_channel])

    for image_index in range(num_total_images):
        cur_batch = 0
        for i in range(batch_per_col):
            for j in range(batch_per_row):
                batches[image_index][cur_batch][:][:][:] = image[image_index][i : (i + window_size), j : (j + window_size), :]
                cur_batch += 1
    return batches

def view_batches(batches, batch_index):
    print(batches[batch_index])
    plt.title(None)
    plt.imshow(batches[batch_index])
    plt.show()

# Hard-coded function that can plot 9 images in a row
def plot_9_images():
    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)

    # Adjust vertical spacing if we need to print ensemble and best-net.
    hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Interpolation type.
        interpolation = 'nearest'

        # Plot image.
        ax.imshow(batches[50,i+10, :, :, :],
                  interpolation=interpolation)

        xlabel = 'x'

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


# Define global variables

# Width and height of each image.
img_size = 32

num_classes = 10

# number of images in MCL10, which is total image divided by 10
num_per_class = 30

num_total_images = num_classes * num_per_class

# Load the data
image_data_mcl = np.fromfile('/Users/erichsieh/Desktop/USC/2017_Summer/MCL10/tflearn/MCL-10/MCL10_dat/imgData_300.dat', dtype=np.uint8)
label_data_mcl = np.loadtxt('/Users/erichsieh/Desktop/USC/2017_Summer/MCL10/tflearn/MCL-10/MCL10_dat/labelData_300.txt', dtype=np.int64)

image_data_mcl = image_data_mcl.reshape([int((image_data_mcl.shape)[0]/(img_size*img_size*3)), img_size, img_size, 3])
image_data_mcl = image_data_mcl.astype(np.float64)/255

batches = batching(image_data_mcl, img_size, 8)

batches = batches.reshape([batches.shape[0] * batches.shape[1],batches.shape[2] * batches.shape[3] * batches.shape[4]])

pca = PCA(n_components = 75)
pca.fit(batches)

print(pca.explained_variance_ratio_)
print(pca.singular_values_)
# batches = batches.reshape
