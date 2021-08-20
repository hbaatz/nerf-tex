import tensorflow as tf
#import tensorflow_probability as tfp

def interpolate_img(x: tf.Tensor, y_ref: tf.Tensor) -> tf.Tensor:
    """Bilinear interpolation of an image at pixel-coordinates given by x in [0, height-1]x[0, width-1] with zero padding outside of that range."""

    # Get pixel indices of x
    idxs_00 = tf.cast(x, tf.int32)
    idxs_10 = idxs_00 + [1,0]
    idxs_01 = idxs_00 + [0,1]
    idxs_11 = idxs_00 + [1,1]

    # Get interpolation weights
    weights = x - tf.math.floor(x)

    # Compute the interpolated values
    y_interp = tf.gather_nd(y_ref, idxs_00) * (1 - weights[:,:1]) * (1 - weights[:,1:]) \
               + tf.gather_nd(y_ref, idxs_10) * weights[:,:1] * (1 - weights[:,1:]) \
               + tf.gather_nd(y_ref, idxs_01) * (1 - weights[:,:1]) * weights[:,1:] \
               + tf.gather_nd(y_ref, idxs_11) * weights[:,:1] * weights[:,1:]

    return y_interp

def interpolate_nd(x: tf.Tensor, y_ref: tf.Tensor) -> tf.Tensor:
    """Multilinear interpolation of data y_ref on a regular grid covering [0,1]^(rank(y_ref)-1) at positions x."""
    
    # Get the grid dimension
    dim = tf.rank(y_ref) - 1

    # Scale up x so that whole numbers lay on the grid points
    x_scaled = x * (tf.constant(y_ref.shape[:-1].as_list(), dtype=tf.float32) - 1)

    # Get grid indices of x
    idxs = tf.cast(x_scaled, tf.int32)

    # Get interpolation weights
    weights = x_scaled - tf.math.floor(x_scaled)

    # Get list of all binary vectors of length dim (i.e. the vertices of a grid cell)
    def dec2bin(d):
        return tf.math.mod(tf.bitwise.right_shift([d], tf.range(dim)), 2)

    vertices = tf.map_fn(dec2bin, tf.range(2 ** dim))

    # Compute interpolated value by looping (implicitly via tf.foldl) over grid cell vertices
    def vertex_map(vertex):
        weight_prod = tf.reduce_prod(tf.where(tf.cast(vertex, tf.bool), weights, 1 - weights), axis=1, keepdims=True)
        return tf.gather_nd(y_ref, idxs + vertex) * weight_prod

    interp = tf.foldl(lambda i, v: i + vertex_map(v), vertices[1:], initializer=vertex_map(vertices[0]))

    return interp

# def interpolate_nd_ref(x: tf.Tensor, y_ref: tf.Tensor) -> tf.Tensor:
#     """Multilinear interpolation of data on a regular grid covering [0,1]^(rank(y_ref)-1)."""
#     # The method above is considerably faster, this is just for reference.

#     # Get the dimension of the grid
#     grid_dim = x.shape[-1]
#     data_dim = len(y_ref.shape) - grid_dim

#     # Prepare data for tfp call
#     x_ref_min = tf.zeros((grid_dim,), dtype=tf.float32)
#     x_ref_max = tf.ones((grid_dim,), dtype=tf.float32)

#     return tfp.math.batch_interp_regular_nd_grid(x, x_ref_min, x_ref_max, y_ref, -(grid_dim + data_dim))

def gaussian_kernel(size: int, std: float, channels: int=3):
    """Build gaussian kernel."""

    x = tf.linspace(-(size - 1) / 2, (size - 1) / 2, size) + (.5 if size % 2 == 0 else 0)
    kernel_1d = tf.math.exp(-.5 * (x / std) ** 2)
    kernel_2d = tf.tensordot(kernel_1d, kernel_1d, axes=0)
    kernel_2d /= tf.reduce_sum(kernel_2d)

    return tf.repeat(kernel_2d[:,:,None,None], channels, axis=2)

def filtered_downsample(img, downsampling_factor, std=.5):
    """Lowpass filter and downsample image."""

    kernel = gaussian_kernel(int(downsampling_factor * std * 6), downsampling_factor * std, img.shape[-1])

    return tf.nn.depthwise_conv2d(img[None,:], kernel, strides=[1,downsampling_factor,downsampling_factor,1], padding='SAME')[0]