import tensorflow as tf
from ops import avg_downsample

"""
Interface: create and sample neural textures
"""
def create_neural_texture(args):
    if args.mipmap:
        neural_texture = create_neural_texture_hierarchy_mipmap(args.texture_size, args.texture_channels, args.texture_levels, args.texture_init)
    else:
        neural_texture = create_neural_texture_hierarchy(args.texture_size, args.texture_channels, args.texture_levels, args.texture_init)
    return neural_texture

def sample_texture(neural_texture, uv, args):
    if args.mipmap:
        sampled_texture = sample_texture_hierarchy_mipmap(neural_texture, uv)
    else:
        sampled_texture = sample_texture_hierarchy(neural_texture, uv)
    return sampled_texture


"""
Internal implementation
"""

def _uv2index(uvs, shape, mode='clamp', flip_y = False):
    u = uvs[...,0:1]
    v = uvs[...,1:2]
    
    if flip_y:
        v = (tf.ones_like(v)  - v )

    uvs = tf.concat([v,u], axis = -1)
    
    texture_shape = tf.cast(shape, tf.float32)
    if mode == 'repeat':
        return uvs % 1. * texture_shape
    elif mode == 'clamp':
        return tf.clip_by_value(uvs, 0., (texture_shape-1.0 - 1e-3)/texture_shape) * texture_shape
    else:
        raise NotImplementedError


def _sample_texture(texture, indices, mode='bilinear', resolution=512):
   
    if mode == 'nearest':
        return tf.gather_nd(texture, tf.cast(indices, tf.int32))   
      
    elif mode == 'bilinear':
        X = indices[..., 0:1] - 0.5
        Y = indices[..., 1:2] - 0.5
     
        update_indices = tf.concat([X,Y], axis = -1)

        X0 = tf.maximum(tf.floor(X), 0)
        X1 = tf.minimum(tf.ceil(X), resolution-1)
        Y0 = tf.maximum(tf.floor(Y), 0)
        Y1 = tf.minimum(tf.ceil(Y), resolution-1)

        X0 = tf.cast(X0, tf.int32)
        X1 = tf.cast(X1, tf.int32)
        Y0 = tf.cast(Y0, tf.int32)
        Y1 = tf.cast(Y1, tf.int32)
        
        frac_indices = update_indices - tf.floor(update_indices)

        top_left = tf.gather_nd(texture, tf.concat([X0,Y0], axis=-1))
        top_right = tf.gather_nd(texture,  tf.concat([X0,Y1], axis=-1))
        bottom_left = tf.gather_nd(texture,  tf.concat([X1,Y0], axis=-1))
        bottom_right = tf.gather_nd(texture,  tf.concat([X1,Y1], axis=-1))

        return \
            top_left * (1. - frac_indices[..., 1:]) * (1. - frac_indices[..., :1]) + \
            top_right * frac_indices[..., 1:] * (1. - frac_indices[..., :1]) + \
            bottom_left * (1. - frac_indices[..., 1:]) * frac_indices[..., :1] + \
            bottom_right * frac_indices[..., 1:] * frac_indices[..., :1]
    else:
        raise NotImplementedError

def create_neural_texture_single(resolution = 512, channels = 16, init_type="glorot_uniform"):
    shape = (resolution, resolution, channels)
    init = tf.initializers.glorot_uniform()

    if init_type == "normal":
        init = tf.initializers.random_normal()
    elif init_type == "glorot_uniform":
        pass
    elif init_type == "zeros":
        init = tf.initializers.zeros()
    elif init_type == "ones":
        init = tf.initializers.ones()
    elif init_type == "uniform":
        init = tf.initializers.uniform_unit_scaling()

    neural_texture = tf.get_variable(name='neural_texture', shape = shape, dtype=tf.float32, trainable=True, initializer=init)
    return neural_texture

def sample_texture_single(texture, uvs, mode = 'bilinear'):
    indices = _uv2index(uvs, tf.shape(texture)[0:2])
    results = []
    batch_size = uvs.get_shape().as_list()[0]
    for i in range(batch_size):
        result = _sample_texture(texture, indices[i], mode)
        results.append(result)    
    return tf.stack(results)


def create_neural_texture_hierarchy(resolution = 512, channels = 16, levels = 4, init_type="glorot_uniform"):
    neural_texture_hierarchy = []
    
    for level in range(levels):
        level_resolution = int(resolution / (2 ** level))
        shape = (level_resolution, level_resolution, channels)

        init = tf.initializers.glorot_uniform()

        if init_type == "normal":
            init = tf.initializers.random_normal()
        elif init_type == "glorot_uniform":
            pass
        elif init_type == "zeros":
            init = tf.initializers.zeros()
        elif init_type == "ones":
            init = tf.initializers.ones()
        elif init_type == "uniform":
            init = tf.initializers.uniform_unit_scaling()

        neural_texture_hierarchy.append(tf.get_variable(name='neural_texture_%d' % level, shape = shape, dtype=tf.float32, trainable=True, initializer=init))

    return neural_texture_hierarchy

def sample_texture_hierarchy(textures, uvs, mode = 'bilinear'):
    result = None

    for i in range(len(textures)):
        sampled = sample_texture_single(textures[i], uvs, mode=mode)
        if i == 0:
            result = sampled
        else:
            result += sampled
    
    result /= len(textures)
    
    return result


def create_neural_texture_hierarchy_mipmap(resolution = 512, channels = 16, levels = 4, init_type="glorot_uniform"):
    neural_texture_hierarchy = []
    shape = (resolution, resolution, channels)

    neural_texture_hierarchy.append(tf.get_variable(name='neural_texture_%d' % 0, shape = shape, dtype=tf.float32, trainable=True, initializer=tf.initializers.glorot_uniform()))

    for level in range(1, levels):
        t = avg_downsample(neural_texture_hierarchy[-1])
        neural_texture_hierarchy.append(t)

    return neural_texture_hierarchy

def sample_texture_hierarchy_mipmap(textures, uvs):
    def dot(a,b):
        return tf.reduce_sum(a * b, axis=-1, keepdims=True)

    levels = len(textures)
    resolution = tf.shape(textures[0])[0:2]
    resolution = tf.cast(resolution, tf.float32)
    channels = textures[0].get_shape().as_list()[2]

    duvdy, duvdx= tf.image.image_gradients(uvs * resolution)
    
    # https://www.khronos.org/registry/OpenGL/specs/gl/glspec42.core.pdf#page=262&zoom=100,0,230
    # https://community.khronos.org/t/mipmap-level-calculation-using-dfdx-dfdy/67480#post1236952
    # Eq-3.21
    mipmap_level = 0.5 * tf.log(tf.maximum(dot(duvdx, duvdx), dot(duvdy, duvdy)) ) / tf.log(2.0)
    mipmap_weight_lod_list = []
    
    mipmap_weight_lod0 = tf.zeros_like(mipmap_level)
    mipmap_weight_lod0 = tf.where(tf.less_equal(mipmap_level, 0), tf.ones_like(mipmap_level), mipmap_weight_lod0)
    mipmap_weight_lod0 = tf.where(tf.math.logical_and(tf.greater(mipmap_level, 0), tf.less_equal(mipmap_level, 1)), 1.0 - mipmap_level, mipmap_weight_lod0)
    mipmap_weight_lod0 = tf.concat([mipmap_weight_lod0] * channels, axis = -1)
    mipmap_weight_lod_list.append(mipmap_weight_lod0)

    for l in range(1, levels-1):
        mipmap_weight_lodl = tf.zeros_like(mipmap_level)
        mipmap_weight_lodl = tf.where(tf.math.logical_and(tf.greater(mipmap_level, l-1), tf.less_equal(mipmap_level, l)), mipmap_level - (l-1.0), mipmap_weight_lodl) 
        mipmap_weight_lodl = tf.where(tf.math.logical_and(tf.greater(mipmap_level, l), tf.less_equal(mipmap_level, l+1)), l + 1.0 - mipmap_level, mipmap_weight_lodl)
        mipmap_weight_lodl = tf.concat([mipmap_weight_lodl] * channels, axis = -1)
        mipmap_weight_lod_list.append(mipmap_weight_lodl)

    mipmap_weight_lod_last = tf.zeros_like(mipmap_level)
    mipmap_weight_lod_last = tf.where(tf.math.logical_and(tf.greater(mipmap_level, levels-2), tf.less_equal(mipmap_level, levels-1)), mipmap_level -  (levels - 2.0), mipmap_weight_lod_last) 
    mipmap_weight_lod_last = tf.where(tf.greater(mipmap_level, levels-1.0), tf.ones_like(mipmap_level), mipmap_weight_lod_last)
    mipmap_weight_lod_last = tf.concat([mipmap_weight_lod_last] * channels, axis = -1)
    mipmap_weight_lod_list.append(mipmap_weight_lod_last)

    mipmap_weight = tf.convert_to_tensor(mipmap_weight_lod_list, dtype=tf.float32)
  
    sampled_list = []
    for i in range(levels):
        sampled = sample_texture_single(textures[i], uvs, mode="bilinear")
        sampled_list.append(sampled) 
            
    sampled_list = tf.convert_to_tensor(sampled_list, dtype=tf.float32)
    result = sampled_list * mipmap_weight 

    result = tf.reduce_sum(result, axis = 0)
    return result 


