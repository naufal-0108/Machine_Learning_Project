import numpy as np
import tensorflow as tf
from tensorflow import math
from tensorflow.keras.layers import Layer, Dense, Dropout, Embedding, LayerNormalization
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

def positional_encoder(seq_length: int, embed_dim: int) -> tf.Tensor:
    position = tf.range(seq_length, dtype=tf.float32)[:, tf.newaxis]
    div_term = math.exp(tf.range(0, embed_dim, 2, dtype=tf.float32)[tf.newaxis, :] * -(math.log(10000.0)/embed_dim))
    pos_encoding = tf.zeros((seq_length, embed_dim), dtype=np.float32).numpy()
    pos_encoding[:, 0::2] = np.sin(position * div_term)
    pos_encoding[:, 1::2] = np.cos(position * div_term)
    return tf.constant(pos_encoding, dtype=tf.float32, shape=(seq_length, embed_dim))

class EmbeddingLayerCustom(Layer):
    def __init__(self, vocab_size:int, embed_dim: int, max_len:int, dropout_rate=0.1, initializer="normal"):
        super(EmbeddingLayerCustom, self).__init__()

        assert initializer in ["normal", "uniform"], "Initializer must either normal or uniform."

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        if initializer == "normal":
            self.initializer = tf.random_normal_initializer(mean=0., stddev=0.02)

        else:
            self.initializer = tf.random_uniform_initializer(minval=-0.01, maxval=0.01)

        self.embedding = Embedding(vocab_size, embed_dim, embeddings_initializer=self.initializer)
        self.dropout = Dropout(rate=dropout_rate)
        self.pos_encoding = positional_encoder(max_len, embed_dim)

    def _create_mask(self, x):
        mask = tf.cast(tf.not_equal(x, 0), tf.float32)[:, :, tf.newaxis]
        return mask

    def call(self, x, mask_zero=False):
        x_shape = tf.shape(x)
        token_embedding = self.embedding(x)
        token_embedding *= tf.math.sqrt(tf.cast(self.embed_dim, tf.float32))
        token_embedding += self.pos_encoding[tf.newaxis, :x_shape[1], :]

        if mask_zero:
            mask = self._create_mask(x)
            token_embedding *= mask

        return self.dropout(token_embedding)


class MultiHeadSelfAttention(Layer):
    def __init__(self, num_heads: int, embed_dim: int, n: int, dropout_rate=0.1, initializer="normal"):
        super(MultiHeadSelfAttention, self).__init__()

        assert initializer in ["normal", "uniform"], "Initializer must be either normal or uniform."
        assert embed_dim%num_heads == 0, "Number of head must devide embedding dim without remainder."

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim//num_heads

        if initializer == "normal":
            self.initializer = tf.random_normal_initializer(mean=0., stddev=0.02)

        else:
            self.initializer = tf.random_uniform_initializer(minval=-0.01, maxval=0.01)

        self.c_attn = Dense(3*embed_dim, kernel_initializer=self.initializer, use_bias=False)

        self.attn_proj = Dense(embed_dim, kernel_initializer=tf.random_normal_initializer(mean=0., stddev=0.02/math.sqrt(2.*n)), use_bias=True)

        self.attn_dropout = Dropout(rate=dropout_rate)
        self.proj_dropout = Dropout(rate=dropout_rate)

        self.layer_norm = LayerNormalization(epsilon=1e-5)

    def call(self, context, pad_mask=None):

        context_shape = tf.shape(context)

        resid = context

        q, k, v = tf.split(self.c_attn(context), 3, axis=-1)

        q_reshape = tf.reshape(q, shape=(context_shape[0], context_shape[1], self.num_heads, self.head_dim))
        k_reshape = tf.reshape(k, shape=(context_shape[0], context_shape[1], self.num_heads, self.head_dim))
        v_reshape = tf.reshape(v, shape=(context_shape[0], context_shape[1], self.num_heads, self.head_dim))

        q_transposed = tf.transpose(q_reshape, perm=[0, 2, 1, 3])
        k_transposed = tf.transpose(k_reshape, perm=[0, 2, 1, 3])
        v_transposed = tf.transpose(v_reshape, perm=[0, 2, 1, 3])

        w_qk = tf.matmul(q_transposed, k_transposed, transpose_b=True)
        w_qk_normalized = w_qk * (1./math.sqrt(tf.cast(self.head_dim, tf.float32)))

        if pad_mask is not None:
            padding_mask = pad_mask[:, tf.newaxis, tf.newaxis, :]
            w_qk_normalized = w_qk_normalized*padding_mask - ((1-padding_mask) * tf.cast(1e10, dtype=w_qk_normalized.dtype))


        w_qk_normalized_softmax = tf.nn.softmax(w_qk_normalized, axis=-1)
        w_qk_normalized_softmax_drop = self.attn_dropout(w_qk_normalized_softmax)
        w_qkv = tf.matmul(w_qk_normalized_softmax_drop, v_transposed)
        w_qkv_reshape = tf.reshape(tf.transpose(w_qkv, perm=[0, 2, 1, 3]), shape=context_shape)
        w_qkv_proj = self.attn_proj(w_qkv_reshape)
        w_qkv_proj_drop = self.proj_dropout(w_qkv_proj)

        return self.layer_norm(w_qkv_proj_drop + resid)

class MultiHeadCausalAttention(Layer):
    def __init__(self, num_heads: int, embed_dim: int, n: int, dropout_rate=0.1, initializer="normal"):
        super(MultiHeadCausalAttention, self).__init__()

        assert initializer in ["normal", "uniform"], "Initializer must be either normal or uniform."
        assert embed_dim%num_heads == 0, "Number of head must devide embedding dim without remainder."

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        if initializer == "normal":
            self.initializer = tf.random_normal_initializer(mean=0., stddev=0.02)

        else:
            self.initializer = tf.random_uniform_initializer(minval=-0.01, maxval=0.01)


        self.c_attn = Dense(3*embed_dim, kernel_initializer=self.initializer, use_bias=False)

        self.attn_proj = Dense(embed_dim, kernel_initializer=tf.random_normal_initializer(mean=0., stddev=0.02/math.sqrt(2.*n)), use_bias=True)

        self.attn_dropout = Dropout(rate=dropout_rate)
        self.proj_dropout = Dropout(rate=dropout_rate)

        self.layer_norm = LayerNormalization(epsilon=1e-5)

    def _create_causal_mask(self, row, col, dtype=tf.float32):

        i = tf.range(row)[:, tf.newaxis]
        j = tf.range(col)[tf.newaxis, :]

        return tf.cast(i >= j, dtype=dtype)

    def call(self, target, pad_mask=None):

        target_shape = tf.shape(target)

        resid = target

        causal_mask = self._create_causal_mask(target_shape[1], target_shape[1], dtype=target.dtype)

        q, k, v = tf.split(self.c_attn(target), 3, axis=-1)

        q_reshape = tf.reshape(q, shape=(target_shape[0], target_shape[1], self.num_heads, self.head_dim))
        k_reshape = tf.reshape(k, shape=(target_shape[0], target_shape[1], self.num_heads, self.head_dim))
        v_reshape = tf.reshape(v, shape=(target_shape[0], target_shape[1], self.num_heads, self.head_dim))

        q_transposed = tf.transpose(q_reshape, perm=[0, 2, 1, 3])
        k_transposed = tf.transpose(k_reshape, perm=[0, 2, 1, 3])
        v_transposed = tf.transpose(v_reshape, perm=[0, 2, 1, 3])

        w_qk = tf.matmul(q_transposed, k_transposed, transpose_b=True)
        w_qk_normalized = w_qk * (1./math.sqrt(tf.cast(self.head_dim, tf.float32)))

        if pad_mask is not None:

            padding_mask = pad_mask[:, tf.newaxis, tf.newaxis, :]
            merge_mask = causal_mask * padding_mask

            w_qk_normalized = w_qk_normalized*merge_mask - ((1-merge_mask) * tf.cast(1e10, dtype=w_qk_normalized.dtype))
            w_qk_normalized_softmax = tf.nn.softmax(w_qk_normalized, axis=-1)
            w_qk_normalized_softmax_drop = self.attn_dropout(w_qk_normalized_softmax)

            w_qkv = tf.matmul(w_qk_normalized_softmax_drop, v_transposed)
            w_qkv_reshape = tf.reshape(tf.transpose(w_qkv, perm=[0, 2, 1, 3]), shape=target_shape)
            w_qkv_proj = self.attn_proj(w_qkv_reshape)
            w_qkv_proj_drop = self.proj_dropout(w_qkv_proj)

            return self.layer_norm(w_qkv_proj_drop + resid)

        else:
            w_qk_normalized = w_qk_normalized*causal_mask - ((1-causal_mask) * tf.cast(1e10, dtype=w_qk_normalized.dtype))

            w_qk_normalized_softmax = tf.nn.softmax(w_qk_normalized, axis=-1)
            w_qk_normalized_softmax_drop = self.attn_dropout(w_qk_normalized_softmax)

            w_qkv = tf.matmul(w_qk_normalized_softmax_drop, v_transposed)
            w_qkv_reshape = tf.reshape(tf.transpose(w_qkv, perm=[0, 2, 1, 3]), shape=target_shape)
            w_qkv_proj = self.attn_proj(w_qkv_reshape)
            w_qkv_proj_drop = self.proj_dropout(w_qkv_proj)

            return self.layer_norm(w_qkv_proj_drop + resid)

class MultiHeadCrossAttention(Layer):
    def __init__(self, num_heads: int, embed_dim: int, n: int, dropout_rate=0.1, initializer="normal"):

        super(MultiHeadCrossAttention, self).__init__()

        assert embed_dim%num_heads == 0, "Number of heads must devide embed dim without remainder."
        assert initializer in ["normal", "uniform"], "Initializer must be either normal or uniform."

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads


        if initializer == "normal":
            self.initializer = tf.random_normal_initializer(mean=0., stddev=0.02)

        else:
            self.initializer = tf.random_uniform_initializer(minval=-0.01, maxval=0.01)

        self.wq = Dense(embed_dim, kernel_initializer=self.initializer, use_bias=False)
        self.wk = Dense(embed_dim, kernel_initializer=self.initializer, use_bias=False)
        self.wv = Dense(embed_dim, kernel_initializer=self.initializer, use_bias=False)


        self.attn_proj = Dense(embed_dim, kernel_initializer=tf.random_normal_initializer(mean=0., stddev=0.02/math.sqrt(2.*n)), use_bias=True)

        self.attn_dropout = Dropout(rate=dropout_rate)
        self.proj_dropout = Dropout(rate=dropout_rate)

        self.layer_norm = LayerNormalization(epsilon=1e-5)

    def call(self, target, context, pad_mask=None):

        assert context.shape[1] == pad_mask.shape[1]

        target_shape = tf.shape(target)
        context_shape = tf.shape(context)

        resid = target

        q, k, v = self.wq(target), self.wk(context), self.wv(context)

        q_reshape = tf.reshape(q, shape=(target_shape[0], target_shape[1], self.num_heads, self.head_dim))
        k_reshape = tf.reshape(k, shape=(context_shape[0], context_shape[1], self.num_heads, self.head_dim))
        v_reshape = tf.reshape(v, shape=(context_shape[0], context_shape[1], self.num_heads, self.head_dim))

        q_transposed = tf.transpose(q_reshape, perm=[0, 2, 1, 3])
        k_transposed = tf.transpose(k_reshape, perm=[0, 2, 1, 3])
        v_transposed = tf.transpose(v_reshape, perm=[0, 2, 1, 3])

        w_qk = tf.matmul(q_transposed, k_transposed, transpose_b=True)
        w_qk_normalized = w_qk * (1./ tf.math.sqrt(tf.cast(self.head_dim, tf.float32)))

        if pad_mask is not None:
            padding_mask = pad_mask[:, tf.newaxis, tf.newaxis, :]
            w_qk_normalized = w_qk_normalized * padding_mask - ((1-padding_mask) * tf.cast(1e10, dtype=w_qk_normalized.dtype))

        w_qk_normalized_softmax = tf.nn.softmax(w_qk_normalized, axis=-1)
        w_qk_normalized_softmax_drop = self.attn_dropout(w_qk_normalized_softmax)

        w_qkv = tf.matmul(w_qk_normalized_softmax_drop, v_transposed)
        w_qkv_reshape = tf.reshape(tf.transpose(w_qkv, perm=[0, 2, 1, 3]), shape=target_shape)
        w_qkv_proj = self.attn_proj(w_qkv_reshape)
        w_qkv_proj_drop = self.proj_dropout(w_qkv_proj)

        return self.layer_norm(w_qkv_proj_drop + resid)

class FeedForwardCustom(Layer):
    def __init__(self, embed_dim, inner_dim, dropout_rate=0.1, initializer="normal"):
        super(FeedForwardCustom, self).__init__()

        assert initializer in ["normal", "uniform"], "Initializer must be either normal or uniform."

        self.embed_dim = embed_dim
        self.inner_dim = inner_dim

        if initializer == "normal":
            self.initializer = tf.random_normal_initializer(mean=0., stddev=0.02)

        else:
            self.initializer = tf.random_uniform_initializer(minval=-0.01, maxval=0.01)


        self.fc1 = Dense(inner_dim, kernel_initializer=self.initializer, use_bias=True, activation='gelu')
        self.fc2 = Dense(embed_dim, kernel_initializer=self.initializer, use_bias=True)

        self.dropout = Dropout(rate=dropout_rate)

        self.layer_norm = LayerNormalization(epsilon=1e-6)

    def call(self, x):

        resid = x
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return self.layer_norm(x + resid)

class EncoderBlock(Layer):
    def __init__(self, num_heads: int, embed_dim: int, inner_dim: int, n:int, dropout_rate=0.1, initializer="normal"):
        super(EncoderBlock, self).__init__()

        self.mhsa = MultiHeadSelfAttention(num_heads=num_heads, embed_dim=embed_dim, n=n, dropout_rate=dropout_rate,
                                           initializer=initializer)

        self.ff = FeedForwardCustom(embed_dim=embed_dim, inner_dim=inner_dim, dropout_rate=dropout_rate,
                                    initializer=initializer)

    def call(self, context, pad_mask=None):

        assert context.shape[1] == pad_mask.shape[1]

        mhsa_outs = self.mhsa(context, pad_mask=pad_mask)
        ff_outs = self.ff(mhsa_outs)

        return ff_outs

class DecoderBlock(Layer):
    def __init__(self, num_heads: int, embed_dim: int, inner_dim: int, n: int, dropout_rate=0.1, initializer="normal"):
        super(DecoderBlock, self).__init__()
        self.mhca = MultiHeadCausalAttention(num_heads=num_heads, embed_dim=embed_dim, n=n, dropout_rate=dropout_rate, initializer=initializer)
        self.mhcra = MultiHeadCrossAttention(num_heads=num_heads, embed_dim=embed_dim, n=n, dropout_rate=dropout_rate, initializer=initializer)
        self.ff = FeedForwardCustom(embed_dim=embed_dim, inner_dim=inner_dim, dropout_rate=dropout_rate, initializer=initializer)

    def call(self, target, context, target_pad_mask=None, context_pad_mask=None):


        assert target.shape[1] == target_pad_mask.shape[1]
        assert context.shape[1] == context_pad_mask.shape[1]

        target = self.mhca(target, pad_mask=target_pad_mask)
        target = self.mhcra(target=target, context=context, pad_mask=context_pad_mask)
        target = self.ff(target)

        return target

class Encoder(Layer):
    def __init__(self, vocab_size: int, max_len: int, num_heads: int, embed_dim: int, inner_dim: int, n: int, dropout_rate=0.1, initializer="normal"):
        super(Encoder, self).__init__()

        self.encoder_blocks = [EncoderBlock(num_heads=num_heads, embed_dim=embed_dim, inner_dim=inner_dim, n=n, dropout_rate=dropout_rate, initializer=initializer) for _ in range(n)]

    def call(self, context, pad_mask=None):

        assert context.shape[1] == pad_mask.shape[1]

        for i in range(len(self.encoder_blocks)):
            context = self.encoder_blocks[i](context=context, pad_mask=pad_mask)

        return context

class Decoder(Layer):
    def __init__(self, vocab_size:int, max_len:int, num_heads: int, embed_dim: int, inner_dim: int, n: int, dropout_rate=0.1, initializer="normal"):
        super(Decoder, self).__init__()

        self.decoder_blocks = [DecoderBlock(num_heads=num_heads, embed_dim=embed_dim, inner_dim=inner_dim, n=n, dropout_rate=dropout_rate, initializer=initializer) for _ in range(n)]

    def call(self, target, context, target_pad_mask, context_pad_mask):

        assert target.shape[1] == target_pad_mask.shape[1]
        assert context.shape[1] == context_pad_mask.shape[1]

        for i in range(len(self.decoder_blocks)):
            target = self.decoder_blocks[i](target=target, context=context, target_pad_mask=target_pad_mask, context_pad_mask=context_pad_mask)

        return target

class CustomScheduler(LearningRateSchedule):
  def __init__(self, embed_dim, warmup_steps=4000):
    super(CustomScheduler, self).__init__()
    self.embed_dim = tf.cast(embed_dim, tf.float32)
    self.warmup_steps = warmup_steps

  def __call__(self, step):
    step = tf.cast(step, tf.float32)
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    lr = tf.math.rsqrt(self.embed_dim) * tf.math.minimum(arg1, arg2)
    return lr

  def get_config(self):
    config = {
    'embed_dim': self.embed_dim,
    'warmup_steps': self.warmup_steps,}
    return config

class CreatePadMask():
  def __init__(self, mask_token=0):
    self.mask_token = mask_token
  
  def __call__(self, tokens):

    assert isinstance(tokens, (np.ndarray, tf.Tensor)), "Tokens must be a numpy array or tensor."

    if isinstance(tokens, np.ndarray):
      mask = np.not_equal(tokens, self.mask_token).astype(np.float32)
      return mask

    else:
      mask = tf.cast(tf.not_equal(tokens, self.mask_token), tf.float32)
      mask = tf.constant(mask)
      return mask
