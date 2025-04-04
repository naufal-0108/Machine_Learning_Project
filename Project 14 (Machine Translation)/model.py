import numpy as np
import tensorflow as tf
from utils import EmbeddingLayerCustom, Encoder, Decoder
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model

class Transformer(Model):
  def __init__(self, context_num_vocabs: int, context_max_len: int, target_num_vocabs: int, target_max_len: int, num_heads: int, embed_dim: int, inner_dim: int, n: int, dropout_rate: float, initializer: str):
    super(Transformer, self).__init__()
    self.context_embedding = EmbeddingLayerCustom(vocab_size=context_num_vocabs, embed_dim=embed_dim, max_len=context_max_len, dropout_rate=dropout_rate, initializer=initializer)
    self.target_embedding = EmbeddingLayerCustom(vocab_size=target_num_vocabs, embed_dim=embed_dim, max_len=target_max_len, dropout_rate=dropout_rate, initializer=initializer)
    self.encoder = Encoder(vocab_size=context_num_vocabs, max_len=context_max_len, num_heads=num_heads, embed_dim=embed_dim, inner_dim=inner_dim, n=n, dropout_rate=dropout_rate, initializer=initializer)
    self.decoder = Decoder(vocab_size=target_num_vocabs, max_len=target_max_len, num_heads=num_heads, embed_dim=embed_dim, inner_dim=inner_dim, n=n, dropout_rate=dropout_rate, initializer=initializer)
    self.fc = Dense(target_num_vocabs, use_bias=False, kernel_initializer=tf.random_normal_initializer(mean=0., stddev=0.02))

  def call(self, inputs):
    context, target, context_mask, target_mask = inputs

    context_embedding_outs = self.context_embedding(context, mask_zero=True)
    target_embedding_outs = self.target_embedding(target, mask_zero=True)

    encoder_outs = self.encoder(context=context_embedding_outs, pad_mask=context_mask)
    decoder_outs = self.decoder(target=target_embedding_outs, context=encoder_outs, target_pad_mask=target_mask, context_pad_mask=context_mask)
    logits = self.fc(decoder_outs)

    return logits
