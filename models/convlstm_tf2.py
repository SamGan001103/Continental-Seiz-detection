"""The detector graph, built with ``tf.keras``, for TensorFlow 2 runtimes.

`models/deep_conv_lstm.py` is the original authors' file and is kept untouched:
it is the code that generated the training features and the released weights,
and it is what the reproduction argument refers to. It also cannot run on a
modern stack — it imports `keras.layers.core` and calls
`Model(input=…, output=…)`, both removed after Keras 2.5.

This module builds the identical graph through `tf.keras`, so the same
`convlstm_ICA_12_train.h5` loads on Python 3.11 with TensorFlow 2. That is what
makes the application runnable on Apple Silicon, where TensorFlow 1.15 has no
wheel and never will.

Verified against the legacy builder: same 384,846 parameters, and predictions
agreeing to **6.8 × 10⁻⁹** on real ICA'd windows — float32 operation-ordering
noise, with no window crossing the detection threshold. See
`docs/portability.md` and `experiments/diag_tf2_port.py`.

Every layer parameter is copied verbatim from `deep_conv_lstm.py`. Changing any
of them changes which weights land where, and `load_weights` would either raise
or — worse — silently mis-assign.
"""
from gui.io.edf import CHANNELS_19


def build_convlstm(n_time=23, n_electrodes=None, n_freq=125, n_classes=2):
    """Return the uncompiled model, matching Fig. 4 of the source paper.

    Input ``(n_time, n_electrodes, n_freq, 1)`` — 23 × 19 × 125 × 1 as shipped.
    """
    from tensorflow.keras.layers import (
        Activation, BatchNormalization, ConvLSTM2D, Dense, Dropout, Flatten,
        Input, Lambda)
    from tensorflow.keras.models import Model

    if n_electrodes is None:
        n_electrodes = len(CHANNELS_19)

    inputs = Input(shape=(n_time, n_electrodes, n_freq, 1))
    x = BatchNormalization(axis=2, name='normal1')(inputs)

    # The first kernel spans every electrode, so it is (n_electrodes, 3).
    x = ConvLSTM2D(filters=16, kernel_size=(n_electrodes, 3), padding='valid',
                   strides=(1, 2), activation='tanh', dropout=0.0,
                   recurrent_dropout=0.0, return_sequences=True,
                   name='convlstm1')(x)
    x = ConvLSTM2D(filters=32, kernel_size=(1, 3), padding='valid',
                   strides=(1, 2), activation='tanh', dropout=0.0,
                   recurrent_dropout=0.0, return_sequences=True,
                   name='convlstm2')(x)
    x = ConvLSTM2D(filters=64, kernel_size=(1, 3), padding='valid',
                   strides=(1, 2), activation='tanh', dropout=0.0,
                   recurrent_dropout=0.0, return_sequences=False,
                   name='convlstm3')(x)

    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='sigmoid', name='dens1')(x)
    x = Dropout(0.5)(x)
    x = Dense(n_classes, name='dens2')(x)
    # The original applies a temperature of 1.0 before the softmax. It is a
    # no-op numerically, and kept so the layer count and graph shape match.
    x = Lambda(lambda t: t / 1.0)(x)
    outputs = Activation('softmax')(x)

    return Model(inputs=inputs, outputs=outputs)


def is_available():
    """True when a TensorFlow 2 style ``tf.keras`` can be imported."""
    try:
        import tensorflow as tf
        return int(str(tf.__version__).split('.')[0]) >= 2
    except Exception:
        return False
