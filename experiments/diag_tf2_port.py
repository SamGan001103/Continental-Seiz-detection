"""Can the released weights run under TensorFlow 2 / tf.keras?

This is the question that decides whether the application can ever run on Apple
Silicon. Checked against PyPI, every dependency has a macOS arm64 wheel —
numpy, scipy, scikit-learn, PyQt5, pyEDFlib — and MNE and pyqtgraph are pure
Python. **TensorFlow 1.15 is the only blocker**: it has no arm64 wheel at any
Python version, and never will, because it predates the platform.

So the migration reduces to one question: does `convlstm_ICA_12_train.h5`, saved
by Keras 2.2.5, load into an equivalent `tf.keras` model and produce the same
numbers? If yes, the Mac path is open and the remaining work is ordinary
packaging. If no, it is a research problem.

    # tensorflow 2.6 is the last release supporting Python 3.6, so this can be
    # tested inside seiz36 without building a second environment
    python -m pip install --no-deps --target /tmp/tf26 tensorflow==2.6.2 keras==2.6.0 ...

    python experiments/diag_tf2_port.py --inputs tfx.npz --out tf2.npz --lib-path /tmp/tf26
    python experiments/diag_tf_version.py --compare tf_a.npz tf2.npz

The architecture is rebuilt here rather than imported from
`models/deep_conv_lstm.py`, because that module uses the Keras 1 style
`Model(input=…, output=…)` and `keras.layers.normalization`, both removed by
Keras 2.6. Rebuilding it is exactly the porting work under test — the layer
parameters are copied verbatim from the original so the weight shapes, names
and ordering match.
"""
from __future__ import print_function

import argparse
import os
import sys


def build(shape=(23, 19, 125, 1), n_classes=2):
    """The same graph as models/deep_conv_lstm.py, in tf.keras."""
    from tensorflow.keras.layers import (
        Input, BatchNormalization, ConvLSTM2D, Flatten, Dropout, Dense,
        Lambda, Activation)
    from tensorflow.keras.models import Model

    inputs = Input(shape=shape)
    x = BatchNormalization(axis=2, name='normal1')(inputs)
    x = ConvLSTM2D(filters=16, kernel_size=(shape[1], 3), padding='valid',
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
    x = Lambda(lambda t: t / 1.0)(x)
    out = Activation('softmax')(x)
    return Model(inputs=inputs, outputs=out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--lib-path', default=None)
    ap.add_argument('--inputs', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)

    if args.lib_path:
        sys.path.insert(0, os.path.abspath(args.lib_path))
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if repo not in sys.path:
        sys.path.append(repo)

    import numpy as np
    import tensorflow as tf
    print('TensorFlow {}'.format(tf.__version__))

    from gui.paths import weights_path
    model = build()
    print('parameters: {}'.format(model.count_params()))
    model.load_weights(weights_path())
    print('weights loaded from {}'.format(os.path.basename(weights_path())))

    x = np.load(args.inputs)['x']
    probs = np.asarray(
        [float(model.predict(x[i:i + 1], verbose=0)[0, 1])
         for i in range(x.shape[0])], dtype=np.float64)
    np.savez(args.out, probs=probs, keras='tf.keras',
             tf=str(tf.__version__))
    print('wrote {} predictions'.format(probs.size))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
