#!/usr/bin/env python3


import tensorflow as tf


def main():
    a = tf.Variable([1, 1], tf.float32)
    b = tf.Variable([1, 1])
    c = tf.placeholder(tf.bool, [])


if __name__ == '__main__':
    main()
