#!/usr/bin/env python
"""
pip install -U pygame numpy tensorflow keras opencv-python
"""
from datetime import datetime

import pygame
import os
import sys
import cv2
import numpy as np

fname = 'mnist.h5'


def get_model():
    from keras.models import Sequential, load_model
    if os.path.exists(fname):
        return load_model(fname)
    from keras.datasets import mnist
    from keras.layers import Dense, Dropout, Flatten

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1)) / 255.0
    test_images = test_images.reshape((10000, 28, 28, 1)) / 255.0

    model = Sequential()
    model.add(Flatten(input_shape=(28, 28, 1)))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=5)

    test_loss, testacc = model.evaluate(test_images, test_labels)
    print("Finished training:", test_loss)
    model.save(fname)
    return model


if __name__ == "__main__":
    model = get_model()
    fps = 60
    fps_clock = pygame.time.Clock()

    pygame.init()
    screen = pygame.display.set_mode((512, 512))
    screen.fill((0, 0, 0))
    start = datetime.now()
    drawing = False
    while True:
        events = pygame.event.get()
        for e in events:
            if e.type == pygame.QUIT:
                sys.exit()
            elif e.type == pygame.MOUSEBUTTONDOWN:
                # DRAWING
                if e.button == pygame.BUTTON_RIGHT:
                    screen.fill((0, 0, 0))
                else:
                    drawing = True
            elif e.type == pygame.MOUSEBUTTONUP:
                # STOPPED drawing
                drawing = False
        if drawing:
            pos = pygame.mouse.get_pos()
            pygame.draw.circle(screen, (255, 255, 255), pos, 12)
        # make prediction
        small_img = (cv2.cvtColor(cv2.resize(np.flipud(np.rot90(pygame.surfarray.array3d(screen))), (28, 28)), cv2.COLOR_RGB2GRAY) / 255.0)
        small_img =small_img.reshape(28, 28, 1)
        pred = str(model.predict_classes(np.array([small_img]), batch_size=1)[0])
        pygame.display.set_caption("MNIST Pred: {} at {:.2f} FPS".format(pred, fps_clock.get_fps()))
        pygame.display.flip()
        fps_clock.tick(fps)