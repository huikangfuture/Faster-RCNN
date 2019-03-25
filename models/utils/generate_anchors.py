import os
import sys
import torch
import numpy as np


__all__ = ['generate_anchors']


def generate_anchors(size=16, scales=[8, 16, 32], ratios=[0.5, 1, 2]):
    anchor = np.array([0, 0, size, size])
    scales, ratios = np.array(scales), np.array(ratios)
    anchors = scales_enum(anchor, scales)
    anchors = np.vstack([ratios_enum(v, ratios) for v in anchors])
    return np.round(anchors)


def anchor_cvt(anchor):
    w = anchor[2] - anchor[0]
    h = anchor[3] - anchor[1]
    cx = anchor[0] + w * 0.5
    cy = anchor[1] + h * 0.5
    return w, h, cx, cy


def scales_enum(anchor, scales):
    w, h, cx, cy = anchor_cvt(anchor)
    ws = w * scales * 0.5
    hs = h * scales * 0.5
    anchors = np.stack((cx - ws, cy - hs, cx + ws, cy + hs), axis=1)
    return anchors


def ratios_enum(anchor, ratios):
    w, h, cx, cy = anchor_cvt(anchor)
    ws = np.sqrt(w * h * ratios) * 0.5
    hs = np.sqrt(w * h / ratios) * 0.5
    anchors = np.stack((cx - ws, cy - hs, cx + ws, cy + hs), axis=1)
    return anchors


if __name__ == '__main__':
    anchors = generate_anchors()
    print(anchors)
