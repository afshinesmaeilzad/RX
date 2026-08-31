"""RX — the cloud side of CXR GroundAssist.

Serves CURE over HTTP for the desktop app, and runs the continual-learning
loop: corrections in, a continued adapter out, promoted only if it passes the
benchmark gate.
"""
