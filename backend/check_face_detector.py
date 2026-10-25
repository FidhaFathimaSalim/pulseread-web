"""
Run this in the SAME Python environment as live_server.py (e.g. venv_openrppg):

        python check_face_detector.py

It tells you whether face detection works for multi-person mode and, if not, exactly why
and which commands fix it.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2
import face_tracker as ft

print('Python        :', sys.executable)
print('OpenCV        :', cv2.__version__, '|', os.path.dirname(cv2.__file__))
print('CascadeClassifier available :', hasattr(cv2, 'CascadeClassifier'))

try:
    from importlib import metadata
    found = sorted({f'{d.metadata["Name"]} {d.version}' for d in metadata.distributions()
                    if d.metadata['Name'] and 'opencv' in d.metadata['Name'].lower()})
except Exception:                    # noqa: BLE001
    found = []
print('Installed OpenCV packages   :', ', '.join(found) if found else '(none found)')
if len(found) > 1:
    print('  ! Several OpenCV packages are installed together. That is a classic cause of a half-broken cv2.')

blank = np.zeros((480, 640, 3), np.uint8)
ok = {}

print('\n1) OpenCV Haar cascade')
try:
    ft.haar_detector_factory()(blank)
    ok['haar'] = True
    print('   OK')
except Exception as e:               # noqa: BLE001
    ok['haar'] = False
    print('   FAILED:', e)

print('\n2) MediaPipe FaceLandmarker')
try:
    ft.mediapipe_detector_factory()(blank)
    ok['mediapipe'] = True
    print('   OK')
except Exception as e:               # noqa: BLE001
    ok['mediapipe'] = False
    print('   FAILED:', e)

print()
if ok['mediapipe'] or ok['haar']:
    print('RESULT: face detection is READY (live_server will use '
          + ('MediaPipe' if ok['mediapipe'] else 'the OpenCV Haar cascade') + ').')
else:
    print('RESULT: NO face detector works here, so multi-person cannot find faces.\n')
    print('Easiest fix - repair OpenCV (run these in this same environment):')
    print('    pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless')
    print('    pip install opencv-python')
    print('then run this script again.\n')
    print('Alternative: pip install mediapipe   and make sure face_landmarker.task is in the backend folder.')

sys.stdout.flush()
os._exit(0)      # skips MediaPipe's harmless noisy shutdown messages
