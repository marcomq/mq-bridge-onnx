# Regenerates tiny_classifier.onnx: a 3-feature CatBoost classifier, label = a > 5 and c < 6.
# uvx -p 3.12 --with numpy --with catboost --with onnx python make_tiny_classifier.py tiny_classifier.onnx
import sys, numpy as np
from catboost import CatBoostClassifier
rng = np.random.default_rng(7)
X = rng.uniform(0, 10, size=(400, 3))
y = ((X[:, 0] > 5) & (X[:, 2] < 6)).astype(int)
m = CatBoostClassifier(iterations=20, depth=3, verbose=False, random_seed=7).fit(X, y)
m.save_model(sys.argv[1], format="onnx")
for row in [[8, 1, 2], [1, 1, 9]]:
    print(row, m.predict_proba(np.array([row], dtype=float))[0].round(6).tolist(), m.predict(np.array([row], dtype=float)).tolist())
