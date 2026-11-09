# Model weights

No trained model binaries are committed to this repository. The selected
classification and regression pipelines can be refit by the committed scripts
in `src/` using fixed seeds and the shared feature table; the bias analysis
trains nothing and reads saved predictions. This avoids committing `.joblib`,
`.pkl`, `.pt`, `.pth`, or `.onnx` files and keeps the repository below GitHub's
100 MB per-file limit.

External download link for project artifacts and any exported model weights:

<https://drive.google.com/drive/folders/1gRY4v3sCW90rdJ97O1RFDDux_Dxpd89Y>

The current submission does not include pretrained weight files. If reviewers
require them, export the fitted classification and regression pipelines to a
`models/` subfolder in that Drive folder and keep the link in the root
`README.md`; do not commit the binaries to Git.
