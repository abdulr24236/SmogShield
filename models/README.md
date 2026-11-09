# Model weights

No trained model binaries are committed to this repository. The final
classification, regression, and bias models are refit deterministically by the
committed scripts in `src/` from the shared feature table. This avoids
committing `.joblib`, `.pkl`, `.pt`, `.pth`, or `.onnx` files and keeps the
repository below GitHub's 100 MB per-file limit.

External download link for project artifacts and any exported model weights:

<https://drive.google.com/drive/u/1/folders/1gRY4v3sCW90rdJ97O1RFDDux_Dxpd89Y>

If exported model weights are needed for review, place them in that Drive
folder and keep the download link in the root `README.md`; the weight binaries
should not be committed to Git.
