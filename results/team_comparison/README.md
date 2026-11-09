# Team comparison artifacts

Notebook: `notebooks/team_mnist_cifar_comparison.ipynb`.

- Khoi newest literal-profile MNIST/SmallResNet: 96.5% validation accuracy.
- Nguyen updated CIFAR/SmallResNet: 43.2%, from Alka's combined CSV.
- Alka CIFAR/GaussianNB: 20.6%, from that same published CSV.

All five code cells executed successfully and the notebook passed nbformat
validation. Both exported learning-curve charts were visually checked.
Source CSV fingerprint, run aliases, counts, checkpoint schedule, core neural
optimizer settings and Khoi's checkpoint metrics are checked by assertions.
Missing CIFAR precision/recall/F1 and wall times remain unavailable.

Pinned evidence and attribution: `results/team_comparison_sources/provenance.json`.
The updated 43.2% CSV became available in Alka's fetched branch at 10c9ef7;
it resolves the earlier absence of exported 43.2% curve values. Original CIFAR
checkpoints/split artifacts have NOT been independently re-evaluated here.
The older 39.5% / LR 0.002 run is distinct and excluded from these charts.

Different datasets/true classes, pipeline preprocessing and hardware prevent a
controlled three-way model ranking or direct speed ratio. See notebook discussion.
No training or official test evaluation was performed to produce this report.
