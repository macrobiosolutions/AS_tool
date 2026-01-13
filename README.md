# Active-Site Pocket Predictor

A Streamlit-based web application for predicting active site pockets in protein structures using computational methods.

## Features

- Upload PDB files for protein structure analysis
- Customizable grid-based pocket detection
- DBSCAN clustering for pocket identification
- Interactive 3D visualization using py3Dmol
- Identification of key residues near pocket centroids
- Export results as CSV and PDB files

## Installation

### Local Installation

```bash
git clone https://github.com/macrobiosolutions/AS_tool.git
cd AS_tool
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run active_site_gui.py
```

## Usage

1. Upload a PDB file
2. Adjust parameters:
   - Grid spacing
   - Bounding box padding
   - Distance thresholds
   - Clustering parameters
3. Click "Run prediction"
4. View results in the 3D viewer
5. Download pocket summaries and residue data

## Requirements

- Python 3.8+
- streamlit
- biopython
- numpy
- pandas
- scikit-learn
- py3Dmol
- stmol
- scipy

## About

Developed by [Macro Bio Solutions](https://github.com/macrobiosolutions)

## License

MIT License
