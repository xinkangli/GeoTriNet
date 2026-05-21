# GeoTriNet - Getting Started

Welcome to **GeoTriNet** (Geometry-Aware Trimodal Network) - a deep learning framework for toxicity prediction combining 2D graph attention networks, molecular descriptors, and 3D geometric information.

## 📦 What's Inside

This is a **professional, production-ready codebase** organized for easy reproduction and extension:

### Core Modules
- **`encoders.py`** - Neural network building blocks (GAT, EGNN, MLP)
- **`models.py`** - GeoTriNet architecture combining all modalities  
- **`features.py`** - Molecular feature extraction (graphs, fingerprints, descriptors)
- **`data.py`** - Dataset loading and preprocessing
- **`metrics.py`** - Evaluation metrics (ACC, AUC, Sensitivity, Specificity)
- **`train.py`** - Training engine with focal loss, R-Drop, and ensemble support

### Scripts
- **`train_phase3.py`** - Main entry point for training (2D + Descriptor + 3D model)

### Documentation
- **`README.md`** - Architecture overview
- **`EXAMPLE.md`** - Usage examples and tutorials
- **`PROJECT_STRUCTURE.md`** - Detailed project layout

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
Create three CSV files with columns: `SMILES`, `label` (and optionally `source`)
- `data/contact.csv`
- `data/oral.csv`
- `data/overall.csv`

### 3. Train Model
```bash
python train_phase3.py \
    --contact_data data/contact.csv \
    --oral_data data/oral.csv \
    --overall_data data/overall.csv \
    --device cuda \
    --epochs 120
```

### 4. Check Results
Results saved to `results/results.json` with metrics for each task.

## 📚 Documentation Map

| File | Purpose |
|------|---------|
| **README.md** | Start here - architecture overview |
| **EXAMPLE.md** | Code examples for training and inference |
| **PROJECT_STRUCTURE.md** | Detailed module descriptions |
| **setup.py** | Package installation info |

## 🔧 Key Features

✅ **Modular Architecture** - Swap encoders or add new modalities  
✅ **Multi-Task Learning** - Contact, Oral, Overall toxicity prediction  
✅ **3D-Aware** - Leverages multi-conformer EGNN encoding  
✅ **Robust Training** - Focal loss, R-Drop, label smoothing, warmup  
✅ **Ensemble Ready** - Multi-seed training built-in  
✅ **Reproducible** - Fixed seeds, deterministic splits, detailed logging  
✅ **Production-Ready** - Error handling, validation, clean code structure  

## 📖 Reading Order

**For Users:**
1. README.md (understand architecture)
2. EXAMPLE.md (run examples)
3. train_phase3.py (customize and run)

**For Developers:**
1. PROJECT_STRUCTURE.md (understand organization)
2. models.py (main architecture)
3. encoders.py (building blocks)
4. train.py (training logic)

## 🎯 Model Architecture

```
Input: SMILES, Descriptors, 3D Conformers

┌─ 2D Graph (SMILES) ──→ GAT Encoder ─────┐
├─ Descriptors ─────────→ MLP Encoder ──┬─→ Gated Fusion ─→ ┌─ Contact Head
└─ 3D Conformers ──→ Multi-Conf EGNN ──┴──────────────────├─ Oral Head
                                                            └─ Overall Head
                                        (Hierarchical: Overall uses Contact + Oral)
```

## 💻 System Requirements

- Python 3.8+
- PyTorch 1.12+
- RDKit (for molecular processing)
- 8GB RAM minimum (GPU recommended for batch_size > 16)

## 📝 Citation

If you use GeoTriNet, please reference:

```bibtex
@software{geotri-net2024,
  title={GeoTriNet: Geometry-Aware Trimodal Network for Toxicity Prediction},
  year={2024}
}
```

## 📬 File Summary

```
geotri-net/
├── 🎯 train_phase3.py       ← Start here: main training script
├── 📖 README.md              ← Architecture overview  
├── 📚 EXAMPLE.md             ← Usage examples
├── 🔨 setup.py               ← Package installation
├── 📋 requirements.txt        ← Dependencies
│
├── 🧠 models.py              ← GeoTriNet main class
├── 🧩 encoders.py            ← Neural network modules
├── 🧬 features.py            ← Molecular feature extraction
├── 📊 data.py                ← Dataset loading
├── 📈 metrics.py             ← Evaluation metrics
├── 🔄 train.py               ← Training loop & loss functions
│
├── 📄 LICENSE                ← MIT License
├── 🚫 .gitignore             ← Git settings
└── 📋 PROJECT_STRUCTURE.md   ← Detailed documentation
```

## 🔗 Next Steps

1. **Read** `README.md` for architecture details
2. **Check** `EXAMPLE.md` for code samples
3. **Run** `python train_phase3.py --help` for all options
4. **Explore** individual modules for customization

---

**Happy training! 🚀**

For questions or issues, refer to the documentation files above or examine the source code directly - it's well-commented and modular.
