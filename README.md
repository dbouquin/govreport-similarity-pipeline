# Government Report Summarization Quality Assessment Pipeline

A comprehensive CLI pipeline for evaluating semantic similarity between AI-generated summaries and original government reports using distilled embedding models.

Dataset on Hugging Face:
- [`ccdv/govreport-summarization`](https://huggingface.co/datasets/ccdv/govreport-summarization/)

## Features

- **Model Distillation**: Convert large transformer models into ultra-fast static embeddings using Model2Vec
- **Semantic Analysis**: Calculate cosine similarity between reports and summaries
- **Statistical Reporting**: Generate comprehensive quality assessment reports
- **Memory Efficient**: Stream large datasets without loading everything into memory
- **Professional CLI**: Clean command-line interface with progress tracking and error handling

## Quick Start

### Requirements
```bash
pip install -r requirements.txt
```
*Note- I had some dependency nightmares as I had an older version of pyarrow on my personal machine that didn't want to work with model2vec, so if for some reason `pip` doesn't work for you, `conda` worked for me:*

```bash
conda create -n govreport-similarity python=3.10 numpy=1.26.4 -y
conda activate govreport-similarity
# Install packages in this specific order
conda install pytorch torchvision torchaudio -c pytorch
conda install pyarrow pandas scikit-learn -c conda-forge
pip install transformers datasets model2vec click rich tqdm
```

### Validate the Environment
```bash
# make sure you're ready to go
python main.py validate
```

### Basic Workflow

1. **Create a distilled model (fast static embeddings)**
```bash
# Files download into models/
# PCA reduces 1024-dimensional embeddings to 256 dimensions (4 x smaller)
python main.py distill --model BAAI/bge-m3 --pca-dims 256
```

2. **Test on a sample**
Replace with the path to your model
```bash
python main.py analyze --model /models/BAAI_bge-m3_distilled --num-samples 100
```
**Example output:**
```bash
Analyzing Semantic Similarity
Model: /Users/dbouquin/Documents/govreport-similarity-pipeline/models/BAAI_bge-m3_distilled
Dataset: ccdv/govreport-summarization (test)
Output: /Users/dbouquin/Documents/govreport-similarity-pipeline/results/analysis_BAAI_bge-m3_distilled_20250723_154541
Max samples: 100
Batch size: 32

Validating model...
✓ Model loaded successfully
  Vocabulary size: 249,999
  Embedding dimension: 256

Dataset Information:
  Available splits: []
  Required columns present: True

Starting similarity analysis...

Analysis Results:
        Similarity Analysis Summary         
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Metric                      ┃ Value      ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ Samples Processed           │ 100        │
│ Samples Failed              │ 0          │
│ Success Rate                │ 100.0%     │
├─────────────────────────────┼────────────┤
│ Mean Similarity             │ 0.890      │
│ Median Similarity           │ 0.895      │
│ Min Similarity              │ 0.753      │
│ Max Similarity              │ 0.950      │
├─────────────────────────────┼────────────┤
│ High Similarity (>0.8)      │ 98 (98.0%) │
│ Medium Similarity (0.5-0.8) │ 2 (2.0%)   │
│ Low Similarity (≤0.5)       │ 0 (0.0%)   │
└─────────────────────────────┴────────────┘

Distance Distribution in Tranches:
                                          Distance Distribution Tranches                                           
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Tranche # ┃ Distance Range ┃ Similarity Range ┃ Count ┃ Percentage ┃ Quality Level                              ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│     1     │ 0.050 - 0.089  │ 0.911 - 0.950    │    37 │      37.0% │ Very High Similarity (Excellent summaries) │
│     2     │ 0.089 - 0.128  │ 0.872 - 0.911    │    33 │      33.0% │ High Similarity (Good summaries)           │
│     3     │ 0.128 - 0.168  │ 0.832 - 0.872    │    20 │      20.0% │ Medium Similarity (Acceptable summaries)   │
│     4     │ 0.168 - 0.207  │ 0.793 - 0.832    │     9 │       9.0% │ Low Similarity (Poor summaries)            │
│     5     │ 0.207 - 0.247  │ 0.753 - 0.793    │     1 │       1.0% │ Very Low Similarity (Very poor summaries)  │
└───────────┴────────────────┴──────────────────┴───────┴────────────┴────────────────────────────────────────────┘

Tranches Summary:
  Total tranches: 5
  Total samples: 100
  Distance range: 0.050 - 0.247
  Most populated tranche: #1
  Least populated tranche: #5
  Concentration metric: 1.85 (1.0 = uniform distribution)

Analysis Summary:
  • High-quality samples (>0.8 similarity): 98/100 (98.0%)
  • Standard deviation: 0.040 (consistency indicator)
  • Top 2 tranches contain: 70.0% of samples

✓ Analysis completed successfully!
Results saved to: /Users/dbouquin/Documents/govreport-similarity-pipeline/results/analysis_BAAI_bge-m3_distilled_20250723_154541
Samples processed: 100
Samples failed: 0
Mean similarity: 0.890
```

1. **Generate a report**
```bash
python main.py report --input /results/analysis_*.csv
```
**Example output:**
```bash
Generating Statistical Report
Input: /Users/dbouquin/Documents/govreport-similarity-pipeline/results/analysis_BAAI_bge-m3_distilled_20250723_154950.csv
Output: /Users/dbouquin/Documents/govreport-similarity-pipeline/results/report_BAAI_bge-m3_distilled_20250723_154950
Format: both

Validating input file...
✓ Input file validation passed

Generating report...

Report Summary
    Descriptive Statistics     
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Metric             ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Count              │ 100    │
│ Mean               │ 0.8901 │
│ Median             │ 0.8952 │
│ Standard Deviation │ 0.0395 │
│ Minimum            │ 0.7535 │
│ Maximum            │ 0.9503 │
│ Range              │ 0.1969 │
└────────────────────┴────────┘
             Similarity Distribution              
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━┓
┃ Quality Level ┃ Range     ┃ Count ┃ Percentage ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━┩
│ Very High     │ 0.9 - 1.0 │    46 │      46.0% │
│ High          │ 0.8 - 0.9 │    52 │      52.0% │
│ Medium        │ 0.6 - 0.8 │     2 │       2.0% │
│ Low           │ 0.4 - 0.6 │     0 │       0.0% │
│ Very Low      │ 0.0 - 0.4 │     0 │       0.0% │
└───────────────┴───────────┴───────┴────────────┘

Processing: 100 samples processed, 0 failed
            Distance Tranches Summary            
┏━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━┓
┃ Tranche ┃ Distance Range ┃ Count ┃ Percentage ┃
┡━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━┩
│   #1    │ 0.050 - 0.089  │    37 │      37.0% │
│   #2    │ 0.089 - 0.128  │    33 │      33.0% │
│   #3    │ 0.128 - 0.168  │    20 │      20.0% │
│   #4    │ 0.168 - 0.207  │     9 │       9.0% │
│   #5    │ 0.207 - 0.247  │     1 │       1.0% │
└─────────┴────────────────┴───────┴────────────┘

Most populated tranche: #1
Total samples: 100

✓ Report generation completed successfully!
Report files saved to: /Users/dbouquin/Documents/govreport-similarity-pipeline/results/report_BAAI_bge-m3_distilled_20250723_154950

Generated Files:
  ✓ Statistical report (JSON): /Users/dbouquin/Documents/govreport-similarity-pipeline/results/report_BAAI_bge-m3_distilled_20250723_154950.json 
(0.01 MB)
  ✓ Summary statistics (CSV): 
/Users/dbouquin/Documents/govreport-similarity-pipeline/results/report_BAAI_bge-m3_distilled_20250723_154950_summary.csv (0.00 MB)
```

## More to Try

1. **Analyze with Custom Output and Batch Size**
```bash
python main.py analyze \
  --model models/BAAI_bge-m3_distilled \
  --output results/test_custom \
  --num-samples 25 \
  --batch-size 16
```
**Example Output**
```bash
Analyzing Semantic Similarity
Model: models/BAAI_bge-m3_distilled
Dataset: ccdv/govreport-summarization (test)
Output: results/test_custom
Max samples: 25
Batch size: 16

Validating model...
✓ Model loaded successfully
  Vocabulary size: 249,999
  Embedding dimension: 256

Dataset Information:
  Available splits: []
  Required columns present: True

Starting similarity analysis...

Analysis Results:
        Similarity Analysis Summary         
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Metric                      ┃ Value      ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ Samples Processed           │ 25         │
│ Samples Failed              │ 0          │
│ Success Rate                │ 100.0%     │
├─────────────────────────────┼────────────┤
│ Mean Similarity             │ 0.892      │
│ Median Similarity           │ 0.906      │
│ Min Similarity              │ 0.753      │
│ Max Similarity              │ 0.950      │
├─────────────────────────────┼────────────┤
│ High Similarity (>0.8)      │ 24 (96.0%) │
│ Medium Similarity (0.5-0.8) │ 1 (4.0%)   │
│ Low Similarity (≤0.5)       │ 0 (0.0%)   │
└─────────────────────────────┴────────────┘

Distance Distribution in Tranches:
                                          Distance Distribution Tranches                                           
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Tranche # ┃ Distance Range ┃ Similarity Range ┃ Count ┃ Percentage ┃ Quality Level                              ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│     1     │ 0.050 - 0.089  │ 0.911 - 0.950    │    12 │      48.0% │ Very High Similarity (Excellent summaries) │
│     2     │ 0.089 - 0.128  │ 0.872 - 0.911    │     4 │      16.0% │ High Similarity (Good summaries)           │
│     3     │ 0.128 - 0.168  │ 0.832 - 0.872    │     5 │      20.0% │ Medium Similarity (Acceptable summaries)   │
│     4     │ 0.168 - 0.207  │ 0.793 - 0.832    │     3 │      12.0% │ Low Similarity (Poor summaries)            │
│     5     │ 0.207 - 0.247  │ 0.753 - 0.793    │     1 │       4.0% │ Very Low Similarity (Very poor summaries)  │
└───────────┴────────────────┴──────────────────┴───────┴────────────┴────────────────────────────────────────────┘

Tranches Summary:
  Total tranches: 5
  Total samples: 25
  Distance range: 0.050 - 0.247
  Most populated tranche: #1
  Least populated tranche: #5
  Concentration metric: 2.40 (1.0 = uniform distribution)

Analysis Summary:
  • High-quality samples (>0.8 similarity): 24/25 (96.0%)
  • Standard deviation: 0.050 (consistency indicator)
  • Top 2 tranches contain: 64.0% of samples

✓ Analysis completed successfully!
Results saved to: results/test_custom
Samples processed: 25
Samples failed: 0
Mean similarity: 0.892
```

## CLI Interface

## Available Commands & Flags

### `distill` Command
```bash
python main.py distill \
    --model BAAI/bge-m3 \
    --output models/my_model \
    --pca-dims 256
```
**Flags:**
- `--model` / `-m` - Source model to distill (required)
- `--output` / `-o` - Output directory for distilled model
- `--pca-dims` - PCA dimensions for reduction

### `analyze` Command  
```bash
python main.py analyze \
    --model models/my_model \
    --output results/my_experiment \
    --num-samples 1000 \
    --batch-size 64
```
**Flags:**
- `--model` / `-m` - Path to distilled model (required)
- `--output` / `-o` - Output path for results (auto-generated if not specified)
- `--dataset` - Dataset to analyze (default: "ccdv/govreport-summarization")
- `--split` - Dataset split to use (default: "test")
- `--num-samples` / `-n` - Max samples to analyze (default: all) **[STRETCH GOAL]**
- `--batch-size` / `-b` - Batch size for processing (default: 32)

### `report` Command
```bash
python main.py report \
    --input results/analysis_model_20250719_143022.json \
    --output reports/quality_report \
    --format text
```
**Flags:**
- `--input` / `-i` - Path to analysis results file (required)
- `--output` / `-o` - Output path for report (auto-generated if not specified)
- `--format` - Output format: json, text, csv, all (default: all)

## Working Stretch Goal

1. **Sub-sampling**: `--num-samples` parameter allows analyzing dataset portions

## File Naming Convention

- Analysis output: `analysis_modelname_20250719_143022.json` + `.csv`
- Report output: `report_modelname_20250719_143022.json` + `.txt` + `_summary.csv`

Where timestamp is e.g., `report_modelname_yyymmdd_hhmmss.json` + `.txt` + `_summary.csv`

## Core Assignment Completion

- `distill` command - Creates distilled models with Model2Vec  
- `analyze` command - Calculates similarities, saves results with metadata  
- `report` command - Generates statistics and quality metrics  
- Sub-sampling stretch goal - `--num-samples` parameter  
- Clean, professional interface  
- Metadata preservation - Analysis metadata flows to reports  
- Proper file organization - Consistent naming for easy tracking

## Architecture

```
govreport_similarity/
├── src/
│   ├── commands/                # Command implementations
│   │   ├── distill.py           # Model distillation logic
│   │   ├── analyze.py           # Similarity analysis command
│   │   └── report.py            # Report generation command
│   ├── core/                    # Core business logic
│   │   ├── distiller.py         # Model2Vec integration & distillation
│   │   ├── analyzer.py          # Similarity computation engine
│   │   ├── data_loader.py       # HuggingFace dataset streaming
│   │   └── reporter.py          # Statistical analysis & reporting
│   └── utils/                   # Utility modules
│       ├── config.py            # Configuration management
│       ├── logging.py           # Logging setup & utilities
│       └── validation.py        # Environment & input validation
├── models/                      # Distilled models storage
├── results/                     # Analysis results storage
├── requirements.txt             # Python dependencies
├── main.py                      # Primary CLI entry point
└── README.md                    # This file
```

## Key Features

### Memory-Efficient Processing
- **Streaming Datasets**: Process large government report datasets without loading into memory
- **Batch Processing**: Configurable batch sizes for optimal memory/speed tradeoffs
- **Smart Caching**: Intelligent data loading with validation and cleaning

### Professional Model2Vec Integration
- **Parameter Safety**: Addresses Model2Vec "unexpected behaviors" with explicit parameter handling
- **Robust Distillation**: Comprehensive error handling and validation
- **Model Metrics**: Detailed model information and size optimization

### Advanced Analysis Capabilities
- **Parallel Processing**: Optional multiprocessing for large-scale analysis
- **Outlier Detection**: Statistical analysis of quality patterns
- **Distribution Analysis**: Comprehensive similarity score analysis
- **Content Length Analysis**: Correlation between length and quality

### Comprehensive Reporting
- **Executive Summary**: High-level quality assessment with ratings
- **Statistical Analysis**: Detailed descriptive statistics and distributions
- **Quality Assessment**: Categorized quality metrics with thresholds
- **Actionable Recommendations**: Specific improvement suggestions

## Output Formats

### Analysis Results (CSV)
```csv
sample_id,similarity_score,report_length,summary_length,processing_time
doc_001,0.847,1247,156,0.023
doc_002,0.739,2103,203,0.031
```

### Analysis Results (JSON)
```json
{
  "analysis_metadata": {
    "model_path": "models/BAAI_bge-m3_distilled",
    "dataset_name": "ccdv/govreport-summarization",
    "analysis_timestamp": "2025-01-19T15:30:00",
    "similarity_metric": "cosine"
  },
  "similarity_scores": [0.847, 0.739, ...],
  "processing_stats": {
    "samples_processed": 1000,
    "samples_failed": 13,
    "average_processing_time": 0.025
  }
}
```

### Report Output
- **JSON**: Complete machine-readable report with all metrics
- **Text**: Human-readable summary with key findings
- **CSV**: Summary statistics for spreadsheet analysis
