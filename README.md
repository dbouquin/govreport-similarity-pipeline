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
*Note - I had an older version of `pyarrow` on my machine that didn't want to work with `datasets`, so if for some reason `pip` doesn't work for you, `conda` worked for me:*

```bash
conda create -n govreport-similarity python=3.10 numpy=1.26.4 -y
conda activate govreport-similarity
# Install packages in this order
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
**Example:**
```bash
dbouquin$ python main.py analyze --model /Users/dbouquin/Documents/govreport-similarity-pipeline/models/BAAI_bge-m3_distilled --num-samples 100

Analyzing Semantic Similarity
Model: /Users/dbouquin/Documents/govreport-similarity-pipeline/models/BAAI_bge-m3_distilled
Dataset: ccdv/govreport-summarization (test)
Output: /Users/dbouquin/Documents/govreport-similarity-pipeline/results/analysis_BAAI_bge-m3_distilled_20250723_234958
Max samples: 100

Validating model...
✓ Model loaded successfully
  Vocabulary size: 249,999
  Embedding dimension: 256

Dataset Information:
  Dataset: ccdv/govreport-summarization
  Required columns present: True

Starting similarity analysis...
Using Model2Vec's optimized internal batching

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
Results saved to: /Users/dbouquin/Documents/govreport-similarity-pipeline/results/analysis_BAAI_bge-m3_distilled_20250723_234958
Samples processed: 100
Samples failed: 0
Mean similarity: 0.890

Next: python main.py report --input 
/Users/dbouquin/Documents/govreport-similarity-pipeline/results/analysis_BAAI_bge-m3_distilled_20250723_234958.csv
```

3. **Generate a report**
Replace with the path to your model
```bash
python main.py report --input /results/analysis_*.csv
```
**Example:**
```bash
dbouquin$ python main.py report --input /Users/dbouquin/Documents/govreport-similarity-pipeline/results/analysis_BAAI_bge-m3_distilled_20250723_234958.csv

Generating Statistical Report
Input: /Users/dbouquin/Documents/govreport-similarity-pipeline/results/analysis_BAAI_bge-m3_distilled_20250723_234958.csv
Output: /Users/dbouquin/Documents/govreport-similarity-pipeline/results/report_BAAI_bge-m3_distilled_20250723_234958
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
Report files saved to: /Users/dbouquin/Documents/govreport-similarity-pipeline/results/report_BAAI_bge-m3_distilled_20250723_234958

Generated Files:
  ✓ Statistical report (JSON): 
/Users/dbouquin/Documents/govreport-similarity-pipeline/results/report_BAAI_bge-m3_distilled_20250723_234958.json (0.01 MB)
  ✓ Summary statistics (CSV): 
/Users/dbouquin/Documents/govreport-similarity-pipeline/results/report_BAAI_bge-m3_distilled_20250723_234958_summary.csv (0.00 MB)
```

## More to Try

4. **Analyze with Custom Output and Batch Size**
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

## CLI Commands & Options

### `distill` Command
Create distilled models using Model2Vec from transformer models.

```bash
python main.py distill [OPTIONS]
```

**Options:**
- `--model` / `-m` - Source model to distill (default: BAAI/bge-m3)
- `--output` / `-o` - Output directory for distilled model (default: auto-generated)
- `--pca-dims` - PCA dimensions for reduction (default: 256)
- `--device` - Device to use: cpu, cuda, mps, auto (default: auto)
- `--custom-vocab` - Path to custom vocabulary file (one token per line)
- `--vocab-from-dataset` - Create vocabulary from the target dataset
- `--vocab-size` - Maximum vocabulary size when creating from dataset (default: 10000)
- `--force` - Overwrite existing distilled model
- `--validate` - Validate distilled model after creation

**Examples:**
```bash
# Basic distillation with default settings
python main.py distill

# Distill with custom PCA dimensions
python main.py distill --model BAAI/bge-m3 --pca-dims 512

# Create vocabulary from target dataset
python main.py distill --vocab-from-dataset --vocab-size 15000

# Distill different model with validation
python main.py distill --model sentence-transformers/all-mpnet-base-v2 --validate
```

### `analyze` Command  
Calculate similarity between reports and summaries using distilled models.

```bash
python main.py analyze [OPTIONS]
```

**Options:**
- `--model` / `-m` - Path to distilled model directory (required)
- `--output` / `-o` - Output path for results (default: auto-generated with timestamp)
- `--dataset` - Dataset to analyze (default: ccdv/govreport-summarization)
- `--split` - Dataset split to use (default: test)
- `--num-samples` / `-n` - Maximum number of samples to analyze (default: all)

**Examples:**
```bash
# Basic analysis using all available data
python main.py analyze --model models/BAAI_bge-m3_distilled

# Test with a small sample first
python main.py analyze --model models/my_model --num-samples 100

# Analyze with custom output location
python main.py analyze \
  --model models/BAAI_bge-m3_distilled \
  --output results/my_experiment \
  --num-samples 1000

# Use different dataset split
python main.py analyze \
  --model models/my_model \
  --dataset ccdv/govreport-summarization \
  --split train \
  --num-samples 500
```

### `report` Command
Generate statistical reports from analysis results.

```bash
python main.py report [OPTIONS]
```

**Options:**
- `--input` / `-i` - Path to analysis results file (JSON or CSV) (required)
- `--output` / `-o` - Output path for report files (default: auto-generated)
- `--format` - Output format: json, csv, both (default: both)

**Examples:**
```bash
# Generate report from analysis results
python main.py report --input results/analysis_model_20250123_143022.csv

# Specify output location and format
python main.py report \
  --input results/analysis_my_model.json \
  --output reports/quality_assessment \
  --format both

# Generate only CSV summary
python main.py report \
  --input results/analysis_*.csv \
  --format csv
```

### Utility Commands

#### `validate` - Check Environment
```bash
python main.py validate [OPTIONS]
```
- `--check-env` - Run comprehensive environment checks

#### `info` - Pipeline Information
```bash
python main.py info
```
Shows configuration, directory status, and usage examples.

## Advanced Usage

### Custom Vocabulary
```bash
# Create domain-specific model with custom vocabulary
python main.py distill \
  --model BAAI/bge-m3 \
  --vocab-from-dataset \
  --vocab-size 15000 \
  --output models/domain_specific
```

## Working Features

### **Core Assignment Requirements**
- **`distill` command**: Creates distilled models with Model2Vec from BAAI/bge-m3
- **`analyze` command**: Calculates similarities, saves results with metadata and tranches analysis
- **`report` command**: Generates statistics and quality metrics with distance distribution
### **Stretch Goal Implemented**
- **Sub-sampling**: `--num-samples` parameter allows analyzing dataset portions

### **Professional Features**
- **Metadata preservation**: Analysis metadata flows through to reports
- **File organization**: Consistent naming with timestamps for easy tracking
- **Error handling**: Robust error messages and validation
- **Progress tracking**: Real-time feedback during long operations
- 
## File Naming Convention

- Analysis output: `analysis_modelname_20250719_143022.json` + `.csv`
- Report output: `report_modelname_20250719_143022.json` + `.txt` + `_summary.csv`

Where timestamp is e.g., `report_modelname_yyymmdd_hhmmss.json` + `.txt` + `_summary.csv`

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

## Optimizations
- **Streaming Datasets**: Process large government report datasets without loading into memory
- **Model2Vec Optimization**: Leverages Model2Vec's internal batching for optimal performance
- **Smart Caching**: Intelligent data loading with validation and cleaning
- **Robust Distillation**: Comprehensive error handling and validation
- **Model Metrics**: Detailed model information and size optimization

## Output Formats

### Analysis Results (JSON)
```json
{
  "analysis_metadata": {
    "model_path": "/Users/dbouquin/Documents/govreport-similarity-pipeline/models/BAAI_bge-m3_distilled",
    "dataset_name": "ccdv/govreport-summarization",
    "analysis_timestamp": "2025-07-23T23:50:06.426517",
    "num_samples_requested": 100,
    "similarity_metric": "cosine",
    "processing_mode": "single_worker"
  },
  "similarity_scores": [
    0.9503476023674011,
    0.8494939208030701,
    0.9366579055786133,
    0.9317487478256226,
    0.7534667253494263,
    0.8712045550346375,
    ...
```

### Analysis Results (CSV)
```csv
similarity_score,sample_id,report_length,summary_length,processing_time
0.9503476023674011,972118,15281,3737,0.0012785506248474121
0.8494939208030701,899072,34791,4779,0.0012785506248474121
0.9366579055786133,62990,49978,3203,0.0012785506248474121
...
```

### Report Output

- **Report (JSON)**
```json
{
  "report_metadata": {
    "input_path": "/Users/dbouquin/Documents/govreport-similarity-pipeline/results/analysis_BAAI_bge-m3_distilled_20250723_234958.csv",
    "report_timestamp": "2025-07-23T23:52:36.828534",
    "analysis_metadata": {
      "model_path": "/Users/dbouquin/Documents/govreport-similarity-pipeline/models/BAAI_bge-m3_distilled",
      "dataset_name": "ccdv/govreport-summarization",
      "analysis_timestamp": "2025-07-23T23:50:06.426517",
      "num_samples_requested": 100,
      "similarity_metric": "cosine",
      "processing_mode": "single_worker"
    },
    "generator": "Government Report Similarity Pipeline"
  },
  "statistical_analysis": {
    "descriptive_statistics": {
      "count": 100,
      "mean": 0.890146678686142,
      "median": 0.8951860070228577,
      "standard_deviation": 0.03954881447897131,
      "variance": 0.0015641087266920909,
      "minimum": 0.7534667253494263,
      "maximum": 0.9503476023674011,
      "range": 0.19688087701797485,
      "skewness": -0.741804686331145,
      "kurtosis": 0.22139716521745134
    },
    "percentiles": {
      "5th": 0.8215997278690338,
      "10th": 0.8332657516002655,
      "25th": 0.8641918301582336,
      "50th": 0.8951860070228577,
      "75th": 0.9214906096458435,
      "90th": 0.9354234457015992,
      "95th": 0.9421292126178742
    },
    "confidence_intervals": {
      "95_percent_lower": 0.8822598024126641,
      "95_percent_upper": 0.8980335549596198,
      "99_percent_lower": 0.8797072280656788,
      "99_percent_upper": 0.9005861293066051
    },
    ...
```
 
- **Report (CSV)**
```csv
Category,Metric,Value
Descriptive Statistics,count,100.0
Descriptive Statistics,mean,0.890146678686142
Descriptive Statistics,median,0.8951860070228577
Descriptive Statistics,standard_deviation,0.03954881447897131
Descriptive Statistics,variance,0.0015641087266920909
Descriptive Statistics,minimum,0.7534667253494263
Descriptive Statistics,maximum,0.9503476023674011
Descriptive Statistics,range,0.19688087701797485
Descriptive Statistics,skewness,-0.741804686331145
Descriptive Statistics,kurtosis,0.22139716521745134
Percentiles,5th_percentile,0.8215997278690338
Percentiles,10th_percentile,0.8332657516002655
Percentiles,25th_percentile,0.8641918301582336
Percentiles,50th_percentile,0.8951860070228577
Percentiles,75th_percentile,0.9214906096458435
Percentiles,90th_percentile,0.9354234457015992
Percentiles,95th_percentile,0.9421292126178742
Threshold Analysis,very_high_similarity_count,46.0
Threshold Analysis,very_high_similarity_percentage,46.0
Threshold Analysis,high_similarity_count,98.0
Threshold Analysis,high_similarity_percentage,98.0
Threshold Analysis,medium_similarity_count,100.0
Threshold Analysis,medium_similarity_percentage,100.0
Threshold Analysis,low_similarity_count,100.0
Threshold Analysis,low_similarity_percentage,100.0
```

