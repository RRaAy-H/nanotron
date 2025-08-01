#!/bin/bash
# Complete validation workflow for SmolVLM2-Infini data pipeline
# 
# Usage: ./validate_pipeline.sh [OPTIONS]
#
# This script runs the complete validation workflow:
# 1. Prepares training data from local storage  
# 2. Shows dataset statistics
# 3. Validates mixture configuration
# 4. Runs comprehensive pipeline tests
#
# Options:
#   --skip-prepare    Skip data preparation step
#   --data-dir DIR    Data directory (default: data/datasets)
#   --base-path PATH  Base path for local data (default: /data1/yihao)
#   --mixture PATH    Mixture config path (default: data/smolvlm2_256m_mixture.yaml)

set -e  # Exit on any error

# Default values
SKIP_PREPARE=false
DATA_DIR="data/datasets"
BASE_PATH="/data1/yihao"
MIXTURE_PATH="data/smolvlm2_256m_mixture.yaml"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-prepare)
            SKIP_PREPARE=true
            shift
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --base-path)
            BASE_PATH="$2"
            shift 2
            ;;
        --mixture)
            MIXTURE_PATH="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-prepare      Skip data preparation step"
            echo "  --data-dir DIR      Data directory (default: data/datasets)"
            echo "  --base-path PATH    Base path for local data (default: /data1/yihao)"
            echo "  --mixture PATH      Mixture config path (default: data/smolvlm2_256m_mixture.yaml)"
            echo "  -h, --help         Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if we're in the right directory
if [[ ! -f "scripts/validate_pipeline.sh" ]]; then
    print_error "Please run this script from the smolvlm2_infini directory"
    exit 1
fi

echo "======================================================================="
echo "SmolVLM2-Infini Data Pipeline Validation Workflow"
echo "======================================================================="
echo "Data directory: $DATA_DIR"
echo "Base path: $BASE_PATH"
echo "Mixture config: $MIXTURE_PATH"
echo "Skip preparation: $SKIP_PREPARE"
echo ""

# Step 1: Data Preparation (optional)
if [[ "$SKIP_PREPARE" == "false" ]]; then
    print_step "Preparing training data from local storage..."
    
    if python scripts/prepare_training_data.py \
        --output_dir "$DATA_DIR" \
        --base_path "$BASE_PATH" \
        --seed 42; then
        print_success "Data preparation completed"
    else
        print_error "Data preparation failed"
        exit 1
    fi
else
    print_warning "Skipping data preparation (--skip-prepare specified)"
fi

echo ""

# Step 2: Quick dataset statistics
print_step "Checking dataset statistics..."

if python scripts/debug_data_pipeline.py \
    --data_dir "$DATA_DIR" \
    --stats; then
    print_success "Dataset statistics generated"
else
    print_warning "Could not generate dataset statistics"
fi

echo ""

# Step 3: Validate mixture configuration
print_step "Validating data mixture configuration..."

if python scripts/debug_data_pipeline.py \
    --mixture "$MIXTURE_PATH" \
    --validate_mixture; then
    print_success "Mixture configuration is valid"
else
    print_error "Mixture configuration validation failed"
    exit 1
fi

echo ""

# Step 4: Inspect a few key datasets
print_step "Inspecting key datasets..."

# List of important datasets to check
key_datasets=(
    "magpie_pro_l3_80b_mt.json"
    "llava_onevision_other.json"
    "llava_video_1_2m.json"
    "m4_instruct_data.json"
)

for dataset in "${key_datasets[@]}"; do
    dataset_path="$DATA_DIR/$dataset"
    
    if [[ -f "$dataset_path" ]]; then
        echo ""
        echo "Inspecting $dataset..."
        
        if python scripts/debug_data_pipeline.py \
            --dataset "$dataset_path" \
            --inspect \
            --num_samples 3; then
            print_success "✓ $dataset inspection passed"
        else
            print_warning "⚠ Issues found in $dataset"
        fi
    else
        print_warning "⚠ $dataset not found, skipping inspection"
    fi
done

echo ""

# Step 5: Comprehensive pipeline test
print_step "Running comprehensive pipeline tests..."

if python scripts/test_data_pipeline.py \
    --data_dir "$DATA_DIR" \
    --mixture_path "$MIXTURE_PATH" \
    --num_samples 5; then
    print_success "Comprehensive pipeline tests completed"
    
    # Check if validation passed
    if grep -q "DATA PIPELINE IS READY FOR TRAINING" test_data_pipeline.log 2>/dev/null; then
        echo ""
        echo "======================================================================="
        print_success "🎉 DATA PIPELINE VALIDATION SUCCESSFUL!"
        print_success "The data pipeline is ready for SmolVLM2-Infini training."
        echo "======================================================================="
        echo ""
        echo "Next steps:"
        echo "1. Copy this validated setup to your training server"
        echo "2. Ensure the base path (/data1/yihao) is accessible on the training server"
        echo "3. Run training with:"
        echo "   python scripts/train_smolvlm2_infini.py \\"
        echo "     --data_mixture $MIXTURE_PATH \\"
        echo "     --output_dir checkpoints/smolvlm2_infini"
        echo ""
    else
        echo ""
        echo "======================================================================="
        print_warning "⚠️  DATA PIPELINE VALIDATION COMPLETED WITH WARNINGS"
        print_warning "Check the test logs for details and recommendations."
        echo "======================================================================="
        echo ""
        echo "Review the following files for details:"
        echo "- test_data_pipeline.log"
        echo "- data_pipeline_validation_report.json"
        echo ""
    fi
else
    echo ""
    echo "======================================================================="
    print_error "❌ DATA PIPELINE VALIDATION FAILED"
    print_error "Critical issues found that must be resolved before training."
    echo "======================================================================="
    echo ""
    echo "Check the following for debugging:"
    echo "- test_data_pipeline.log"
    echo "- Console output above"
    echo ""
    echo "Common fixes:"
    echo "1. Ensure all dataset files exist in $DATA_DIR"
    echo "2. Verify mixture config paths in $MIXTURE_PATH"
    echo "3. Check file permissions and disk space"
    echo "4. Ensure dependencies are installed correctly"
    echo ""
    exit 1
fi

echo ""
echo "Validation workflow completed successfully!"
echo "Generated files:"
echo "- test_data_pipeline.log (detailed test log)"
echo "- data_pipeline_validation_report.json (validation report)"
echo ""