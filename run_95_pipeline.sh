#!/bin/bash
# Quick Start Script for 95%+ Accuracy Training Pipeline

echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║                    95%+ ACCURACY TRAINING PIPELINE                             ║"
echo "║         Hypertensive vs Diabetic Retinopathy Classification                   ║"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"

PYTHON="/usr/local/bin/python3.7"

# Check if Python exists
if ! command -v $PYTHON &> /dev/null; then
    echo "❌ Python 3.7 not found at $PYTHON"
    echo "Using default python3..."
    PYTHON="python3"
fi

echo ""
echo "Step 1: Installing dependencies..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$PYTHON -m pip install -q -r requirements_95.txt
if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed"
else
    echo "⚠️  Some dependencies may not have installed"
fi

echo ""
echo "Step 2: Preparing data..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$PYTHON data_preparation.py
if [ $? -eq 0 ]; then
    echo "✅ Data prepared"
else
    echo "⚠️  Data preparation may have issues"
fi

echo ""
echo "Step 3: Training model with ResNet50 (100 epochs)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⏱️  This may take 30-60 minutes on CPU..."
$PYTHON train_95_advanced.py
if [ $? -eq 0 ]; then
    echo "✅ Model training complete"
else
    echo "❌ Training failed"
    exit 1
fi

echo ""
echo "Step 4: Comprehensive evaluation..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$PYTHON evaluate_95_advanced.py
if [ $? -eq 0 ]; then
    echo "✅ Evaluation complete"
else
    echo "⚠️  Evaluation may have issues"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║                         ✅ PIPELINE COMPLETE! ✅                              ║"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Results saved to:"
echo "   - Model: checkpoints/advanced/best_model.pt"
echo "   - Metrics: results/advanced_95/comprehensive_results_*.json"
echo "   - Visualizations: results/advanced_95/*.png"
echo "   - Reports: results/advanced_95/*.txt"
echo ""
echo "📈 Next steps:"
echo "   1. Check results/advanced_95/evaluation_report_test.txt for details"
echo "   2. View confusion_matrix_test.png for visual analysis"
echo "   3. Review per_class_metrics_test.png for performance by class"
echo ""
