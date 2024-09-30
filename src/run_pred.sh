#!/bin/bash

# Default log file path
LOG_FILE="output_log.txt"

# List of paths to iterate over
paths=(
    "/home/avi/BoxTaxo_QLM/result/science/model/exp_model_bert_science_0_100_128_superposn_2e-05_trace_complex.checkpoint"
    "/home/avi/BoxTaxo_QLM/result/science/model/exp_model_bert_science_0_100_128_False_False_None.checkpoint"
    "/home/avi/BoxTaxo_QLM/result/science/model/exp_model_bert_science_0_100_128_False_False_uniform.checkpoint" #Observable?
    "/home/avi/BoxTaxo_QLM/result/environment/model/exp_model_bert_environment_0_100_128_constant_2e-05_trace_complex.checkpoint"
    "/home/avi/BoxTaxo_QLM/result/environment/model/exp_model_bert_environment_0_100_128_False_False_None.checkpoint"
    # Add more paths here
)

modeltype=(
    true
    false
    false
    true
    false    
)

# Function to display help
show_help() {
    echo "Usage: $0 [-l <log_file_path>]"
    echo "  -l   (Optional) Log file to store the output (default: output_log.txt)"
}

# Parse command-line arguments
while getopts "l:h" opt; do
    case $opt in
        l) LOG_FILE="$OPTARG"
        ;;
        h) show_help
        exit 0
        ;;
        *) show_help
        exit 1
        ;;
    esac
done

# Iterate over each path and run the Python script
for i in "${!paths[@]}"; do
    path=${paths[$i]}
    flag=${modeltype[$i]}
    echo "Processing path: $path"

    echo "Path: $path" >> "$LOG_FILE"
    python main_pred.py --path "$path" --complex "$flag" | \
    grep -E '^Namespace|^acc' | \
    head -n 2 >> "$LOG_FILE"

    # Append the path and output to the log file
    {
        echo "--------------------"
    } >> "$LOG_FILE"

    echo "Output for $path has been logged to $LOG_FILE"
done
