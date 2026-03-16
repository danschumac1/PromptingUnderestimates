'''
python ./src/data_management/format_results.py 
    
'''
import pandas as pd
import os
import re

def parse_pred_path(path):
    # If path is nan or not a string, return None
    if not isinstance(path, str) or '/' not in path:
        return None
        
    parts = path.strip().split('/')
    
    # Safety check: ensure we have at least 5 parts to extract metadata
    if len(parts) < 5:
        print(f"[WARNING] Path too short to parse: {path}")
        return None

    try:
        filename = parts[-1]               
        variant = parts[-2]                
        method = parts[-3]                 
        dataset = parts[-4]                
        model = parts[-5]                  
        
        # Extract layer
        layer_match = re.search(r'layer(\d+)', filename)
        layer = layer_match.group(1) if layer_match else 0
        
        # Extract variant details
        v_parts = variant.split('_')
        embedding_type = v_parts[0] if len(v_parts) > 0 else "unknown"
        # Handle cases where shots might not be in the string
        shots = 0
        if len(v_parts) > 1:
            shots = v_parts[1].replace('-shot', '')
        style = v_parts[2] if len(v_parts) > 2 else "Direct"
        
        return {
            'model': model,
            'dataset': dataset,
            'method': method,
            'embedding_type': embedding_type,
            'style': style,
            'shots': float(shots) if str(shots).replace('.','').isdigit() else 0.0,
            'layer': int(layer),
            'variant': variant,
            'pred_file': filename,
            'pred_stem': filename.replace('.jsonl', ''),
            'run_dir': variant
        }
    except Exception as e:
        print(f"[ERROR] Could not parse path {path}: {e}")
        return None

def process_results(input_csv, output_tsv):
    # Use sep=None with engine='python' to auto-detect Tab vs Space
    df_raw = pd.read_csv(input_csv, sep=None, engine='python')
    
    # Drop rows where pred_path is missing
    df_raw = df_raw.dropna(subset=['pred_path'])
    
    parsed_rows = []
    for _, row in df_raw.iterrows():
        meta = parse_pred_path(row['pred_path'])
        
        if meta is None:
            continue # Skip rows that couldn't be parsed        # Combine original data with parsed metadata
        new_row = {
            **meta,
            'accuracy': round(row['accuracy'], 3),
            'macro_f1': round(row['macro_f1'], 3),
            'timestamp': row['timestamp'] + " " + row.get('timestamp.1', ''), # Handles split timestamp if space-sep
            'pred_path': row['pred_path'],
            'variant': meta['variant'],
            # Fill the requested Unnamed columns with empty strings
            **{f'Unnamed: {i}': "" for i in range(13, 21)}
        }
        parsed_rows.append(new_row)
        
    # Create final DataFrame in the specific column order requested
    cols = [
        'model', 'dataset', 'method', 'embedding_type', 'style', 'shots', 'layer', 
        'accuracy', 'macro_f1', 'timestamp', 'pred_path', 'variant', 'run_dir', 
        'pred_file', 'pred_stem', 'Unnamed: 13', 'Unnamed: 14', 'Unnamed: 15', 
        'Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18', 'Unnamed: 19', 'Unnamed: 20'
    ]
    
    df_final = pd.DataFrame(parsed_rows)
    # Ensure all columns exist even if empty
    for c in cols:
        if c not in df_final.columns:
            df_final[c] = ""
            
    df_final = df_final[cols]
    
    # Save as TSV
    df_final.to_csv(output_tsv, sep='\t', index=False)
    print(f"Successfully formatted results to {output_tsv}")

if __name__ == "__main__":
    # Change these filenames as needed
    process_results('./data/qwen_random.tsv', './data/plot_ready_qwen.tsv')