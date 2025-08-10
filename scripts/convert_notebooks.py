#!/usr/bin/env python3
"""
Convert Jupyter notebooks to Python files with proper structure
"""

import os
import sys
import argparse
from pathlib import Path
import nbformat
from nbconvert import PythonExporter

def convert_notebook_to_python(notebook_path: str, output_dir: str = None) -> str:
    """
    Convert a Jupyter notebook to a Python file
    
    Args:
        notebook_path: Path to the notebook file
        output_dir: Output directory (optional)
        
    Returns:
        Path to the converted Python file
    """
    notebook_path = Path(notebook_path)
    
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Convert to Python
    python_exporter = PythonExporter()
    python_code, _ = python_exporter.from_notebook_node(nb)
    
    # Determine output path
    if output_dir:
        output_path = Path(output_dir) / f"{notebook_path.stem}.py"
    else:
        output_path = notebook_path.parent / f"{notebook_path.stem}.py"
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write Python file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(python_code)
    
    print(f"Converted {notebook_path} to {output_path}")
    
    return str(output_path)

def convert_all_notebooks(notebooks_dir: str, output_dir: str = None):
    """
    Convert all notebooks in a directory
    
    Args:
        notebooks_dir: Directory containing notebooks
        output_dir: Output directory (optional)
    """
    notebooks_dir = Path(notebooks_dir)
    
    if not notebooks_dir.exists():
        raise FileNotFoundError(f"Directory not found: {notebooks_dir}")
    
    # Find all notebook files
    notebook_files = list(notebooks_dir.glob("*.ipynb"))
    
    if not notebook_files:
        print(f"No notebook files found in {notebooks_dir}")
        return
    
    print(f"Found {len(notebook_files)} notebook files")
    
    # Convert each notebook
    for notebook_file in notebook_files:
        try:
            convert_notebook_to_python(notebook_file, output_dir)
        except Exception as e:
            print(f"Error converting {notebook_file}: {e}")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Convert Jupyter notebooks to Python files")
    parser.add_argument(
        "input", 
        help="Notebook file or directory to convert"
    )
    parser.add_argument(
        "--output-dir", 
        type=str,
        help="Output directory for converted files"
    )
    parser.add_argument(
        "--recursive", 
        action="store_true",
        help="Convert notebooks recursively in subdirectories"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Input path not found: {input_path}")
        sys.exit(1)
    
    try:
        if input_path.is_file():
            # Convert single file
            convert_notebook_to_python(input_path, args.output_dir)
        elif input_path.is_dir():
            # Convert directory
            if args.recursive:
                # Find all notebooks recursively
                notebook_files = list(input_path.rglob("*.ipynb"))
                for notebook_file in notebook_files:
                    convert_notebook_to_python(notebook_file, args.output_dir)
            else:
                # Convert notebooks in directory only
                convert_all_notebooks(input_path, args.output_dir)
        
        print("Conversion completed successfully!")
        
    except Exception as e:
        print(f"Conversion failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 