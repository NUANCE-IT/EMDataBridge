#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Discovery module for electron microscopy data organization.

This module provides functionality to automatically discover and extract
self-describing data organization from electron microscopy files using LLM-based approaches.
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Union, List, Optional, Tuple, Set
from collections import defaultdict
# Lazy-transformers import to avoid requiring heavy ML deps at package import
_TRANSFORMERS_AVAILABLE = False
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import the standardizer for common functionality
from .standardizer import EMDataStandardizer


class EMDataDiscovery:
    """
    A class for discovering and extracting self-describing data organization
    from electron microscopy files using LLM-based approaches.
    
    This discovery tool can analyze collections of electron microscopy files,
    extract metadata patterns, and infer relationships between files.
    """
    
    def __init__(self, llm_model_name: str = "t5-small", llm_backend: str = "none", llm_model_path: str | None = None):
        """
        Initialize the discovery tool with an optional LLM model for metadata analysis.
        
        Args:
            llm_model_name: Name of the pre-trained model to use for metadata analysis
        """
        # Initialize the standardizer for common functionality, forwarding LLM config
        self.standardizer = EMDataStandardizer(llm_model_name=llm_model_name, llm_backend=llm_backend, llm_model_path=llm_model_path)
        
        # Initialize LLM model lazily. Load only if transformers available and
        # needed by downstream methods.
        self.llm_available = False
        self.tokenizer = None
        self.model = None
        self.llm_model_name = llm_model_name
        
        # Initialize TF-IDF vectorizer for text similarity
        self.vectorizer = TfidfVectorizer()
    
    def discover_dataset_structure(self, directory: Union[str, Path], 
                                  recursive: bool = True,
                                  file_pattern: str = "*") -> Dict[str, Any]:
        """
        Discover the structure of a dataset in a directory.
        
        Args:
            directory: Directory containing electron microscopy files
            recursive: Whether to search recursively in subdirectories
            file_pattern: Glob pattern to match files
            
        Returns:
            Dictionary with dataset structure information
        """
        directory = Path(directory)
        
        # Find all files matching the pattern
        if recursive:
            files = list(directory.glob(f"**/{file_pattern}"))
        else:
            files = list(directory.glob(file_pattern))
        
        # Filter for known file extensions
        supported_extensions = set(self.standardizer.supported_formats)
        
        em_files = [f for f in files if f.suffix.lower() in supported_extensions]
        
        if not em_files:
            print(f"No supported electron microscopy files found in {directory}")
            return {"files": [], "patterns": {}, "relationships": []}
        
        print(f"Found {len(em_files)} electron microscopy files")
        
        # Extract metadata from files
        metadata_collection = []
        for file_path in em_files:
            try:
                # Standardize the file to extract metadata
                standardized = self.standardizer.standardize(file_path)
                metadata = standardized["metadata"]
                
                # Add file information
                file_info = {
                    "path": str(file_path),
                    "name": file_path.name,
                    "format": standardized["original_format"],
                    "relative_path": str(file_path.relative_to(directory)),
                    "metadata": metadata
                }
                
                metadata_collection.append(file_info)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        # Analyze metadata patterns
        patterns = self._analyze_metadata_patterns(metadata_collection)
        
        # Discover relationships between files
        relationships = self._discover_relationships(metadata_collection)
        
        # Discover dataset hierarchy
        hierarchy = self._discover_hierarchy(metadata_collection)
        
        return {
            "files": metadata_collection,
            "patterns": patterns,
            "relationships": relationships,
            "hierarchy": hierarchy
        }
    
    def _analyze_metadata_patterns(self, metadata_collection: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze patterns in metadata across files.
        
        Args:
            metadata_collection: List of file metadata dictionaries
            
        Returns:
            Dictionary with metadata pattern information
        """
        # Count occurrences of metadata fields
        field_counts = defaultdict(int)
        field_values = defaultdict(set)
        
        for file_info in metadata_collection:
            metadata = file_info["metadata"]
            
            # Recursively process nested metadata
            def process_metadata(data, prefix=""):
                if isinstance(data, dict):
                    for key, value in data.items():
                        if str(key).startswith("_"):      # bookkeeping (_source, _llm_provenance), not data
                            continue
                        full_key = f"{prefix}.{key}" if prefix else key
                        
                        if value is not None:
                            field_counts[full_key] += 1
                            
                            # Store unique values (convert non-hashable types to strings)
                            if isinstance(value, (list, dict)):
                                field_values[full_key].add(str(value))
                            else:
                                field_values[full_key].add(value)
                            
                            # Recurse into nested structures
                            process_metadata(value, full_key)
            
            process_metadata(metadata)
        
        # Calculate statistics
        total_files = len(metadata_collection)
        field_stats = {}
        
        for field, count in field_counts.items():
            coverage = count / total_files
            unique_values = field_values[field]
            cardinality = len(unique_values)
            
            # Determine if field is an identifier
            is_identifier = cardinality == total_files and coverage == 1.0
            
            # Determine if field is categorical
            is_categorical = cardinality < total_files * 0.5 and cardinality > 1
            
            # Determine field type
            if all(isinstance(v, (int, float)) for v in unique_values if not isinstance(v, bool)):
                field_type = "numeric"
            elif all(isinstance(v, str) for v in unique_values):
                field_type = "text"
            elif all(isinstance(v, bool) for v in unique_values):
                field_type = "boolean"
            else:
                field_type = "mixed"
            
            field_stats[field] = {
                "coverage": coverage,
                "cardinality": cardinality,
                "is_identifier": is_identifier,
                "is_categorical": is_categorical,
                "type": field_type,
                "example_values": list(unique_values)[:5]  # Show up to 5 examples
            }
        
        return field_stats
    
    def _discover_relationships(self, metadata_collection: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Discover relationships between files based on metadata.
        
        Args:
            metadata_collection: List of file metadata dictionaries
            
        Returns:
            List of relationship dictionaries
        """
        relationships = []
        
        # Extract file names and paths for easier reference
        file_info_by_name = {file_info["name"]: file_info for file_info in metadata_collection}
        
        # Create a text representation of each file's metadata for similarity comparison
        metadata_texts = []
        for file_info in metadata_collection:
            # Flatten metadata to a string representation
            metadata_text = self._flatten_metadata_to_text(file_info["metadata"])
            metadata_texts.append(metadata_text)
        
        # Calculate similarity matrix if we have at least 2 files
        if len(metadata_texts) >= 2:
            # Vectorize the metadata texts
            try:
                tfidf_matrix = self.vectorizer.fit_transform(metadata_texts)
                
                # Calculate cosine similarity
                similarity_matrix = cosine_similarity(tfidf_matrix)
                
                # Find relationships based on similarity threshold
                threshold = 0.7  # Similarity threshold
                
                for i in range(len(metadata_collection)):
                    for j in range(i + 1, len(metadata_collection)):
                        similarity = similarity_matrix[i, j]
                        
                        if similarity > threshold:
                            file1 = metadata_collection[i]
                            file2 = metadata_collection[j]
                            
                            relationship = {
                                "type": "similarity",
                                "file1": file1["path"],
                                "file2": file2["path"],
                                "similarity": float(similarity),
                                "description": f"Similar metadata content"
                            }
                            
                            relationships.append(relationship)
            except Exception as e:
                print(f"Error calculating similarities: {e}")
        
        # Look for specific relationship patterns in metadata
        self._find_sequence_relationships(metadata_collection, relationships)
        self._find_parent_child_relationships(metadata_collection, relationships)
        
        return relationships
    
    def _flatten_metadata_to_text(self, metadata: Dict[str, Any], prefix: str = "") -> str:
        """
        Flatten nested metadata dictionary to a single text string.
        
        Args:
            metadata: Metadata dictionary
            prefix: Prefix for nested keys
            
        Returns:
            Flattened metadata as text
        """
        text_parts = []
        
        for key, value in metadata.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                # Recurse into nested dictionaries
                text_parts.append(self._flatten_metadata_to_text(value, full_key))
            elif value is not None:
                # Add key-value pair to text
                text_parts.append(f"{full_key}: {value}")
        
        return " ".join(text_parts)
    
    def _find_sequence_relationships(self, metadata_collection: List[Dict[str, Any]], relationships: List[Dict[str, Any]]) -> None:
        """
        Find sequence relationships between files (e.g., time series, z-stacks).
        
        Args:
            metadata_collection: List of file metadata dictionaries
            relationships: List to append discovered relationships to
        """
        # Group files by similar names (potential sequences)
        name_patterns = defaultdict(list)
        
        # Simple pattern matching for common sequence indicators
        import re
        sequence_patterns = [
            r'(.+)_([0-9]+)\.',  # name_001.ext
            r'(.+)\(([0-9]+)\)\.',  # name(001).ext
            r'(.+)-([0-9]+)\.',  # name-001.ext
            r'(.+)_t([0-9]+)\.',  # name_t001.ext (time series)
            r'(.+)_z([0-9]+)\.',  # name_z001.ext (z-stack)
        ]
        
        for file_info in metadata_collection:
            filename = file_info["name"]
            
            for pattern in sequence_patterns:
                match = re.match(pattern, filename)
                if match:
                    base_name = match.group(1)
                    sequence_num = int(match.group(2))
                    name_patterns[base_name].append((sequence_num, file_info))
                    break
        
        # Process potential sequences
        for base_name, files in name_patterns.items():
            if len(files) > 1:
                # Sort by sequence number
                files.sort(key=lambda x: x[0])
                
                # Determine sequence type based on metadata or filename
                sequence_type = "unknown"
                if any("_t" in f[1]["name"] for f in files):
                    sequence_type = "time_series"
                elif any("_z" in f[1]["name"] for f in files):
                    sequence_type = "z_stack"
                
                # Create sequence relationship
                sequence = {
                    "type": "sequence",
                    "sequence_type": sequence_type,
                    "base_name": base_name,
                    "files": [f[1]["path"] for f in files],
                    "description": f"{sequence_type.replace('_', ' ')} sequence with {len(files)} files"
                }
                
                relationships.append(sequence)
    
    def _find_parent_child_relationships(self, metadata_collection: List[Dict[str, Any]], relationships: List[Dict[str, Any]]) -> None:
        """
        Find parent-child relationships between files (e.g., raw/processed pairs).
        
        Args:
            metadata_collection: List of file metadata dictionaries
            relationships: List to append discovered relationships to
        """
        # Look for processing software mentions that might indicate derived files
        for i, file1 in enumerate(metadata_collection):
            for j, file2 in enumerate(metadata_collection):
                if i == j:
                    continue
                
                # Check if file2 mentions file1 in its processing metadata
                if "processing" in file2["metadata"] and "operations" in file2["metadata"]["processing"]:
                    operations = file2["metadata"]["processing"]["operations"]
                    
                    if operations and isinstance(operations, list):
                        for operation in operations:
                            if file1["name"] in str(operation):
                                # Likely a parent-child relationship
                                relationship = {
                                    "type": "derived",
                                    "parent": file1["path"],
                                    "child": file2["path"],
                                    "operation": operation,
                                    "description": f"Derived file through {operation}"
                                }
                                
                                relationships.append(relationship)
    
    def _discover_hierarchy(self, metadata_collection: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Discover hierarchical organization of the dataset.
        
        Args:
            metadata_collection: List of file metadata dictionaries
            
        Returns:
            Dictionary with hierarchy information
        """
        # Extract directory structure
        directories = set()
        for file_info in metadata_collection:
            path = Path(file_info["path"])
            parent_dir = path.parent
            directories.add(str(parent_dir))
        
        # Build directory tree
        dir_tree = {}
        for directory in sorted(directories):
            parts = Path(directory).parts
            current = dir_tree
            
            for part in parts:
                if part not in current:
                    current[part] = {}
                current = current[part]
        
        # Count files in each directory
        dir_counts = defaultdict(int)
        for file_info in metadata_collection:
            path = Path(file_info["path"])
            parent_dir = str(path.parent)
            dir_counts[parent_dir] += 1
        
        # Add file counts to tree
        def add_counts_to_tree(tree, path_prefix=""):
            result = {}
            
            for key, subtree in tree.items():
                current_path = os.path.join(path_prefix, key)
                count = dir_counts.get(current_path, 0)
                
                result[key] = {
                    "file_count": count,
                    "children": add_counts_to_tree(subtree, current_path)
                }
            
            return result
        
        hierarchy = {
            "directories": add_counts_to_tree(dir_tree),
            "total_files": len(metadata_collection),
            "total_directories": len(directories)
        }
        
        return hierarchy
    
    def generate_report(self, dataset_structure: Dict[str, Any], output_file: Union[str, Path]) -> None:
        """
        Generate a report of the dataset structure.
        
        Args:
            dataset_structure: Dataset structure dictionary from discover_dataset_structure
            output_file: Path to save the report
        """
        output_file = Path(output_file)
        
        # Create parent directories if they don't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate report content
        report = []
        report.append("# Electron Microscopy Dataset Report")
        report.append("")
        
        # Summary
        report.append("## Summary")
        report.append("")
        report.append(f"Total files: {len(dataset_structure['files'])}")
        report.append(f"Total directories: {dataset_structure['hierarchy']['total_directories']}")
        report.append("")
        
        # File formats
        formats = defaultdict(int)
        for file_info in dataset_structure["files"]:
            formats[file_info["format"]] += 1
        
        report.append("## File Formats")
        report.append("")
        for fmt, count in sorted(formats.items(), key=lambda x: x[1], reverse=True):
            report.append(f"- {fmt}: {count} files")
        report.append("")
        
        # Metadata patterns
        report.append("## Metadata Patterns")
        report.append("")
        
        patterns = dataset_structure["patterns"]
        for field, stats in sorted(patterns.items()):
            if stats["coverage"] > 0.5:  # Only show fields with >50% coverage
                report.append(f"### {field}")
                report.append(f"- Coverage: {stats['coverage']*100:.1f}%")
                report.append(f"- Type: {stats['type']}")
                
                if stats["is_identifier"]:
                    report.append(f"- Role: Identifier (unique for each file)")
                elif stats["is_categorical"]:
                    report.append(f"- Role: Categorical ({stats['cardinality']} categories)")
                
                if stats["example_values"]:
                    report.append(f"- Example values: {', '.join(str(v) for v in stats['example_values'])}")
                    
                report.append("")
        
        # Relationships
        if dataset_structure["relationships"]:
            report.append("## Relationships")
            report.append("")
            
            # Group by relationship type
            rel_by_type = defaultdict(list)
            for rel in dataset_structure["relationships"]:
                rel_by_type[rel["type"]].append(rel)
            
            for rel_type, rels in rel_by_type.items():
                report.append(f"### {rel_type.title()} Relationships")
                report.append(f"Found {len(rels)} {rel_type} relationships")
                report.append("")
                
                for i, rel in enumerate(rels[:5]):  # Show first 5 examples
                    report.append(f"#### Example {i+1}: {rel['description']}")
                    
                    if rel_type == "sequence":
                        report.append(f"- Base name: {rel['base_name']}")
                        report.append(f"- Sequence type: {rel['sequence_type']}")
                        report.append(f"- Files: {len(rel['files'])} files")
                    elif rel_type == "similarity":
                        report.append(f"- File 1: {Path(rel['file1']).name}")
                        report.append(f"- File 2: {Path(rel['file2']).name}")
                        report.append(f"- Similarity: {rel['similarity']:.2f}")
                    elif rel_type == "derived":
                        report.append(f"- Parent: {Path(rel['parent']).name}")
                        report.append(f"- Child: {Path(rel['child']).name}")
                        report.append(f"- Operation: {rel['operation']}")
                    
                    report.append("")
                
                if len(rels) > 5:
                    report.append(f"*...and {len(rels)-5} more {rel_type} relationships*")
                    report.append("")
        
        # Write report to file
        with open(output_file, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"Report saved to {output_file}")
    
    def generate_schema(self, dataset_structure: Dict[str, Any], output_file: Union[str, Path]) -> None:
        """
        Generate a schema based on the dataset structure.
        
        Args:
            dataset_structure: Dataset structure dictionary from discover_dataset_structure
            output_file: Path to save the schema
        """
        output_file = Path(output_file)
        
        # Create parent directories if they don't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate schema based on metadata patterns
        patterns = dataset_structure["patterns"]
        
        schema = {
            "title": "Electron Microscopy Dataset Schema",
            "description": "Automatically generated schema for electron microscopy dataset",
            "type": "object",
            "properties": {}
        }
        
        # Process each field with good coverage
        for field, stats in patterns.items():
            if stats["coverage"] > 0.7:  # Only include fields with >70% coverage
                # Split field path into parts
                parts = field.split('.')
                
                # Navigate to the correct position in the schema
                current = schema["properties"]
                for i, part in enumerate(parts[:-1]):
                    if part not in current:
                        current[part] = {
                            "type": "object",
                            "properties": {}
                        }
                    elif "properties" not in current[part]:
                        current[part]["properties"] = {}
                    
                    current = current[part]["properties"]
                
                # Add the field definition
                field_name = parts[-1]
                field_def = {
                    "description": f"{field_name} field"
                }
                
                # Set type based on field statistics
                if stats["type"] == "numeric":
                    field_def["type"] = "number"
                elif stats["type"] == "boolean":
                    field_def["type"] = "boolean"
                else:
                    field_def["type"] = "string"
                
                # Add enum for categorical fields with few values
                if stats["is_categorical"] and stats["cardinality"] <= 10:
                    field_def["enum"] = [str(v) for v in stats["example_values"]]
                
                current[field_name] = field_def
        
        # Write schema to file
        with open(output_file, 'w') as f:
            if output_file.suffix.lower() == '.json':
                json.dump(schema, f, indent=2)
            else:
                yaml.dump(schema, f, default_flow_style=False)
        
        print(f"Schema saved to {output_file}")
    
    def visualize_relationships(self, relationships: List[Dict[str, Any]], output_file: Union[str, Path]) -> None:
        """
        Visualize relationships between files.
        
        Args:
            relationships: List of relationship dictionaries
            output_file: Path to save the visualization
        """
        if not relationships:
            print("No relationships to visualize")
            return
        
        output_file = Path(output_file)
        
        # Create parent directories if they don't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes and edges based on relationships
        for rel in relationships:
            rel_type = rel["type"]
            
            if rel_type == "similarity":
                file1 = Path(rel["file1"]).name
                file2 = Path(rel["file2"]).name
                
                G.add_node(file1)
                G.add_node(file2)
                G.add_edge(file1, file2, type="similarity", weight=rel["similarity"])
            
            elif rel_type == "derived":
                parent = Path(rel["parent"]).name
                child = Path(rel["child"]).name
                
                G.add_node(parent)
                G.add_node(child)
                G.add_edge(parent, child, type="derived", operation=rel["operation"])
            
            elif rel_type == "sequence":
                # Add sequence as a special node
                sequence_node = f"Sequence: {rel['base_name']}"
                G.add_node(sequence_node, type="sequence")
                
                # Connect all files to the sequence node
                for file_path in rel["files"]:
                    file_name = Path(file_path).name
                    G.add_node(file_name)
                    G.add_edge(sequence_node, file_name, type="member")
        
        # Create visualization
        plt.figure(figsize=(12, 10))
        
        # Define node colors based on type
        node_colors = []
        for node in G.nodes():
            if G.nodes[node].get("type") == "sequence":
                node_colors.append("lightgreen")
            else:
                node_colors.append("lightblue")
        
        # Define edge colors based on type
        edge_colors = []
        for u, v, data in G.edges(data=True):
            if data.get("type") == "similarity":
                edge_colors.append("blue")
            elif data.get("type") == "derived":
                edge_colors.append("red")
            else:  # member
                edge_colors.append("green")
        
        # Draw the graph
        pos = nx.spring_layout(G, seed=42)  # Consistent layout
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, alpha=0.8)
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=1.5, alpha=0.7)
        nx.draw_networkx_labels(G, pos, font_size=8)
        
        plt.title("Electron Microscopy Dataset Relationships")
        plt.axis("off")
        
        # Save the visualization
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"Visualization saved to {output_file}")
