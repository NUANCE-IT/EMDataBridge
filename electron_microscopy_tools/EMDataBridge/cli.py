#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command-line interface for EMDataBridge

A unified command-line interface for working with electron microscopy data formats:
- Standardizing various formats to a common schema
- Translating between different formats
- Discovering self-describing data organization
- Managing your researcher identity (ORCID and signing key)
"""

import os
import sys
import argparse
from pathlib import Path

# The format tools are imported inside their commands, so `emdb identity` stays light.


def main():
    """Main entry point for the EMDataBridge CLI."""
    parser = argparse.ArgumentParser(
        description='EMDataBridge: A bridge between electron microscopy data formats',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
        Examples:
          # Standardize an MRC file and output metadata as JSON
          emdb standardize sample.mrc -o metadata.json
          
          # Translate a DM4 file to TIFF format
          emdb translate sample.dm4 sample.tiff
          
          # Discover data organization in a directory of EM files
          emdb discover /path/to/data -o report.md

          # Create your identity (ORCID + signing key), used by NUScope, NUScope-TOMO and NUANCE-MCP
          emdb identity create "Your Name" --orcid 0000-0000-0000-0000
        ''')
    
    # Create subparsers for each tool
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Standardizer subcommand
    standardize_parser = subparsers.add_parser('standardize', help='Standardize EM data formats')
    standardize_parser.add_argument('input_file', help='Input electron microscopy data file')
    standardize_parser.add_argument('--output', '-o', help='Output file for standardized metadata')
    standardize_parser.add_argument('--format', '-f', choices=['json', 'yaml'], default='json',
                                  help='Output format for metadata')
    standardize_parser.add_argument('--model', '-m', default='bert-base-uncased',
                                  help='LLM model to use for metadata extraction')
    
    # Translator subcommand
    translate_parser = subparsers.add_parser('translate', help='Translate between EM data formats')
    translate_parser.add_argument('input_file', help='Input electron microscopy data file')
    translate_parser.add_argument('output_file', help='Output file path')
    translate_parser.add_argument('--format', '-f', help='Output format (if not specified, inferred from output file extension)')
    translate_parser.add_argument('--model', '-m', default='bert-base-uncased',
                                help='LLM model to use for metadata mapping')
    translate_parser.add_argument('--list-formats', '-l', action='store_true',
                                help='List supported formats and exit')
    translate_parser.add_argument('--batch', '-b', action='store_true',
                                help='Batch mode: input_file is a directory, output_file is a directory')
    translate_parser.add_argument('--pattern', '-p', default='*',
                                help='File pattern for batch mode (default: *)')
    
    # Discovery subcommand
    discover_parser = subparsers.add_parser('discover', help='Discover self-describing data organization')
    discover_parser.add_argument('directory', help='Directory containing electron microscopy files')
    discover_parser.add_argument('--output', '-o', help='Output file for report')
    discover_parser.add_argument('--schema', '-s', help='Output file for generated schema')
    discover_parser.add_argument('--visualize', '-v', help='Output file for relationship visualization')
    discover_parser.add_argument('--recursive', '-r', action='store_true', default=True,
                               help='Search recursively in subdirectories')
    discover_parser.add_argument('--pattern', '-p', default='*',
                               help='File pattern to match (default: *)')
    discover_parser.add_argument('--model', '-m', default='t5-small',
                               help='LLM model to use for metadata analysis')
    
    # Identity subcommand
    id_parser = subparsers.add_parser('identity', help='Your ORCID identity and signing key')
    id_sub = id_parser.add_subparsers(dest='id_command', required=True)
    p = id_sub.add_parser('create', help='Create your identity and Ed25519 signing key')
    p.add_argument('name')
    p.add_argument('--orcid', help='ORCID iD, e.g. 0000-0002-1825-0097')
    p.add_argument('--overwrite', action='store_true', help='Replace an existing key')
    id_sub.add_parser('show', help='Show your identity and key fingerprint')
    p = id_sub.add_parser('export', help='Write your public identity to share with collaborators')
    p.add_argument('file')
    p = id_sub.add_parser('trust', help="Trust someone's public identity file")
    p.add_argument('file')
    p = id_sub.add_parser('untrust', help='Remove a key from your trusted list')
    p.add_argument('fingerprint')
    id_sub.add_parser('list', help='Keys you trust')
    p = id_sub.add_parser('check-orcid', help='Validate an ORCID iD (format and check digit)')
    p.add_argument('orcid')

    args = parser.parse_args()
    
    # Handle no command
    if args.command is None:
        parser.print_help()
        return 0
    
    try:
        # Execute the appropriate command
        if args.command == 'standardize':
            return standardize_command(args)
        elif args.command == 'translate':
            return translate_command(args)
        elif args.command == 'discover':
            return discover_command(args)
        elif args.command == 'identity':
            return identity_command(args)
        else:
            print(f"Unknown command: {args.command}")
            return 1
    except ImportError as e:
        print(f"Error: {e}\nThe format tools need extra packages: pip install \"EMDataBridge[tools,io]\"")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


def identity_command(args):
    """Execute the identity subcommands."""
    from . import identity
    c = args.id_command
    if c == 'create':
        ident = identity.create(args.name, orcid=args.orcid, overwrite=args.overwrite)
        print(f"Created identity {ident}")
        print(f"Private key: {identity.key_dir() / identity.KEY_FILE} (keep it safe; back it up)")
        print("Share your public identity with: emdb identity export me.json")
    elif c == 'show':
        ident = identity.me()
        if ident is None:
            print('No identity yet. Create one with: emdb identity create "Your Name" --orcid ...')
            return 1
        print(f"{ident.name}\nORCID       {ident.orcid or '-'}\nfingerprint {ident.fingerprint}\n"
              f"public key  {ident.public_key}\nstored in   {identity.key_dir()}")
    elif c == 'export':
        print(f"Wrote {identity.export(args.file)}")
    elif c == 'trust':
        print(f"Trusted {identity.trust(args.file)}")
    elif c == 'untrust':
        print("Removed" if identity.untrust(args.fingerprint) else "Not in your trusted list")
    elif c == 'list':
        mine = identity.me()
        for fp, k in identity.trusted().items():
            tag = "  (you)" if mine and fp == mine.fingerprint else ""
            print(f"{fp}  {k['name']}{'  ' + k['orcid'] if k.get('orcid') else ''}{tag}")
    elif c == 'check-orcid':
        print(f"Valid: {identity.normalize_orcid(args.orcid)}")
    return 0


def standardize_command(args):
    """Execute the standardize command."""
    from .standardizer import EMDataStandardizer
    # Create standardizer
    standardizer = EMDataStandardizer(llm_model_name=args.model)
    
    # Standardize the input file
    standardized = standardizer.standardize(args.input_file)
    
    # Print summary
    print(f"Standardized {args.input_file}")
    print(f"Original format: {standardized['original_format']}")
    print(f"Image dimensions: {standardized['data'].shape}")
    
    # Save metadata if output file specified
    if args.output:
        standardizer.save_standardized(standardized, args.output, format=args.format)
        print(f"Saved standardized metadata to {args.output}")
    else:
        # Print metadata to console
        import json
        import yaml
        if args.format == 'json':
            print(json.dumps(standardized['metadata'], indent=2))
        else:
            print(yaml.dump(standardized['metadata']))
    
    return 0


def translate_command(args):
    """Execute the translate command."""
    from .translator import EMFormatTranslator
    # Create translator
    translator = EMFormatTranslator(llm_model_name=args.model)
    
    # List formats if requested
    if args.list_formats:
        formats = translator.list_supported_formats()
        print("Supported formats:")
        for fmt in formats:
            print(f"\n{fmt['name']} ({', '.join(fmt['extensions'])})")
            print(f"  Description: {fmt['description']}")
            print(f"  Libraries: {', '.join(fmt['libraries'])}")
            print(f"  Metadata support: {fmt['metadata_support']}")
            print(f"  Typical use: {fmt['typical_use']}")
        return 0
    
    if args.batch:
        # Batch mode
        if not args.format:
            print("Error: Output format must be specified in batch mode")
            return 1
            
        translator.batch_translate(args.input_file, args.output_file, 
                                 args.format, file_pattern=args.pattern)
    else:
        # Single file mode
        translator.translate(args.input_file, args.output_file, args.format)
    
    return 0


def discover_command(args):
    """Execute the discover command."""
    from .discovery import EMDataDiscovery
    # Create discovery tool
    discovery = EMDataDiscovery(llm_model_name=args.model)
    
    # Discover dataset structure
    print(f"Analyzing directory: {args.directory}")
    dataset_structure = discovery.discover_dataset_structure(
        args.directory, recursive=args.recursive, file_pattern=args.pattern)
    
    print(f"Found {len(dataset_structure['files'])} files")
    
    # Generate report if output file specified
    if args.output:
        discovery.generate_report(dataset_structure, args.output)
    
    # Generate schema if schema file specified
    if args.schema:
        discovery.generate_schema(dataset_structure, args.schema)
    
    # Visualize relationships if visualization file specified
    if args.visualize:
        discovery.visualize_relationships(dataset_structure["relationships"], args.visualize)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
