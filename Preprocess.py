#!/usr/bin/env python3
"""
LIDC-IDRI Lung Nodule Patch Dataset Creation Script

This script extracts representative 2D patches from the LIDC-IDRI (Lung Image Database 
Consortium and Image Database Resource Initiative) dataset for machine learning applications 
in lung nodule analysis.

Key Features:
- Multi-radiologist Consensus: Combines annotations from multiple radiologists
- Intelligent Slice Selection: Selects representative slices based on nodule area
- Standardized CT Windowing: Applies lung window settings for consistent visualization
- Flexible Output: Generates both individual patch images and comprehensive metadata CSV files
- Robust Fallback Strategies: Handles edge cases with empty consensus masks

Usage:
    python Preprocess.py

Author: Generated from [Patch_Dataset_Making]Lidc_ENG.ipynb
"""

import os
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import pylidc as pl
from pylidc.utils import consensus
from tqdm import tqdm
from PIL import Image

# ===============================================================================
# 1) UTILITY FUNCTIONS
# ===============================================================================

# Annotation fields from LIDC-IDRI dataset (9 semantic attributes)
ANNOT_FIELDS = [
    'subtlety', 'internalStructure', 'calcification',
    'sphericity', 'margin', 'lobulation', 'spiculation',
    'texture', 'malignancy',
]

def _inplane_area_mm2(scan) -> float:
    """
    Calculate the area of each pixel in the axial plane (mm^2).
    
    Args:
        scan: pylidc Scan object
        
    Returns:
        float: Area per pixel in mm^2
    """
    ps = scan.pixel_spacing  # Pixel spacing from pylidc
    try:
        sx, sy = float(ps[0]), float(ps[1])
    except Exception:
        sx = sy = float(ps)
    return sx * sy  # mm^2/px

def _aggregate_ann_fields(anns, agg="round") -> Dict[str, int]:
    """
    Aggregate annotations from multiple radiologists for the same nodule.
    
    Annotation fields include: 'subtlety', 'internalStructure', 'calcification',
    'sphericity', 'margin', 'lobulation', 'spiculation', 'texture', 'malignancy'.
    
    Args:
        anns: List of pylidc Annotation objects from multiple radiologists
        agg: Aggregation method - "round" (mean then round) or "mean" (truncated mean)
        
    Returns:
        Dict[str, int]: Aggregated annotation scores
    """
    out = {}
    for f in ANNOT_FIELDS:
        vals = [getattr(a, f) for a in anns if getattr(a, f) is not None]
        if agg == "round":
            out[f] = int(np.rint(np.mean(vals))) if len(vals) else None
        elif agg == "mean":
            out[f] = int(np.mean(vals)) if len(vals) else None
    return out

def get_final_indices(kept, neigh, mode="intersect"):
    """
    Combine area-filtered slices with max-area neighboring slices.
    
    Args:
        kept: Slice indices that pass area threshold
        neigh: Neighboring slice indices around max-area slice
        mode: "union" (combine all) or "intersect" (only overlapping)
        
    Returns:
        np.ndarray: Final selected slice indices
    """
    if mode == "union":
        return np.unique(np.concatenate([kept, np.array(neigh, dtype=int)]))
    elif mode == "intersect":
        return np.intersect1d(kept, neigh)
    else:
        raise ValueError("mode must be 'union' or 'intersect'")

def window_hu(img_hu, wl=-600, ww=1500):
    """
    Apply CT windowing (lung window) to convert HU values to 0-255 grayscale.
    
    Standard lung window: WL=-600, WW=1500 (range: -1350 to 150 HU)
    
    Args:
        img_hu: CT image in Hounsfield Units
        wl: Window level (center)
        ww: Window width
        
    Returns:
        np.ndarray: 8-bit grayscale image (0-255)
    """
    lo, hi = wl - ww/2.0, wl + ww/2.0
    x = np.clip(img_hu, lo, hi)
    x = (x - lo) / (hi - lo)
    return (x * 255).astype(np.uint8)

def _select_repr_slices_from_cmask(
    cmask: np.ndarray,
    scan,
    min_area_mm2: float = 50.0,   # Minimum nodule area threshold (adjustable)
    drop_ends: bool = False,      # Whether to drop first/last slices
    n_neighbors: int = 2,         # Number of neighboring slices around max-area slice
    mode: str = "intersect"       # Intersection or union of filtering criteria
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select representative slices from consensus volume mask based on:
    1. Optional removal of first/last slices (boundary artifacts)
    2. Area-based filtering (remove slices with area < min_area_mm2)
    3. Forced inclusion of max-area slice ± neighboring slices
    
    Args:
        cmask: 3D consensus mask (H, W, K)
        scan: pylidc Scan object for pixel spacing
        min_area_mm2: Minimum area threshold in mm^2
        drop_ends: Whether to exclude first/last slices
        n_neighbors: Number of slices above/below max-area slice to include
        mode: "intersect" or "union" for combining area and neighbor criteria
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (selected slice indices, corresponding areas)
    """
    assert cmask.ndim == 3, "cmask expected to be (H, W, K)"
    H, W, K = cmask.shape

    # Calculate area for each slice
    px_area = _inplane_area_mm2(scan)
    areas = np.array([(cmask[:, :, k].sum()) * px_area for k in range(K)])

    # Initialize slice indices
    idx = np.arange(K)
    if drop_ends and K >= 3:
        idx = idx[1:-1]  # Remove first and last slices

    # Filter by area threshold
    kept = idx[areas[idx] >= float(min_area_mm2)]
    if kept.size == 0:
        # Fallback: keep at least the max-area slice
        kept = np.array([int(np.argmax(areas))])

    # Always include max-area slice and its neighbors
    k0 = int(np.argmax(areas))
    neigh = [k0 + d for d in range(-n_neighbors, n_neighbors + 1) 
             if 0 <= k0 + d < K]  # Clamp to valid range
    
    # Combine filtering results
    kept_final = get_final_indices(kept, neigh, mode=mode)

    return kept_final, areas[kept_final]

# ===============================================================================
# 2) CORE FUNCTION: Extract patches for a single nodule
# ===============================================================================

def extract_patches_for_nodule(
    scan,
    anns: List[pl.Annotation],
    clevel: float = 0.5,
    pad: List[Tuple[int, int]] = [(25, 25), (25, 25), (0, 0)],
    drop_ends: bool = False,
    min_area_mm2: float = 50.0,
    n_neighbors: int = 2,
    mode: str = "intersect",
) -> Dict[str, Any]:
    """
    Extract representative 2D patches from a 3D nodule annotation.
    
    Process:
    1. Generate consensus mask from multiple radiologist annotations
    2. Apply fallback strategies if consensus is empty
    3. Select representative slices based on area and spatial criteria
    4. Extract 2D image/mask patches for each selected slice
    
    Args:
        scan: pylidc Scan object
        anns: List of Annotation objects from multiple radiologists
        clevel: Consensus level (0.5 = majority agreement)
        pad: Padding around nodule bounding box [(x_pad), (y_pad), (z_pad)]
        drop_ends: Whether to exclude first/last slices of the nodule
        min_area_mm2: Minimum area threshold for slice selection
        n_neighbors: Number of neighboring slices around max-area slice
        mode: Slice selection mode ("intersect" or "union")
        
    Returns:
        Dict containing:
        - 'bbox': Consensus bounding box
        - 'patches': List of patch dictionaries with img, mask, indices, area
        - 'ann_agg': Aggregated annotation scores from multiple radiologists
    """
    # Load full CT volume
    vol = scan.to_volume()  # Shape: (H, W, K) in HU values
    
    # Generate consensus mask (with fallback strategies)
    cmask, cbbox, masks = consensus(anns, clevel=clevel, pad=pad)
    
    if cmask.sum() == 0:
        # Fallback 1: Lower consensus threshold
        for cl in (0.25, 0.1):
            cmask_try, _, _ = consensus(anns, clevel=cl, pad=pad)
            if cmask_try.sum() > 0:
                cmask = cmask_try
                break
        
        # Fallback 2: Use union of all individual masks
        if cmask.sum() == 0 and len(masks) > 0:
            cmask = np.any(np.stack(masks, axis=0), axis=0).astype(bool)

    # Extract subvolume using consensus bounding box
    subvol = vol[cbbox]
    
    # Select representative slices
    kept_k, kept_areas = _select_repr_slices_from_cmask(
        cmask, scan, min_area_mm2=min_area_mm2, n_neighbors=n_neighbors, mode=mode,
        drop_ends=drop_ends
    )

    # Generate 2D patches for selected slices
    patches = []
    for k, k_area in zip(kept_k, kept_areas):
        img2d = subvol[:, :, int(k)]  # 2D CT slice
        m2d = cmask[:, :, int(k)].astype(bool)  # 2D binary mask
        
        patches.append({
            "img": img2d,                         # 2D HU patch
            "mask": m2d,                          # 2D boolean mask
            "k_local": int(k),                    # Index within bounding box
            "k_global": int(cbbox[2].start + k),  # Index in original volume
            "area": k_area,                       # Slice area in mm^2
        })

    # Aggregate annotations from multiple radiologists
    ann_agg = _aggregate_ann_fields(anns)

    return {"bbox": cbbox, "patches": patches, "ann_agg": ann_agg}

# ===============================================================================
# 3) TOP-LEVEL: Patient-wise batch processing
# ===============================================================================

def extract_patient_level_patches(
    patient_id: str = None,
    clevel: float = 0.5,
    pad: List[Tuple[int, int]] = [(25, 25), (25, 25), (0, 0)],
    drop_ends: bool = False,
    min_area_mm2: float = 50.0,
    n_neighbors: int = 2,
    mode: str = "intersect"
):
    """
    Extract patches for all nodules in specified patient(s).
    
    Processing pipeline:
    1. Query LIDC-IDRI database for patient scan(s)
    2. Cluster annotations by spatial proximity (nodule grouping)
    3. Extract representative patches for each nodule
    4. Yield structured results for downstream processing
    
    Args:
        patient_id: Specific patient ID (e.g., "LIDC-IDRI-0001") or None for all
        clevel: Consensus level for multi-radiologist agreement
        pad: Bounding box padding
        drop_ends: Whether to exclude first/last slices of each nodule
        min_area_mm2: Minimum area threshold for slice filtering
        n_neighbors: Number of neighboring slices around max-area slice
        mode: Slice selection mode ("intersect" or "union")
        
    Yields:
        Dict containing patient metadata and extracted patches
    """
    q = pl.query(pl.Scan)
    if patient_id:
        q = q.filter(pl.Scan.patient_id == patient_id)
    
    for scan in q:  # Each scan = one patient CT examination
        nodule_groups = scan.cluster_annotations()  # Group annotations by spatial proximity
        
        for n_idx, anns in enumerate(nodule_groups):  # Each nodule cluster
            res = extract_patches_for_nodule(
                scan, anns,
                clevel=clevel, pad=pad,
                min_area_mm2=min_area_mm2, n_neighbors=n_neighbors,
                mode=mode, drop_ends=drop_ends
            )
            
            yield {
                "patient_id": scan.patient_id,
                "scan_id": scan.id,
                "nodule_index": int(n_idx),
                "bbox": res["bbox"],
                "ann_summary": res["ann_agg"],     # Aggregated annotation scores
                "patches": res["patches"],         # Representative slice patches
            }

def save_patches_and_metadata(
    out_dir: str,
    patient_id: str,
    metadata_csv: str,
    clevel: float = 0.5,
    pad: List[Tuple[int, int]] = [(25, 25), (25, 25), (0, 0)],
    drop_ends: bool = False,
    min_area_mm2: float = 50.0,
    n_neighbors: int = 2,
    mode: str = "intersect",
):
    """
    Extract patches and save as images with metadata CSV.
    
    Output structure:
    - Individual PNG files for each patch (image and mask)
    - CSV metadata file with patch information and annotation scores
    
    Args:
        out_dir: Output directory for patch images
        patient_id: Target patient ID (LIDC-IDRI-xxxx format)
        metadata_csv: Path for output CSV metadata file
        clevel: Consensus level for multi-radiologist agreement
        pad: Bounding box padding
        drop_ends: Whether to exclude first/last slices of each nodule
        min_area_mm2: Minimum area threshold (mm^2)
        n_neighbors: Number of neighboring slices around max-area slice
        mode: Slice selection mode ("intersect" or "union")
        
    Returns:
        pd.DataFrame: Generated metadata with patch information
    """
    os.makedirs(out_dir, exist_ok=True)
    records = []
    
    for item in extract_patient_level_patches(
        patient_id=patient_id,
        clevel=clevel,
        pad=pad,
        min_area_mm2=min_area_mm2,
        n_neighbors=n_neighbors,
        mode=mode,
        drop_ends=drop_ends
    ):
        sid = item["scan_id"]
        pid = item["patient_id"]
        nid = item["nodule_index"]

        for p in item["patches"]:
            k_global = p["k_global"]
            img = p["img"]
            mask = p["mask"]

            # Generate file paths
            img_path = os.path.join(out_dir, f"{pid}_n{nid}_k{k_global}_img.png")
            mask_path = os.path.join(out_dir, f"{pid}_n{nid}_k{k_global}_mask.png")

            # Apply CT windowing for standardized visualization
            img_windowed = window_hu(img)  # Lung window: filters noise and artifacts
            
            # Save images (avoiding per-image normalization for consistency)
            Image.fromarray(img_windowed).save(img_path)
            Image.fromarray((mask.astype(np.uint8) * 255)).save(mask_path)

            # bbox information
            nodule_bbox = item["bbox"]
            nodule_bbox = (
                nodule_bbox[0].start, nodule_bbox[1].start,
                nodule_bbox[0].stop, nodule_bbox[1].stop
            )  # (x_min, y_min, x_max, y_max)

            # Build metadata record
            record = {
                "scan_id": sid,
                "patient_id": pid,
                "nodule_index": nid,
                "k_global": k_global,
                "img_path": img_path,
                "mask_path": mask_path,
                "area_mm2": p["area"],
                "nodule_bbox_xmin": nodule_bbox[0],
                "nodule_bbox_ymin": nodule_bbox[1],
                "nodule_bbox_xmax": nodule_bbox[2],
                "nodule_bbox_ymax": nodule_bbox[3],
            }
            # Add aggregated annotation scores
            record.update({f"ann_{k}": v for k, v in item["ann_summary"].items()})

            records.append(record)

    # Save metadata to CSV
    df = pd.DataFrame(records)
    df.to_csv(metadata_csv, index=False)
    print(f"Metadata saved to {metadata_csv}, total patches: {len(df)}")

    return df

# ===============================================================================
# 4) MAIN EXECUTION FUNCTIONS
# ===============================================================================


def generate_scan_metadata():
    """
    Generate overview table for all patients in database
    """
    print("Generating scan metadata for all patients...")
    
    # Query all scans in LIDC-IDRI database
    scans_all = pl.query(pl.Scan)
    scans_all = scans_all.all()

    # Create metadata DataFrame with scan information
    scan_metainfo = pd.DataFrame(scans_all, columns=['Scan_obj'])
    scan_metainfo['id'] = scan_metainfo['Scan_obj'].apply(lambda x: x.id)
    scan_metainfo['patient_id'] = scan_metainfo['Scan_obj'].apply(lambda x: x.patient_id)
    scan_metainfo['num_annotations'] = scan_metainfo['Scan_obj'].apply(lambda x: len(x.annotations))

    print(f"Found {len(scan_metainfo)} scans in database")
    return scan_metainfo

def batch_process_all_patients():
    """
    Batch processing: Extract patches for all patients in database
    """
    print("Starting batch processing for all patients...")
    
    metadata_df_all = []
    directory = "./lidc_patches_all"
    os.makedirs(directory, exist_ok=True)

    # Get scan metadata first
    scan_metainfo = generate_scan_metadata()
    
    # Get list of all unique patient IDs
    # patient_list = scan_metainfo.patient_id.unique()
    # For testing with subset, uncomment the line below and comment the line above:
    patient_list = ["LIDC-IDRI-0078", "LIDC-IDRI-0151", "LIDC-IDRI-0115", "LIDC-IDRI-0054"]

    # Process each patient sequentially with progress bar
    for patient in tqdm(patient_list, desc="Processing patients"):
        out_dir = os.path.join(directory, patient)
        os.makedirs(out_dir, exist_ok=True)
        target_pid = patient

        # Extract patches for current patient
        metadata_df_patient = save_patches_and_metadata(
            out_dir=out_dir,
            metadata_csv=os.path.join(out_dir, "patches_metadata.csv"),
            clevel=0.5,
            pad=[(25, 25), (25, 25), (0, 0)],
            drop_ends=False,
            min_area_mm2=50.0,
            n_neighbors=2,
            mode="intersect",
            patient_id=target_pid
        )

        # Accumulate metadata from all patients
        metadata_df_all.append(metadata_df_patient)

    # Combine all patient metadata into single DataFrame
    metadata_df = pd.concat(metadata_df_all, ignore_index=True)
    metadata_df.to_csv(os.path.join(directory, "all_patches_metadata.csv"), index=False)

    print(f"Batch processing complete!")
    print(f"Output directory: {directory}")
    print(f"Total patches extracted: {len(metadata_df)}")
    print(f"Patients processed: {len(patient_list)}")

# ===============================================================================
# 5) MAIN EXECUTION
# ===============================================================================

def main():
    """
    Main execution function with different processing options
    """
    print("LIDC-IDRI Lung Nodule Patch Dataset Creation")
    print("=" * 50)

    # Uncomment the lines below to process all patients in the database
    # WARNING: This will process all 1010+ patients and may take a very long time!

    print("\nBatch process all patients")
    batch_process_all_patients()

    print("\nProcessing complete!")

if __name__ == "__main__":
    main()
