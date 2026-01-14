import io
import os
import numpy as np
import pandas as pd
import streamlit as st

from Bio.PDB import PDBParser, Selection, NeighborSearch
from sklearn.cluster import DBSCAN
import py3Dmol
from stmol import showmol

st.set_page_config(page_title="Active-site / pocket predictor", layout="wide")

# ----------------------------
# Helpers
# ----------------------------
def _safe_chain_list(structure):
    chains = []
    for model in structure:
        for ch in model:
            if ch.id not in chains:
                chains.append(ch.id)
    return chains

def load_structure_from_bytes(pdb_bytes: bytes, name="prot"):
    parser = PDBParser(QUIET=True)
    handle = io.StringIO(pdb_bytes.decode("utf-8", errors="ignore"))
    structure = parser.get_structure(name, handle)
    return structure

def get_atoms(structure, chains_selected=None):
    atoms = []
    for model in structure:
        for chain in model:
            if chains_selected and chain.id not in chains_selected:
                continue
            for residue in chain:
                # skip waters
                if residue.id[0].strip() != "":
                    continue
                for atom in residue:
                    # skip altloc non-primary if present
                    altloc = atom.get_altloc()
                    if altloc not in (" ", "A"):
                        continue
                    atoms.append(atom)
    return atoms

def protein_bbox(atom_coords, padding=3.0):
    mn = atom_coords.min(axis=0) - padding
    mx = atom_coords.max(axis=0) + padding
    return mn, mx

def build_grid(mn, mx, spacing):
    xs = np.arange(mn[0], mx[0] + spacing, spacing)
    ys = np.arange(mn[1], mx[1] + spacing, spacing)
    zs = np.arange(mn[2], mx[2] + spacing, spacing)
    grid = np.array(np.meshgrid(xs, ys, zs, indexing="ij")).reshape(3, -1).T
    return grid

def pocket_points_from_grid(grid_xyz, atom_xyz, min_dist, max_dist):
    # distance to nearest protein atom
    from scipy.spatial import cKDTree
    tree = cKDTree(atom_xyz)
    d, _ = tree.query(grid_xyz, k=1, workers=-1)
    # candidate pocket points: not inside protein, but close to surface
    mask = (d >= min_dist) & (d <= max_dist)
    return grid_xyz[mask], d[mask]

def cluster_points(points_xyz, eps, min_samples):
    if len(points_xyz) == 0:
        return np.array([]), 0
    cl = DBSCAN(eps=eps, min_samples=min_samples).fit(points_xyz)
    labels = cl.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return labels, n_clusters

def estimate_volume(n_points, spacing):
    return float(n_points) * (float(spacing) ** 3)

def estimate_area(n_points, spacing):
    # crude proxy: scales with exposed "shell"; keep simple & stable
    return float(n_points) * (float(spacing) ** 2)

def residues_near_centroid(structure, centroid, radius, top_n, chains_selected=None):
    atoms = get_atoms(structure, chains_selected)
    if len(atoms) == 0:
        return []

    ns = NeighborSearch(atoms)
    residues = ns.search(np.array(centroid, dtype=float), float(radius), level="R")

    # rank residues by distance of their CA (or centroid of atoms) to pocket centroid
    ranked = []
    for res in residues:
        # skip hetero/water
        if res.id[0].strip() != "":
            continue
        coords = np.array([a.get_coord() for a in res.get_atoms()])
        res_cent = coords.mean(axis=0)
        dist = float(np.linalg.norm(res_cent - centroid))
        ranked.append((dist, res))

    ranked.sort(key=lambda x: x[0])
    out = []
    seen = set()
    for dist, res in ranked:
        chain_id = res.get_parent().id
        resname = res.get_resname()
        resnum = res.id[1]
        icode = res.id[2].strip() if isinstance(res.id[2], str) else ""
        key = (chain_id, resnum, icode, resname)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "chain": chain_id,
                "resname": resname,
                "resnum": int(resnum),
                "icode": icode,
                "distance_to_centroid": dist,
            }
        )
        if len(out) >= int(top_n):
            break
    return out

def points_to_pdb_bytes(points_xyz, name="PCK"):
    # write pocket points as HETATM pseudo-atoms
    lines = []
    for i, (x, y, z) in enumerate(points_xyz, start=1):
        lines.append(
            f"HETATM{i:5d}  C   {name} A{1:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        )
    lines.append("END")
    return ("\n".join(lines) + "\n").encode("utf-8")

@st.cache_data(show_spinner=False)
def df_to_csv_bytes(df: pd.DataFrame):
    return df.to_csv(index=False).encode("utf-8")

def build_viewer(pdb_text, pocket_sets, sphere_radius=1.0):
    v = py3Dmol.view(width=900, height=650)
    v.addModel(pdb_text, "pdb")
    v.setStyle({"cartoon": {"color": "spectrum"}})

    # pocket points as spheres, one color per pocket
    palette = ["red", "orange", "yellow", "lime", "cyan", "blue", "magenta", "white"]
    for i, pts in enumerate(pocket_sets):
        color = palette[i % len(palette)]
        for (x, y, z) in pts:
            v.addSphere({"center": {"x": float(x), "y": float(y), "z": float(z)},
                         "radius": float(sphere_radius),
                         "color": color,
                         "opacity": 0.85})
    v.zoomTo()
    return v


# ----------------------------
# UI
# ----------------------------
st.title("Active-site / pocket prediction (offline heuristic)")

left, right = st.columns([0.42, 0.58], gap="large")

with left:
    st.subheader("Input")
    up = st.file_uploader("Upload PDB file", type=["pdb"])

    st.subheader("Parameters")
    spacing = st.slider("Grid spacing (Å)", 0.5, 2.5, 1.0, 0.1)
    bbox_padding = st.slider("BBox padding (Å)", 1.0, 10.0, 3.0, 0.5)

    min_dist = st.slider("Min distance from protein (Å)", 0.5, 3.0, 1.4, 0.1)
    max_dist = st.slider("Max distance from protein (Å)", 2.0, 10.0, 5.0, 0.1)

    eps = st.slider("DBSCAN eps (Å)", 1.0, 6.0, 2.5, 0.1)
    min_samples = st.slider("DBSCAN min_samples", 3, 30, 6, 1)

    top_pockets = st.slider("Top pockets (by volume)", 1, 10, 3, 1)

    residue_radius = st.slider("Residue assignment radius (Å)", 2.0, 12.0, 6.0, 0.5)
    top_residues = st.slider("Top residues per pocket (closest to centroid)", 3, 30, 10, 1)

    sphere_radius = st.slider("Pocket-point sphere radius (Å)", 0.3, 2.5, 0.8, 0.1)

    # Initialize session state for run button
    if 'run_prediction' not in st.session_state:
        st.session_state.run_prediction = False
    
    # Button click sets the state
    if = st.button("Run prediction", type="primary", use_container_width=True):
        st.session_state.run_prediction = True

with right:
    st.subheader("3D view + downloads")

    if up is None:
        st.info("Upload a PDB to begin.")
        st.stop()

    pdb_bytes = up.getvalue()
    pdb_text = pdb_bytes.decode("utf-8", errors="ignore")

    try:
        structure = load_structure_from_bytes(pdb_bytes, name="query")
    except Exception as e:
        st.error(f"Failed to parse PDB: {e}")
        st.stop()

    all_chains = _safe_chain_list(structure)
    chains_selected = st.multiselect("Chains to include", options=all_chains, default=all_chains)

    if run:
        atoms = get_atoms(structure, chains_selected)
        if len(atoms) == 0:
            st.error("No atoms found after chain/altloc/water filtering.")
            st.stop()

        atom_xyz = np.array([a.get_coord() for a in atoms], dtype=float)
        mn, mx = protein_bbox(atom_xyz, padding=bbox_padding)

        with st.spinner("Building grid..."):
            grid = build_grid(mn, mx, spacing)

        with st.spinner("Finding pocket candidate points..."):
            pocket_xyz, pocket_d = pocket_points_from_grid(grid, atom_xyz, min_dist, max_dist)

        if len(pocket_xyz) == 0:
            st.warning("No pocket points found with current distance thresholds.")
            # Still show protein
            viewer = build_viewer(pdb_text, [], sphere_radius=sphere_radius)
            showmol(viewer, height=650, width=900)
            st.stop()

        with st.spinner("Clustering pocket points (DBSCAN)..."):
            labels, n_clusters = cluster_points(pocket_xyz, eps=eps, min_samples=min_samples)

        if n_clusters == 0:
            st.warning("No clusters found (all points labeled as noise). Try increasing eps or lowering min_samples.")
            viewer = build_viewer(pdb_text, [], sphere_radius=sphere_radius)
            showmol(viewer, height=650, width=900)
            st.stop()

        # pocket summary
        pockets = []
        for lab in sorted([l for l in set(labels) if l != -1]):
            pts = pocket_xyz[labels == lab]
            centroid = pts.mean(axis=0)
            pockets.append(
                {
                    "pocket_id": int(lab),
                    "n_points": int(len(pts)),
                    "centroid_x": float(centroid[0]),
                    "centroid_y": float(centroid[1]),
                    "centroid_z": float(centroid[2]),
                    "volume_A3": estimate_volume(len(pts), spacing),
                    "area_proxy_A2": estimate_area(len(pts), spacing),
                }
            )

        pockets_df = pd.DataFrame(pockets).sort_values("volume_A3", ascending=False).reset_index(drop=True)
        pockets_df["rank"] = np.arange(1, len(pockets_df) + 1)

        # keep top pockets
        pockets_df = pockets_df.head(int(top_pockets)).copy()

        # residue table (one row per pocket-residue)
        residue_rows = []
        pocket_point_sets = []
        pocket_pdb_files = []

        for _, row in pockets_df.iterrows():
            pid = int(row["pocket_id"])
            pts = pocket_xyz[labels == pid]
            pocket_point_sets.append(pts)

            centroid = np.array([row["centroid_x"], row["centroid_y"], row["centroid_z"]], dtype=float)
            top_res = residues_near_centroid(
                structure, centroid=centroid, radius=residue_radius, top_n=top_residues, chains_selected=chains_selected
            )

            for r in top_res:
                residue_rows.append(
                    {
                        "rank": int(row["rank"]),
                        "pocket_id": pid,
                        "volume_A3": float(row["volume_A3"]),
                        "area_proxy_A2": float(row["area_proxy_A2"]),
                        "centroid_x": float(row["centroid_x"]),
                        "centroid_y": float(row["centroid_y"]),
                        "centroid_z": float(row["centroid_z"]),
                        "chain": r["chain"],
                        "resname": r["resname"],
                        "resnum": r["resnum"],
                        "icode": r["icode"],
                        "distance_to_centroid": float(r["distance_to_centroid"]),
                    }
                )

            pocket_pdb_files.append((f"pocket_rank{int(row['rank'])}_id{pid}.pdb", points_to_pdb_bytes(pts, name="PCK")))

        res_df = pd.DataFrame(residue_rows)

        # render viewer
        viewer = build_viewer(pdb_text, pocket_point_sets, sphere_radius=sphere_radius)
        showmol(viewer, height=650, width=900)

        # show tables
        st.markdown("### Pocket summary")
        st.dataframe(pockets_df, use_container_width=True)

        st.markdown("### Top residues per pocket")
        if len(res_df) == 0:
            st.warning("No residues found near pocket centroids with current residue radius.")
        else:
            st.dataframe(res_df, use_container_width=True)

        # downloads
        st.markdown("### Downloads")
        st.download_button(
            "Download pocket summary CSV",
            data=df_to_csv_bytes(pockets_df),
            file_name="pocket_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.download_button(
            "Download pocket residues CSV",
            data=df_to_csv_bytes(res_df) if len(res_df) else b"rank,pocket_id\n",
            file_name="pocket_top_residues.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # pocket point PDB downloads (separate buttons)
        for fname, blob in pocket_pdb_files:
            st.download_button(
                f"Download {fname}",
                data=blob,
                file_name=fname,
                mime="chemical/x-pdb",
                use_container_width=True,
            )
    else:
        # show protein only (pre-run)
        viewer = build_viewer(pdb_text, [], sphere_radius=sphere_radius)
        showmol(viewer, height=650, width=900)
