"""
src/heartprint/puzzle.py

Deterministic 20-piece Heart Puzzle representation and subspace stability mapping.
Maps the 256-dimensional neural cardiac embedding into 20 representation groups.
Pieces unlock non-destructively through repeated evidence of representation stability.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from src.config import HeartPrintConfig, HEARTPRINT_CFG


@dataclass
class PuzzlePieceInfo:
    piece_id: str             # "piece_001" .. "piece_020"
    index: int                # 0 .. 19
    dim_start: int            # Start index in 256-d embedding
    dim_end: int              # End index in 256-d embedding
    dim_count: int            # Number of latent dimensions in this piece
    grid_row: int             # Row index in visual heart grid
    grid_col: int             # Col index in visual heart grid
    center_x: float           # Normalized center x [-1.0, 1.0]
    center_y: float           # Normalized center y [-1.0, 1.0]
    svg_path: str             # SVG path descriptor for frontend rendering


class HeartPuzzle:
    """
    Manages deterministic piece definitions, embedding subspace mapping,
    and piece stability evaluation.
    """

    def __init__(self, cfg: HeartPrintConfig | None = None, total_dims: int = 256):
        self.cfg = cfg or HEARTPRINT_CFG
        self.total_pieces = self.cfg.total_pieces
        self.total_dims = total_dims
        self.pieces = self._generate_pieces()

    def _generate_pieces(self) -> list[PuzzlePieceInfo]:
        """
        Deterministically creates 20 puzzle pieces bounded inside a heart curve.
        Splits 256 embedding dimensions evenly across the 20 pieces.
        """
        # 1. Coordinate layout inside mathematical heart mask
        # Heart equation: (x^2 + y^2 - 1)^3 - x^2 * y^3 <= 0
        # In a discrete 6x6 grid, find cells inside the heart
        grid_coords = [
            # Row 0 (top lobes)
            (0, 1, -0.50, 0.70), (0, 2, -0.22, 0.75), (0, 3, 0.22, 0.75), (0, 4, 0.50, 0.70),
            # Row 1 (upper body)
            (1, 0, -0.75, 0.40), (1, 1, -0.45, 0.45), (1, 2, -0.15, 0.50), (1, 3, 0.15, 0.50), (1, 4, 0.45, 0.45), (1, 5, 0.75, 0.40),
            # Row 2 (mid body)
            (2, 0, -0.70, 0.10), (2, 1, -0.40, 0.15), (2, 2, -0.12, 0.20), (2, 3, 0.12, 0.20), (2, 4, 0.40, 0.15), (2, 5, 0.70, 0.10),
            # Row 3 (lower taper)
            (3, 1, -0.45, -0.25), (3, 2, -0.18, -0.20), (3, 3, 0.18, -0.20), (3, 4, 0.45, -0.25),
            # Row 4 (bottom tip)
            (4, 2, -0.15, -0.60), (4, 3, 0.15, -0.60),
            # Row 5 (apex)
            (5, 2, 0.00, -0.85),
        ]

        # Select exactly 20 canonical positions deterministically
        selected_coords = grid_coords[:20]

        # 2. Embedding dimension slicing
        base_size = self.total_dims // self.total_pieces  # 12
        remainder = self.total_dims % self.total_pieces   # 16
        # First 16 pieces get 13 dims, last 4 get 12 dims -> 16*13 + 4*12 = 256 dims

        pieces = []
        cur_dim = 0
        for i, (row, col, cx, cy) in enumerate(selected_coords):
            piece_id = f"piece_{i+1:03d}"
            dim_count = base_size + (1 if i < remainder else 0)
            dim_start = cur_dim
            dim_end = cur_dim + dim_count
            cur_dim = dim_end

            # Generate SVG polygonal cell boundary (width ~ 0.25, height ~ 0.28)
            hw = 0.12
            hh = 0.13
            # Normalized SVG viewport 0..100
            # map cx [-1, 1] -> [10, 90], cy [1, -1] -> [10, 90]
            svg_x = 50 + cx * 42
            svg_y = 50 - cy * 42
            svg_w = hw * 84
            svg_h = hh * 84

            # Clean standard SVG polygon path for each tile
            x1 = svg_x - svg_w / 2.0
            x2 = svg_x + svg_w / 2.0
            y1 = svg_y - svg_h / 2.0
            y2 = svg_y + svg_h / 2.0
            svg_path = f"M {x1:.1f} {y1:.1f} L {x2:.1f} {y1:.1f} L {x2:.1f} {y2:.1f} L {x1:.1f} {y2:.1f} Z"

            pieces.append(
                PuzzlePieceInfo(
                    piece_id=piece_id,
                    index=i,
                    dim_start=dim_start,
                    dim_end=dim_end,
                    dim_count=dim_count,
                    grid_row=row,
                    grid_col=col,
                    center_x=cx,
                    center_y=cy,
                    svg_path=svg_path,
                )
            )

        return pieces

    def evaluate_piece_stability(
        self,
        current_embedding: np.ndarray,
        reference_embedding: np.ndarray | None,
        existing_piece_states: dict[str, dict] | None = None,
    ) -> dict[str, dict]:
        """
        Evaluates stability for each piece's subspace slice.
        Returns a dictionary mapping piece_id -> {
            "established": bool,
            "stability": float,
            "observations": int,
            "just_unlocked": bool
        }
        """
        existing_states = existing_piece_states or {}
        curr_emb = np.asarray(current_embedding, dtype=np.float64).flatten()
        ref_emb = (
            np.asarray(reference_embedding, dtype=np.float64).flatten()
            if reference_embedding is not None
            else curr_emb
        )

        results = {}
        for piece in self.pieces:
            pid = piece.piece_id
            prior_info = existing_states.get(
                pid,
                {"established": False, "stability": 0.0, "observations": 0}
            )

            # Extract piece subspace
            sub_curr = curr_emb[piece.dim_start : piece.dim_end]
            sub_ref = ref_emb[piece.dim_start : piece.dim_end]

            # Cosine similarity within this latent subspace
            norm_c = np.linalg.norm(sub_curr) + 1e-9
            norm_r = np.linalg.norm(sub_ref) + 1e-9
            cos_sim = float(np.dot(sub_curr, sub_ref) / (norm_c * norm_r))
            # Clip to [0, 1] for stability scoring
            stability = float(np.clip(cos_sim, 0.0, 1.0))

            was_established = bool(prior_info.get("established", False))
            obs = int(prior_info.get("observations", 0))

            # Non-destructive: Once established, it remains established
            if was_established:
                results[pid] = {
                    "established": True,
                    "stability": round(stability, 4),
                    "observations": obs,
                    "just_unlocked": False,
                }
            else:
                # If stable in this recording, increment observations
                if stability >= self.cfg.piece_stability_threshold:
                    obs += 1

                is_now_established = obs >= self.cfg.min_piece_observations
                just_unlocked = is_now_established and not was_established

                results[pid] = {
                    "established": is_now_established,
                    "stability": round(stability, 4),
                    "observations": obs,
                    "just_unlocked": just_unlocked,
                }

        return results
