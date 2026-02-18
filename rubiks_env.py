"""
rubiks_env.py — Complete 3×3 Rubik's Cube Simulator for MARVELS
================================================================
Implements a Gym-style environment for the 3×3 Rubik's Cube.

State Representation:
  - 6 faces × 3×3 grid = 54 stickers
  - Faces indexed: 0=U(white), 1=D(yellow), 2=F(red), 3=B(orange), 4=L(green), 5=R(blue)
  - Each face stored as a 3×3 numpy array of ints [0..5]

Actions (18 total):
  - 6 faces × 3 rotations (CW=90°, HALF=180°, CCW=270°)
  - Encoded as: face_idx * 3 + rotation_idx
  - e.g.  0=U_CW,  1=U_180,  2=U_CCW,  3=D_CW, ...

The environment tracks cubie-level permutation and orientation
for corner and edge cubies, enabling the quaternion encoder
to construct a geometrically meaningful state vector.
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
import copy


# ──────────────────────────── Constants ────────────────────────────
# Face indices
U, D, F, B, L, R = 0, 1, 2, 3, 4, 5

FACE_NAMES = ['U', 'D', 'F', 'B', 'L', 'R']
ROTATION_NAMES = ['CW', '180', 'CCW']

# Action name lookup
ACTION_NAMES = [
    f"{FACE_NAMES[f]}_{ROTATION_NAMES[r]}"
    for f in range(6)
    for r in range(3)
]

NUM_ACTIONS = 18

# Colors for terminal display
FACE_COLORS = {
    0: 'W',   # White  (Up)
    1: 'Y',   # Yellow (Down)
    2: 'R',   # Red    (Front)
    3: 'O',   # Orange (Back)
    4: 'G',   # Green  (Left)
    5: 'B',   # Blue   (Right)
}

# ─────────────── Corner & Edge Cubie Definitions ───────────────
# Each corner is defined by (face, row, col) triples for its 3 stickers,
# ordered so that the first sticker is the U/D face when in home position.

CORNER_POSITIONS = [
    # Corner 0: ULB
    [(U, 0, 0), (L, 0, 0), (B, 0, 2)],
    # Corner 1: UBR
    [(U, 0, 2), (B, 0, 0), (R, 0, 2)],
    # Corner 2: URF
    [(U, 2, 2), (R, 0, 0), (F, 0, 2)],
    # Corner 3: UFL
    [(U, 2, 0), (F, 0, 0), (L, 0, 2)],
    # Corner 4: DLF
    [(D, 0, 0), (L, 2, 2), (F, 2, 0)],
    # Corner 5: DFR
    [(D, 0, 2), (F, 2, 2), (R, 2, 0)],
    # Corner 6: DRB
    [(D, 2, 2), (R, 2, 2), (B, 2, 0)],
    # Corner 7: DBL
    [(D, 2, 0), (B, 2, 2), (L, 2, 0)],
]

EDGE_POSITIONS = [
    # Edge 0: UB
    [(U, 0, 1), (B, 0, 1)],
    # Edge 1: UR
    [(U, 1, 2), (R, 0, 1)],
    # Edge 2: UF
    [(U, 2, 1), (F, 0, 1)],
    # Edge 3: UL
    [(U, 1, 0), (L, 0, 1)],
    # Edge 4: FL
    [(F, 1, 0), (L, 1, 2)],
    # Edge 5: FR
    [(F, 1, 2), (R, 1, 0)],
    # Edge 6: BL
    [(B, 1, 2), (L, 1, 0)],
    # Edge 7: BR
    [(B, 1, 0), (R, 1, 2)],
    # Edge 8: DF
    [(D, 0, 1), (F, 2, 1)],
    # Edge 9: DR
    [(D, 1, 2), (R, 2, 1)],
    # Edge 10: DB
    [(D, 2, 1), (B, 2, 1)],
    # Edge 11: DL
    [(D, 1, 0), (L, 2, 1)],
]


class RubiksCube:
    """
    Full 3×3 Rubik's Cube simulator.

    The internal state uses 6 numpy arrays (one per face, 3×3).
    Moves are performed by index-based permutation of stickers.

    Additionally tracks cubie permutation and orientation arrays
    for the 8 corners and 12 edges, which the QuaternionEncoder
    uses to build a geometrically-grounded state representation.
    """

    def __init__(self):
        self.faces: np.ndarray = np.zeros((6, 3, 3), dtype=np.int8)
        # Corner tracking: corner_perm[i] = which corner is in slot i
        #                   corner_orient[i] = orientation (0,1,2) of that corner
        self.corner_perm = np.zeros(8, dtype=np.int8)
        self.corner_orient = np.zeros(8, dtype=np.int8)
        # Edge tracking: edge_perm[i] = which edge is in slot i
        #                edge_orient[i] = orientation (0 or 1) of that edge
        self.edge_perm = np.zeros(12, dtype=np.int8)
        self.edge_orient = np.zeros(12, dtype=np.int8)

        self.move_count = 0
        self.max_moves = 200  # Episode ends after this many moves

        self.reset()

    # ─────────────────────── Core API ───────────────────────

    def reset(self, scramble_depth: int = 0) -> np.ndarray:
        """Reset to solved state, optionally scramble."""
        # Each face filled with its own index
        for i in range(6):
            self.faces[i] = i

        # Identity permutations
        self.corner_perm = np.arange(8, dtype=np.int8)
        self.corner_orient = np.zeros(8, dtype=np.int8)
        self.edge_perm = np.arange(12, dtype=np.int8)
        self.edge_orient = np.zeros(12, dtype=np.int8)

        self.move_count = 0

        if scramble_depth > 0:
            self.scramble(scramble_depth)

        return self.get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute an action and return (state, reward, done, info).

        Reward:
          +100  if cube is solved
          -0.01 per move (encourages efficiency)
        """
        assert 0 <= action < NUM_ACTIONS, f"Invalid action: {action}"

        self._apply_action(action)
        self.move_count += 1

        solved = self.is_solved()
        reward = -0.01  # Move penalty
        if solved:
            reward += 100.0

        done = solved or (self.move_count >= self.max_moves)

        info = {
            'solved': solved,
            'move_count': self.move_count,
            'action_name': ACTION_NAMES[action],
        }

        return self.get_state(), reward, done, info

    def is_solved(self) -> bool:
        """Check if all faces are uniformly colored."""
        for i in range(6):
            if not np.all(self.faces[i] == self.faces[i, 0, 0]):
                return False
        return True

    def get_state(self) -> dict:
        """
        Return complete state for the encoder.
        Includes raw faces and cubie-level tracking.
        """
        return {
            'faces': self.faces.copy(),
            'corner_perm': self.corner_perm.copy(),
            'corner_orient': self.corner_orient.copy(),
            'edge_perm': self.edge_perm.copy(),
            'edge_orient': self.edge_orient.copy(),
        }

    def get_flat_state(self) -> np.ndarray:
        """Return flattened 54-element sticker array."""
        return self.faces.flatten().astype(np.float32)

    def scramble(self, depth: int, seed: Optional[int] = None) -> List[int]:
        """Apply `depth` random moves. Returns the list of actions applied."""
        rng = np.random.RandomState(seed)
        actions = []
        prev_face = -1
        for _ in range(depth):
            # Avoid immediately undoing the previous move
            while True:
                action = rng.randint(NUM_ACTIONS)
                face = action // 3
                if face != prev_face:
                    break
            self._apply_action(action)
            actions.append(action)
            prev_face = face
        return actions

    def clone(self) -> 'RubiksCube':
        """Deep copy of the current cube state."""
        c = RubiksCube.__new__(RubiksCube)
        c.faces = self.faces.copy()
        c.corner_perm = self.corner_perm.copy()
        c.corner_orient = self.corner_orient.copy()
        c.edge_perm = self.edge_perm.copy()
        c.edge_orient = self.edge_orient.copy()
        c.move_count = self.move_count
        c.max_moves = self.max_moves
        return c

    # ──────────────────── Move Execution ────────────────────

    def _apply_action(self, action: int):
        """Decode action into face + rotation and apply."""
        face = action // 3
        rotation = action % 3  # 0=CW, 1=180, 2=CCW

        if rotation == 0:
            self._rotate_face_cw(face)
        elif rotation == 1:
            self._rotate_face_cw(face)
            self._rotate_face_cw(face)
        else:  # CCW = 3 × CW
            self._rotate_face_cw(face)
            self._rotate_face_cw(face)
            self._rotate_face_cw(face)

    def _rotate_face_cw(self, face: int):
        """
        Rotate a face 90° clockwise.
        1. Rotate the 3×3 face grid itself
        2. Cycle the adjacent stickers from neighboring faces
        3. Update cubie permutation and orientation tables
        """
        # 1. Rotate the face grid
        self.faces[face] = np.rot90(self.faces[face], k=-1)

        # 2. Cycle adjacent stickers
        self._cycle_adjacent(face)

        # 3. Update cubie tracking
        self._update_cubies_for_face_cw(face)

    def _cycle_adjacent(self, face: int):
        """Cycle the 12 adjacent stickers for a CW rotation of `face`."""
        # Each face rotation cycles 4 strips of 3 stickers from neighboring faces.
        # We define these strips explicitly for each face.

        if face == U:
            # U CW: F top → R top → B top → L top → F top  (but reversed for B)
            tmp = self.faces[F, 0, :].copy()
            self.faces[F, 0, :] = self.faces[R, 0, :]
            self.faces[R, 0, :] = self.faces[B, 0, :]
            self.faces[B, 0, :] = self.faces[L, 0, :]
            self.faces[L, 0, :] = tmp

        elif face == D:
            # D CW: F bottom → L bottom → B bottom → R bottom → F bottom
            tmp = self.faces[F, 2, :].copy()
            self.faces[F, 2, :] = self.faces[L, 2, :]
            self.faces[L, 2, :] = self.faces[B, 2, :]
            self.faces[B, 2, :] = self.faces[R, 2, :]
            self.faces[R, 2, :] = tmp

        elif face == F:
            # F CW: U bottom → R left col → D top (rev) → L right col (rev) → U bottom
            tmp = self.faces[U, 2, :].copy()
            self.faces[U, 2, :] = self.faces[L, :, 2][::-1]
            self.faces[L, :, 2] = self.faces[D, 0, :]
            self.faces[D, 0, :] = self.faces[R, :, 0][::-1]
            self.faces[R, :, 0] = tmp

        elif face == B:
            # B CW: U top → L left col (rev) → D bottom (rev) → R right col → U top
            tmp = self.faces[U, 0, :].copy()
            self.faces[U, 0, :] = self.faces[R, :, 2]
            self.faces[R, :, 2] = self.faces[D, 2, :][::-1]
            self.faces[D, 2, :] = self.faces[L, :, 0]
            self.faces[L, :, 0] = tmp[::-1]

        elif face == L:
            # L CW: U left col → F left col → D left col → B right col (rev) → U left col
            tmp = self.faces[U, :, 0].copy()
            self.faces[U, :, 0] = self.faces[B, :, 2][::-1]
            self.faces[B, :, 2] = self.faces[D, :, 0][::-1]
            self.faces[D, :, 0] = self.faces[F, :, 0]
            self.faces[F, :, 0] = tmp

        elif face == R:
            # R CW: U right col → B left col (rev) → D right col → F right col → U right col
            tmp = self.faces[U, :, 2].copy()
            self.faces[U, :, 2] = self.faces[F, :, 2]
            self.faces[F, :, 2] = self.faces[D, :, 2]
            self.faces[D, :, 2] = self.faces[B, :, 0][::-1]
            self.faces[B, :, 0] = tmp[::-1]

    def _update_cubies_for_face_cw(self, face: int):
        """
        Update corner and edge permutation/orientation arrays
        for a clockwise rotation of `face`.

        Corner orientation change: when a corner moves to a new slot
        via a face rotation, its orientation is adjusted depending on
        whether the rotation is around U/D (no twist) or F/B/L/R (twist).

        Edge orientation change: flipped when moved by F or B.
        """
        # Define which corner/edge slots are cycled by each face
        corner_cycles = {
            U: [0, 1, 2, 3],   # ULB → UBR → URF → UFL
            D: [4, 5, 6, 7],   # DLF → DFR → DRB → DBL
            F: [3, 2, 5, 4],   # UFL → URF → DFR → DLF
            B: [1, 0, 7, 6],   # UBR → ULB → DBL → DRB
            L: [0, 3, 4, 7],   # ULB → UFL → DLF → DBL
            R: [2, 1, 6, 5],   # URF → UBR → DRB → DFR
        }

        edge_cycles = {
            U: [0, 1, 2, 3],   # UB → UR → UF → UL
            D: [8, 9, 10, 11], # DF → DR → DB → DL
            F: [2, 5, 8, 4],   # UF → FR → DF → FL
            B: [0, 6, 10, 7],  # UB → BL → DB → BR
            L: [3, 4, 11, 6],  # UL → FL → DL → BL
            R: [1, 7, 9, 5],   # UR → BR → DR → FR
        }

        # Corner orientation deltas for non-U/D faces
        # When cycling corners around F/B/L/R, corners twist
        corner_orient_delta = {
            U: [0, 0, 0, 0],
            D: [0, 0, 0, 0],
            F: [1, 2, 1, 2],
            B: [1, 2, 1, 2],
            L: [1, 2, 1, 2],
            R: [1, 2, 1, 2],
        }

        # Edge orientation deltas
        # Edges flip (toggle orientation) when moved by F or B
        edge_orient_delta = {
            U: [0, 0, 0, 0],
            D: [0, 0, 0, 0],
            F: [1, 1, 1, 1],
            B: [1, 1, 1, 1],
            L: [0, 0, 0, 0],
            R: [0, 0, 0, 0],
        }

        # --- Apply corner cycle: [a, b, c, d] → d takes a's place, etc. ---
        cc = corner_cycles[face]
        cd = corner_orient_delta[face]

        # Save the last element (it will be overwritten first)
        saved_perm = self.corner_perm[cc[3]]
        saved_orient = self.corner_orient[cc[3]]

        for i in range(3, 0, -1):
            self.corner_perm[cc[i]] = self.corner_perm[cc[i - 1]]
            self.corner_orient[cc[i]] = (self.corner_orient[cc[i - 1]] + cd[i]) % 3

        self.corner_perm[cc[0]] = saved_perm
        self.corner_orient[cc[0]] = (saved_orient + cd[0]) % 3

        # --- Apply edge cycle ---
        ec = edge_cycles[face]
        ed = edge_orient_delta[face]

        saved_perm = self.edge_perm[ec[3]]
        saved_orient = self.edge_orient[ec[3]]

        for i in range(3, 0, -1):
            self.edge_perm[ec[i]] = self.edge_perm[ec[i - 1]]
            self.edge_orient[ec[i]] = (self.edge_orient[ec[i - 1]] + ed[i]) % 2

        self.edge_perm[ec[0]] = saved_perm
        self.edge_orient[ec[0]] = (saved_orient + ed[0]) % 2

    # ──────────────────── State Queries ────────────────────

    def get_corner_colors(self) -> List[List[int]]:
        """
        Return the sticker colors for each corner slot.
        corners[i] = [color_0, color_1, color_2] for the 3 stickers of corner slot i.
        """
        corners = []
        for positions in CORNER_POSITIONS:
            colors = [int(self.faces[f, r, c]) for f, r, c in positions]
            corners.append(colors)
        return corners

    def get_edge_colors(self) -> List[List[int]]:
        """Return sticker colors for each edge slot."""
        edges = []
        for positions in EDGE_POSITIONS:
            colors = [int(self.faces[f, r, c]) for f, r, c in positions]
            edges.append(colors)
        return edges

    def count_solved_corners(self) -> int:
        """Count how many corners are in correct position and orientation."""
        count = 0
        for i in range(8):
            if self.corner_perm[i] == i and self.corner_orient[i] == 0:
                count += 1
        return count

    def count_solved_edges(self) -> int:
        """Count how many edges are in correct position and orientation."""
        count = 0
        for i in range(12):
            if self.edge_perm[i] == i and self.edge_orient[i] == 0:
                count += 1
        return count

    def count_solved_centers(self) -> int:
        """Centers are always solved on a standard 3×3 (fixed centers), return 6."""
        # On a standard Rubik's Cube, centers don't move.
        # We report 6 to simplify the reward system.
        # For the agent, center "correctness" is whether surrounding
        # stickers match the center color.
        count = 0
        for face_idx in range(6):
            center_color = self.faces[face_idx, 1, 1]
            # Count how many of the 8 surrounding stickers match center
            face = self.faces[face_idx]
            matches = 0
            for r in range(3):
                for c in range(3):
                    if (r, c) != (1, 1) and face[r, c] == center_color:
                        matches += 1
            if matches == 8:
                count += 1
        return count

    # ──────────────────── Display ────────────────────

    def render(self) -> str:
        """Render the cube as a text-based net."""
        def face_row(face_idx, row):
            return ' '.join(FACE_COLORS[int(c)] for c in self.faces[face_idx, row])

        lines = []
        # Top face (U)
        for r in range(3):
            lines.append('         ' + face_row(U, r))
        lines.append('')

        # Middle band: L F R B
        for r in range(3):
            line = (face_row(L, r) + '  ' +
                    face_row(F, r) + '  ' +
                    face_row(R, r) + '  ' +
                    face_row(B, r))
            lines.append(line)
        lines.append('')

        # Bottom face (D)
        for r in range(3):
            lines.append('         ' + face_row(D, r))

        return '\n'.join(lines)

    def __repr__(self):
        status = "SOLVED" if self.is_solved() else f"moves={self.move_count}"
        return f"RubiksCube({status})"


# ──────────────── Vectorized Environment ────────────────

class VectorizedRubiksCube:
    """
    Manages N independent Rubik's Cube environments for parallel rollouts.
    """

    def __init__(self, num_envs: int, max_moves: int = 200):
        self.num_envs = num_envs
        self.envs = [RubiksCube() for _ in range(num_envs)]
        for env in self.envs:
            env.max_moves = max_moves

    def reset_all(self, scramble_depth: int) -> List[dict]:
        """Reset all environments with given scramble depth."""
        states = []
        for env in self.envs:
            state = env.reset(scramble_depth)
            states.append(state)
        return states

    def reset_one(self, idx: int, scramble_depth: int) -> dict:
        """Reset a single environment."""
        return self.envs[idx].reset(scramble_depth)

    def step(self, actions: List[int]) -> Tuple[List[dict], List[float], List[bool], List[dict]]:
        """Step all environments with given actions."""
        states, rewards, dones, infos = [], [], [], []
        for env, action in zip(self.envs, actions):
            s, r, d, info = env.step(action)
            states.append(s)
            rewards.append(r)
            dones.append(d)
            infos.append(info)
        return states, rewards, dones, infos

    def get_states(self) -> List[dict]:
        """Get current states of all environments."""
        return [env.get_state() for env in self.envs]
