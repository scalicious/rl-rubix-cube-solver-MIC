"""
rubiks_env.py - 3x3 Rubik's Cube simulator with gym-style interface.

State: 6 faces x 3x3 grid = 54 stickers
Actions: 6 faces x 3 rotations (CW, 180, CCW) = 18 total
Also tracks cubie permutation/orientation for the quaternion encoder.
"""

import numpy as np
from typing import Tuple, List, Optional
import copy

# Face indices
U, D, F, B, L, R = 0, 1, 2, 3, 4, 5

FACE_NAMES = ['U', 'D', 'F', 'B', 'L', 'R']
ROTATION_NAMES = ['CW', '180', 'CCW']

ACTION_NAMES = [
    f"{FACE_NAMES[f]}_{ROTATION_NAMES[r]}"
    for f in range(6)
    for r in range(3)
]

NUM_ACTIONS = 18

FACE_COLORS = {
    0: 'W',  # White  (Up)
    1: 'Y',  # Yellow (Down)
    2: 'R',  # Red    (Front)
    3: 'O',  # Orange (Back)
    4: 'G',  # Green  (Left)
    5: 'B',  # Blue   (Right)
}

# Corner definitions: (face, row, col) triples for 3 stickers each.
# First sticker is the U/D face when in home position.
CORNER_POSITIONS = [
    [(U, 0, 0), (L, 0, 0), (B, 0, 2)],  # ULB
    [(U, 0, 2), (B, 0, 0), (R, 0, 2)],  # UBR
    [(U, 2, 2), (R, 0, 0), (F, 0, 2)],  # URF
    [(U, 2, 0), (F, 0, 0), (L, 0, 2)],  # UFL
    [(D, 0, 0), (L, 2, 2), (F, 2, 0)],  # DLF
    [(D, 0, 2), (F, 2, 2), (R, 2, 0)],  # DFR
    [(D, 2, 2), (R, 2, 2), (B, 2, 0)],  # DRB
    [(D, 2, 0), (B, 2, 2), (L, 2, 0)],  # DBL
]

EDGE_POSITIONS = [
    [(U, 0, 1), (B, 0, 1)],  # UB
    [(U, 1, 2), (R, 0, 1)],  # UR
    [(U, 2, 1), (F, 0, 1)],  # UF
    [(U, 1, 0), (L, 0, 1)],  # UL
    [(F, 1, 0), (L, 1, 2)],  # FL
    [(F, 1, 2), (R, 1, 0)],  # FR
    [(B, 1, 2), (L, 1, 0)],  # BL
    [(B, 1, 0), (R, 1, 2)],  # BR
    [(D, 0, 1), (F, 2, 1)],  # DF
    [(D, 1, 2), (R, 2, 1)],  # DR
    [(D, 2, 1), (B, 2, 1)],  # DB
    [(D, 1, 0), (L, 2, 1)],  # DL
]


class RubiksCube:
    """
    Full 3x3 cube simulator. Uses 6 numpy arrays (3x3) for sticker state,
    plus permutation/orientation arrays for corners and edges.
    """

    def __init__(self):
        self.faces = np.zeros((6, 3, 3), dtype=np.int8)
        self.corner_perm = np.zeros(8, dtype=np.int8)
        self.corner_orient = np.zeros(8, dtype=np.int8)
        self.edge_perm = np.zeros(12, dtype=np.int8)
        self.edge_orient = np.zeros(12, dtype=np.int8)

        self.move_count = 0
        self.max_moves = 200

        self.reset()

    def reset(self, scramble_depth=0):
        """Reset to solved state, optionally scramble."""
        for i in range(6):
            self.faces[i] = i

        self.corner_perm = np.arange(8, dtype=np.int8)
        self.corner_orient = np.zeros(8, dtype=np.int8)
        self.edge_perm = np.arange(12, dtype=np.int8)
        self.edge_orient = np.zeros(12, dtype=np.int8)

        self.move_count = 0

        if scramble_depth > 0:
            self.scramble(scramble_depth)

        return self.get_state()

    def step(self, action):
        """Execute action, return (state, reward, done, info)."""
        assert 0 <= action < NUM_ACTIONS, f"Invalid action: {action}"

        self._apply_action(action)
        self.move_count += 1

        solved = self.is_solved()
        reward = -0.01  # small penalty per move
        if solved:
            reward += 100.0

        done = solved or (self.move_count >= self.max_moves)

        info = {
            'solved': solved,
            'move_count': self.move_count,
            'action_name': ACTION_NAMES[action],
        }

        return self.get_state(), reward, done, info

    def is_solved(self):
        """Check if every face is a single color."""
        for i in range(6):
            if not np.all(self.faces[i] == self.faces[i, 0, 0]):
                return False
        return True

    def get_state(self):
        """Full state dict for the encoder (faces + cubie tracking)."""
        return {
            'faces': self.faces.copy(),
            'corner_perm': self.corner_perm.copy(),
            'corner_orient': self.corner_orient.copy(),
            'edge_perm': self.edge_perm.copy(),
            'edge_orient': self.edge_orient.copy(),
        }

    def get_flat_state(self):
        """Flattened 54-element sticker array."""
        return self.faces.flatten().astype(np.float32)

    def scramble(self, depth, seed=None):
        """Apply random moves, avoiding immediate undos. Returns move list."""
        rng = np.random.RandomState(seed)
        actions = []
        prev_face = -1
        for _ in range(depth):
            while True:
                action = rng.randint(NUM_ACTIONS)
                face = action // 3
                if face != prev_face:
                    break
            self._apply_action(action)
            actions.append(action)
            prev_face = face
        return actions

    def clone(self):
        """Deep copy of current state."""
        c = RubiksCube.__new__(RubiksCube)
        c.faces = self.faces.copy()
        c.corner_perm = self.corner_perm.copy()
        c.corner_orient = self.corner_orient.copy()
        c.edge_perm = self.edge_perm.copy()
        c.edge_orient = self.edge_orient.copy()
        c.move_count = self.move_count
        c.max_moves = self.max_moves
        return c

    # --- Move execution ---

    def _apply_action(self, action):
        face = action // 3
        rotation = action % 3  # 0=CW, 1=180, 2=CCW

        if rotation == 0:
            self._rotate_face_cw(face)
        elif rotation == 1:
            self._rotate_face_cw(face)
            self._rotate_face_cw(face)
        else:  # CCW = 3x CW
            self._rotate_face_cw(face)
            self._rotate_face_cw(face)
            self._rotate_face_cw(face)

    def _rotate_face_cw(self, face):
        """90 deg CW rotation: rotate grid, cycle adjacent stickers, update cubies."""
        self.faces[face] = np.rot90(self.faces[face], k=-1)
        self._cycle_adjacent(face)
        self._update_cubies_for_face_cw(face)

    def _cycle_adjacent(self, face):
        """Cycle the 12 adjacent stickers for a CW rotation."""
        if face == U:
            tmp = self.faces[F, 0, :].copy()
            self.faces[F, 0, :] = self.faces[R, 0, :]
            self.faces[R, 0, :] = self.faces[B, 0, :]
            self.faces[B, 0, :] = self.faces[L, 0, :]
            self.faces[L, 0, :] = tmp

        elif face == D:
            tmp = self.faces[F, 2, :].copy()
            self.faces[F, 2, :] = self.faces[L, 2, :]
            self.faces[L, 2, :] = self.faces[B, 2, :]
            self.faces[B, 2, :] = self.faces[R, 2, :]
            self.faces[R, 2, :] = tmp

        elif face == F:
            tmp = self.faces[U, 2, :].copy()
            self.faces[U, 2, :] = self.faces[L, :, 2][::-1]
            self.faces[L, :, 2] = self.faces[D, 0, :]
            self.faces[D, 0, :] = self.faces[R, :, 0][::-1]
            self.faces[R, :, 0] = tmp

        elif face == B:
            tmp = self.faces[U, 0, :].copy()
            self.faces[U, 0, :] = self.faces[R, :, 2]
            self.faces[R, :, 2] = self.faces[D, 2, :][::-1]
            self.faces[D, 2, :] = self.faces[L, :, 0]
            self.faces[L, :, 0] = tmp[::-1]

        elif face == L:
            tmp = self.faces[U, :, 0].copy()
            self.faces[U, :, 0] = self.faces[B, :, 2][::-1]
            self.faces[B, :, 2] = self.faces[D, :, 0][::-1]
            self.faces[D, :, 0] = self.faces[F, :, 0]
            self.faces[F, :, 0] = tmp

        elif face == R:
            tmp = self.faces[U, :, 2].copy()
            self.faces[U, :, 2] = self.faces[F, :, 2]
            self.faces[F, :, 2] = self.faces[D, :, 2]
            self.faces[D, :, 2] = self.faces[B, :, 0][::-1]
            self.faces[B, :, 0] = tmp[::-1]

    def _update_cubies_for_face_cw(self, face):
        """Update cubie permutation/orientation for a CW face rotation."""
        corner_cycles = {
            U: [0, 1, 2, 3],
            D: [4, 5, 6, 7],
            F: [3, 2, 5, 4],
            B: [1, 0, 7, 6],
            L: [0, 3, 4, 7],
            R: [2, 1, 6, 5],
        }

        edge_cycles = {
            U: [0, 1, 2, 3],
            D: [8, 9, 10, 11],
            F: [2, 5, 8, 4],
            B: [0, 6, 10, 7],
            L: [3, 4, 11, 6],
            R: [1, 7, 9, 5],
        }

        # corners twist when rotated around F/B/L/R but not U/D
        corner_orient_delta = {
            U: [0, 0, 0, 0],
            D: [0, 0, 0, 0],
            F: [1, 2, 1, 2],
            B: [1, 2, 1, 2],
            L: [1, 2, 1, 2],
            R: [1, 2, 1, 2],
        }

        # edges flip when moved by F or B
        edge_orient_delta = {
            U: [0, 0, 0, 0],
            D: [0, 0, 0, 0],
            F: [1, 1, 1, 1],
            B: [1, 1, 1, 1],
            L: [0, 0, 0, 0],
            R: [0, 0, 0, 0],
        }

        # apply corner 4-cycle
        cc = corner_cycles[face]
        cd = corner_orient_delta[face]

        saved_perm = self.corner_perm[cc[3]]
        saved_orient = self.corner_orient[cc[3]]

        for i in range(3, 0, -1):
            self.corner_perm[cc[i]] = self.corner_perm[cc[i - 1]]
            self.corner_orient[cc[i]] = (self.corner_orient[cc[i - 1]] + cd[i]) % 3

        self.corner_perm[cc[0]] = saved_perm
        self.corner_orient[cc[0]] = (saved_orient + cd[0]) % 3

        # apply edge 4-cycle
        ec = edge_cycles[face]
        ed = edge_orient_delta[face]

        saved_perm = self.edge_perm[ec[3]]
        saved_orient = self.edge_orient[ec[3]]

        for i in range(3, 0, -1):
            self.edge_perm[ec[i]] = self.edge_perm[ec[i - 1]]
            self.edge_orient[ec[i]] = (self.edge_orient[ec[i - 1]] + ed[i]) % 2

        self.edge_perm[ec[0]] = saved_perm
        self.edge_orient[ec[0]] = (saved_orient + ed[0]) % 2

    # --- State queries ---

    def get_corner_colors(self):
        """Sticker colors for each corner slot: list of [c0, c1, c2]."""
        corners = []
        for positions in CORNER_POSITIONS:
            colors = [int(self.faces[f, r, c]) for f, r, c in positions]
            corners.append(colors)
        return corners

    def get_edge_colors(self):
        """Sticker colors for each edge slot."""
        edges = []
        for positions in EDGE_POSITIONS:
            colors = [int(self.faces[f, r, c]) for f, r, c in positions]
            edges.append(colors)
        return edges

    def count_solved_corners(self):
        count = 0
        for i in range(8):
            if self.corner_perm[i] == i and self.corner_orient[i] == 0:
                count += 1
        return count

    def count_solved_edges(self):
        count = 0
        for i in range(12):
            if self.edge_perm[i] == i and self.edge_orient[i] == 0:
                count += 1
        return count

    def count_solved_centers(self):
        """Check how many faces have all surrounding stickers matching center color."""
        # centers don't actually move on a 3x3, so this is really checking
        # whether the whole face is solved
        count = 0
        for face_idx in range(6):
            center_color = self.faces[face_idx, 1, 1]
            face = self.faces[face_idx]
            matches = 0
            for r in range(3):
                for c in range(3):
                    if (r, c) != (1, 1) and face[r, c] == center_color:
                        matches += 1
            if matches == 8:
                count += 1
        return count

    # --- Display ---

    def render(self):
        """Text-based cube net."""
        def face_row(face_idx, row):
            return ' '.join(FACE_COLORS[int(c)] for c in self.faces[face_idx, row])

        lines = []
        # U face
        for r in range(3):
            lines.append('         ' + face_row(U, r))
        lines.append('')

        # L F R B band
        for r in range(3):
            line = (face_row(L, r) + '  ' +
                    face_row(F, r) + '  ' +
                    face_row(R, r) + '  ' +
                    face_row(B, r))
            lines.append(line)
        lines.append('')

        # D face
        for r in range(3):
            lines.append('         ' + face_row(D, r))

        return '\n'.join(lines)

    def __repr__(self):
        status = "SOLVED" if self.is_solved() else f"moves={self.move_count}"
        return f"RubiksCube({status})"


class VectorizedRubiksCube:
    """N independent cube environments for parallel rollouts."""

    def __init__(self, num_envs, max_moves=200):
        self.num_envs = num_envs
        self.envs = [RubiksCube() for _ in range(num_envs)]
        for env in self.envs:
            env.max_moves = max_moves

    def reset_all(self, scramble_depth):
        return [env.reset(scramble_depth) for env in self.envs]

    def reset_one(self, idx, scramble_depth):
        return self.envs[idx].reset(scramble_depth)

    def step(self, actions):
        states, rewards, dones, infos = [], [], [], []
        for env, action in zip(self.envs, actions):
            s, r, d, info = env.step(action)
            states.append(s)
            rewards.append(r)
            dones.append(d)
            infos.append(info)
        return states, rewards, dones, infos

    def get_states(self):
        return [env.get_state() for env in self.envs]
