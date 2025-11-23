import numpy as np

# Example: 100 patterns of size 8x8, built however you like
# Fill this with your * / _ motifs as 0/1
patterns = np.zeros((100, 8, 8), dtype=int)

## CENTER BLOCK
p = np.zeros((8, 8), dtype=int)
p[3:5, 3:5] = 1  # center block
patterns[0] = p

p = np.zeros((8, 8), dtype=int)
p[2:6, 2:6] = 1  # larger center block
patterns[1] = p 

## INNER HOLLOW SQUARE
p = np.zeros((8, 8), dtype=int)
p[2, 2:6] = 1  # top inner edge
p[5, 2:6] = 1  # bottom inner edge
p[2:6, 2] = 1  # left inner edge
p[2:6, 5] = 1  # right inner edge
patterns[2] = p

## BORDER & LINES
p = np.zeros((8, 8), dtype=int)
p[[0, 7], :] = 1 #set top and bottom rows
p[:, [0, 7]] = 1 #set second and last columns
patterns[3] = p

p = np.zeros((8, 8), dtype=int)
for i in range(8):
    p[i, i//2] = 1  # vertical lines
patterns[4] = p

p = np.zeros((8, 8), dtype=int)
p[:, 3] = 1  # vertical line in column 3
patterns[5] = p

p = np.zeros((8, 8), dtype=int)
p[3, :] = 1  # horizontal line in row 3
patterns[6] = p    

p = np.zeros((8, 8), dtype=int)
p[3:5, :] = 1 #set top and bottom rows
patterns[7] = p

p = np.zeros((8, 8), dtype=int)
p[:, 3:5] = 1 #set second and last columns
patterns[8] = p    

## DIAGONAL LINE
p = np.zeros((8, 8), dtype=int)
np.fill_diagonal(p, 1)  # diagonal line
patterns[9] = p 

p = np.zeros((8, 8), dtype=int)
np.fill_diagonal(np.fliplr(p), 1)  # anti-diagonal line
patterns[10] = p

p = np.zeros((8, 8), dtype=int)
for i in range(8):
    p[i, i] = 1  # diagonal line
    p[i, 7 - i] = 1  # anti-diagonal line
patterns[11] = p

p = np.zeros((8, 8), dtype=int)
p[0, :] = 1  # top row
p[7, :] = 1  # bottom row
p[:, 0] = 1  # left column
p[:, 7] = 1  # right column
for i in range(8):
    p[i, i] = 1  # diagonal line
patterns[12] = p

## PLUS / CROSS SHAPE
p = np.zeros((8, 8), dtype=int)
p[4, :] = 1      # horizontal through center-ish
p[:, 4] = 1      # vertical through center-ish
patterns[13] = p

## OFFSET PLUS SHAPE
p = np.zeros((8, 8), dtype=int)
p[2, :] = 1      # horizontal near top
p[:, 5] = 1      # vertical slightly right
patterns[14] = p

## DOUBLE BORDER (OUTER + INNER RING)
p = np.zeros((8, 8), dtype=int)
# outer border
p[0, :] = 1
p[7, :] = 1
p[:, 0] = 1
p[:, 7] = 1
# inner border offset by 1
p[1, 1:7] = 1
p[6, 1:7] = 1
p[1:7, 1] = 1
p[1:7, 6] = 1
patterns[15] = p

## FULL CHECKERBOARD
p = np.zeros((8, 8), dtype=int)
for i in range(8):
    for j in range(8):
        if (i + j) % 2 == 0:
            p[i, j] = 1
patterns[16] = p

## SPARSE BORDER CHECKERBOARD
p = np.zeros((8, 8), dtype=int)
for j in range(8):
    if j % 2 == 0:
        p[0, j] = 1      # top
        p[7, j] = 1      # bottom
for i in range(8):
    if i % 2 == 0:
        p[i, 0] = 1      # left
        p[i, 7] = 1      # right
patterns[17] = p

## TWO PARALLEL MAIN DIAGONALS
p = np.zeros((8, 8), dtype=int)
for i in range(8):
    if 0 <= i < 8:
        p[i, i] = 1                  # main diagonal
    if 0 <= i+1 < 8:
        p[i, i+1] = 1                # one step to the right
patterns[18] = p

## DIAMOND SHAPE (MANHATTAN DISTANCE)
p = np.zeros((8, 8), dtype=int)
center = (3, 3)  # shift diamond slightly to upper-left
radius = 3
for i in range(8):
    for j in range(8):
        if abs(i - center[0]) + abs(j - center[1]) == radius:
            p[i, j] = 1
patterns[19] = p

## FILLED DIAMOND
p = np.zeros((8, 8), dtype=int)
center = (3, 3)
radius = 3
for i in range(8):
    for j in range(8):
        if abs(i - center[0]) + abs(j - center[1]) <= radius:
            p[i, j] = 1
patterns[20] = p

## TOP-LEFT QUADRANT FILLED
p = np.zeros((8, 8), dtype=int)
p[0:4, 0:4] = 1
patterns[21] = p

## OPPOSITE QUADRANTS
p = np.zeros((8, 8), dtype=int)
p[0:4, 0:4] = 1      # top-left
p[4:8, 4:8] = 1      # bottom-right
patterns[22] = p

## VERTICAL STRIPES
p = np.zeros((8, 8), dtype=int)
for j in range(0, 8, 2):
    p[:, j] = 1
patterns[23] = p

## HORIZONTAL STRIPES
p = np.zeros((8, 8), dtype=int)
for i in range(0, 8, 2):
    p[i, :] = 1
patterns[24] = p

## CROSS OF 2x2 BLOBS
p = np.zeros((8, 8), dtype=int)
# center-ish vertical blobs
p[2:4, 3:5] = 1
p[4:6, 3:5] = 1
# side blobs
p[3:5, 1:3] = 1
p[3:5, 5:7] = 1
patterns[25] = p


