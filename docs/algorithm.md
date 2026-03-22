# Algorithm

ForceAtlas2 is a force-directed graph layout algorithm from the [Jacomy et al. 2014 paper](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0098679), originally implemented in [Gephi](https://gephi.org/).

## Force Model

Each iteration applies three types of forces, then uses an adaptive speed system to update positions:

### Repulsion

Every pair of nodes repels each other. The force is proportional to the product of their masses and inversely proportional to their distance:

$$F_{repulsion} = k_r \cdot \frac{m_1 \cdot m_2}{d}$$

where $k_r$ is `scalingRatio`, $m_i$ is node mass (1 + degree), and $d$ is distance.

### Attraction

Connected nodes attract each other along edges:

| Mode | Formula | Use Case |
|------|---------|----------|
| Linear (default) | $F = -c \cdot w \cdot d$ | General layout |
| LinLog | $F = -c \cdot w \cdot \log(1 + d)$ | Community detection |

where $c$ is `outboundAttCompensation`, $w$ is edge weight, and $d$ is distance.

### Gravity

All nodes are attracted toward the origin to prevent disconnected components from drifting:

| Mode | Formula | Behavior |
|------|---------|----------|
| Standard | $F = m \cdot g / d$ | Weakens with distance |
| Strong | $F = c \cdot m \cdot g$ | Distance-independent |

## Anti-Collision (adjustSizes)

When `adjustSizes=True`, node sizes modify force computation:

- **Repulsion**: Distance is adjusted by subtracting node radii ($d_{adj} = d - s_1 - s_2$). When nodes overlap ($d_{adj} < 0$), a strong constant repulsive force is applied ($100 \times$ normal).
- **Attraction**: Zero force when nodes overlap, preventing them from being pulled through each other.

This matches the Gephi `ForceFactory.java` anti-collision implementation.

## Barnes-Hut Approximation

For large graphs, computing all-pairs repulsion is $O(n^2)$. The Barnes-Hut algorithm groups distant nodes into regions using a spatial tree ($2^{dim}$ partitioning), reducing complexity to $O(n \log n)$.

The `barnesHutTheta` parameter controls accuracy: lower values are more accurate but slower. The default (1.2) provides a good balance.

!!! note "Barnes-Hut and dimensions"
    The spatial tree generalizes to any number of dimensions. In 2D it's a quadtree (4 children), in 3D an octree (8 children), and in general a $2^{dim}$-tree.

## Adaptive Speed

ForceAtlas2 uses a swing/traction mechanism to dynamically adjust simulation speed:

- **Swing**: How much a node's force direction changes between iterations (irregular movement)
- **Traction**: How much a node moves consistently in the same direction (useful movement)

When swing is high relative to traction, speed decreases to prevent oscillation. When traction dominates, speed increases for faster convergence.

## Node Mass

Node mass is defined as $1 + degree$. Higher-mass nodes are harder to move and have stronger repulsion, which naturally pushes high-degree hubs toward the center.

!!! warning "Self-loops inflate mass"
    Self-loops (non-zero diagonal in the adjacency matrix) increase a node's
    degree — and therefore its mass — without creating visible edges. This can
    distort layouts. ForceAtlas2 emits a warning when self-loops are detected.

## References

- Jacomy M, Venturini T, Heymann S, Bastian M (2014). *ForceAtlas2, a Continuous Graph Layout Algorithm for Handy Network Visualization Designed for the Gephi Software.* PLoS ONE 9(6): e98679.
- [Gephi ForceAtlas2.java](https://github.com/gephi/gephi/blob/master/modules/LayoutPlugin/src/main/java/org/gephi/layout/plugin/forceAtlas2/ForceAtlas2.java)
- [Gephi ForceFactory.java](https://github.com/gephi/gephi/blob/master/modules/LayoutPlugin/src/main/java/org/gephi/layout/plugin/forceAtlas2/ForceFactory.java)
