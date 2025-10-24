"""
Enhanced Visualization Utilities

Provides geographic visualization capabilities for traffic networks
using OSMnx, geopandas, and other geospatial libraries.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from typing import Optional, Tuple, Dict
from matplotlib.patches import Patch
import warnings

# Geospatial imports
try:
    import osmnx as ox
    import geopandas as gpd
    from shapely.geometry import Point, LineString
    import contextily as ctx
    HAS_GEO = True
except ImportError:
    HAS_GEO = False
    warnings.warn("Geospatial libraries not available. Install osmnx, geopandas, contextily for enhanced maps.")


class NetworkVisualizer:
    """Enhanced visualization for traffic networks with geographic mapping."""
    
    def __init__(self, env):
        """
        Initialize visualizer with traffic environment.
        
        Parameters
        ----------
        env : BaseTrafficEnvironment
            Traffic environment instance
        """
        self.env = env
        self.has_coords = env.node_coords is not None
        
        if self.has_coords:
            # Check if coordinates are lat/lon (typical range)
            sample_x = env.node_coords['X'].iloc[0]
            sample_y = env.node_coords['Y'].iloc[0]
            
            # Lat/lon typically: lon in [-180, 180], lat in [-90, 90]
            self.is_latlon = (-180 <= sample_x <= 180) and (-90 <= sample_y <= 90)
        else:
            self.is_latlon = False
    
    def plot_network_on_map(self,
                           figsize: Tuple[int, int] = (16, 12),
                           node_size: int = 100,
                           edge_width: float = 2.0,
                           edge_alpha: float = 0.7,
                           show_background: bool = True,
                           background_source: str = 'OpenStreetMap',
                           zoom_level: int = 12,
                           show_node_labels: bool = False,
                           flow_data: Optional[np.ndarray] = None,
                           colormap: str = 'YlOrRd',
                           title: Optional[str] = None,
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot network on geographic map with background.
        
        Parameters
        ----------
        figsize : tuple
            Figure size
        node_size : int
            Size of nodes
        edge_width : float
            Width of edges
        edge_alpha : float
            Transparency of edges
        show_background : bool
            Whether to show map background (requires coordinates)
        background_source : str
            Map tile source ('OpenStreetMap', 'Stamen Terrain', etc.)
        zoom_level : int
            Map zoom level (higher = more detail)
        show_node_labels : bool
            Whether to show node IDs
        flow_data : np.ndarray, optional
            Link flows for edge coloring
        colormap : str
            Matplotlib colormap for flow visualization
        title : str, optional
            Plot title
        save_path : str, optional
            Path to save figure
            
        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        if not self.has_coords:
            print("Warning: No node coordinates available. Using spring layout.")
            return self._plot_without_coords(figsize, node_size, edge_width, title, save_path)
        
        if show_background and not HAS_GEO:
            print("Warning: Geospatial libraries not available. Plotting without background.")
            show_background = False
        
        # Create GeoDataFrame if lat/lon
        if self.is_latlon and HAS_GEO:
            return self._plot_with_geographic_background(
                figsize, node_size, edge_width, edge_alpha,
                show_background, background_source, zoom_level,
                show_node_labels, flow_data, colormap, title, save_path
            )
        else:
            return self._plot_with_coordinates(
                figsize, node_size, edge_width, edge_alpha,
                show_node_labels, flow_data, colormap, title, save_path
            )
    
    def _plot_with_geographic_background(self,
                                         figsize, node_size, edge_width, edge_alpha,
                                         show_background, background_source, zoom_level,
                                         show_node_labels, flow_data, colormap,
                                         title, save_path):
        """Plot network with geographic background using contextily."""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create GeoDataFrame for nodes
        geometry = [Point(xy) for xy in zip(
            self.env.node_coords['X'],
            self.env.node_coords['Y']
        )]
        
        nodes_gdf = gpd.GeoDataFrame(
            self.env.node_coords,
            geometry=geometry,
            crs='EPSG:4326'  # WGS84 lat/lon
        )
        
        # Convert to Web Mercator for basemap
        nodes_gdf = nodes_gdf.to_crs(epsg=3857)
        
        # Create GeoDataFrame for edges
        # Handle double links by creating offset parallel lines for bidirectional edges
        edge_geometries = []
        edge_data = []
        edge_pairs = {}  # Track bidirectional edges

        # First pass: identify bidirectional edges
        for idx, row in self.env.net_data.iterrows():
            init_node = str(int(row['init_node']))
            term_node = str(int(row['term_node']))
            edge_pair_key = tuple(sorted([init_node, term_node]))

            if edge_pair_key not in edge_pairs:
                edge_pairs[edge_pair_key] = []
            edge_pairs[edge_pair_key].append(idx)

        # Second pass: create geometries with offset for bidirectional edges
        for idx, row in self.env.net_data.iterrows():
            init_node = str(int(row['init_node']))
            term_node = str(int(row['term_node']))
            edge_pair_key = tuple(sorted([init_node, term_node]))

            # Get node coordinates
            init_coords = nodes_gdf[nodes_gdf['Node'] == int(init_node)].geometry.iloc[0]
            term_coords = nodes_gdf[nodes_gdf['Node'] == int(term_node)].geometry.iloc[0]

            # If bidirectional, offset the line slightly
            if len(edge_pairs[edge_pair_key]) > 1:
                # Create offset parallel line
                from shapely.geometry import LineString as SLineString
                line = SLineString([init_coords, term_coords])
                # Offset by 5% of the line length perpendicular to the line
                offset_dist = line.length * 0.02
                # Determine offset direction based on which edge this is
                if edge_pairs[edge_pair_key][0] == idx:
                    offset_line = line.parallel_offset(offset_dist, 'left')
                else:
                    offset_line = line.parallel_offset(offset_dist, 'right')

                # Handle potential coordinate reversal from offset
                if offset_line.coords[0] != init_coords:
                    offset_line = SLineString(list(offset_line.coords)[::-1])
                edge_geometries.append(offset_line)
            else:
                # Single direction edge, no offset needed
                edge_geometries.append(LineString([init_coords, term_coords]))

            edge_data.append({
                'init_node': init_node,
                'term_node': term_node,
                'flow': flow_data[idx] if flow_data is not None else 0,
                'capacity': row['capacity']
            })

        edges_gdf = gpd.GeoDataFrame(edge_data, geometry=edge_geometries, crs='EPSG:3857')
        
        # Plot edges with flow coloring
        if flow_data is not None:
            edges_gdf.plot(
                ax=ax,
                column='flow',
                cmap=colormap,
                linewidth=edge_width,
                alpha=edge_alpha,
                legend=True,
                legend_kwds={
                    'label': 'Traffic Flow',
                    'orientation': 'vertical',
                    'shrink': 0.5,
                    'aspect': 15,
                    'pad': 0.02
                }
            )
        else:
            edges_gdf.plot(
                ax=ax,
                color='gray',
                linewidth=edge_width,
                alpha=edge_alpha
            )
        
        # Color nodes by type
        node_colors = []
        for node in nodes_gdf['Node']:
            node_str = str(node)
            if node_str in self.env.origins and node_str in self.env.destinations:
                node_colors.append('purple')
            elif node_str in self.env.origins:
                node_colors.append('red')
            elif node_str in self.env.destinations:
                node_colors.append('blue')
            else:
                node_colors.append('green')
        
        # Plot nodes
        nodes_gdf.plot(
            ax=ax,
            color=node_colors,
            markersize=node_size,
            alpha=0.8,
            edgecolor='white',
            linewidth=0.5,
            zorder=5
        )
        
        # Add node labels if requested
        if show_node_labels:
            for idx, row in nodes_gdf.iterrows():
                ax.annotate(
                    str(row['Node']),
                    xy=(row.geometry.x, row.geometry.y),
                    fontsize=8,
                    ha='center',
                    va='center',
                    color='white',
                    weight='bold',
                    zorder=6
                )
        
        # Add basemap
        if show_background:
            try:
                ctx.add_basemap(
                    ax,
                    source=ctx.providers.OpenStreetMap.Mapnik,
                    zoom="auto",
                    alpha=0.5
                )
            except Exception as e:
                print(f"Warning: Could not add basemap: {e}")
        
        # Add legend (outside the map)
        legend_elements = [
            Patch(facecolor='red', label='Origin'),
            Patch(facecolor='blue', label='Destination'),
            Patch(facecolor='purple', label='Origin & Destination'),
            Patch(facecolor='green', label='Transfer Node')
        ]
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.15, 1),
                 loc='upper left', fontsize=10, frameon=True, fancybox=True)
        
        # Set title
        if title is None:
            title = f"Traffic Network: {self.env.loader.network_name}"
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig
    
    def _plot_with_coordinates(self, figsize, node_size, edge_width, edge_alpha,
                               show_node_labels, flow_data, colormap, title, save_path):
        """Plot network with coordinates but no geographic background."""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get positions from node coords
        pos = {}
        for _, row in self.env.node_coords.iterrows():
            node = str(int(row['Node']))
            pos[node] = (row['X'], row['Y'])
        
        # Draw edges with flow coloring
        if flow_data is not None:
            # Normalize flows for coloring
            norm_flows = flow_data / (flow_data.max() + 1e-10)
            cmap = plt.cm.get_cmap(colormap)
            
            for idx, row in self.env.net_data.iterrows():
                init_node = str(int(row['init_node']))
                term_node = str(int(row['term_node']))
                
                if init_node in pos and term_node in pos:
                    color = cmap(norm_flows[idx])
                    ax.plot(
                        [pos[init_node][0], pos[term_node][0]],
                        [pos[init_node][1], pos[term_node][1]],
                        color=color,
                        linewidth=edge_width,
                        alpha=edge_alpha,
                        zorder=1
                    )
        else:
            # Draw all edges in gray
            for idx, row in self.env.net_data.iterrows():
                init_node = str(int(row['init_node']))
                term_node = str(int(row['term_node']))
                
                if init_node in pos and term_node in pos:
                    ax.plot(
                        [pos[init_node][0], pos[term_node][0]],
                        [pos[init_node][1], pos[term_node][1]],
                        color='gray',
                        linewidth=edge_width,
                        alpha=edge_alpha,
                        zorder=1
                    )
        
        # Draw nodes
        node_colors = []
        for node in pos.keys():
            if node in self.env.origins and node in self.env.destinations:
                node_colors.append('purple')
            elif node in self.env.origins:
                node_colors.append('red')
            elif node in self.env.destinations:
                node_colors.append('blue')
            else:
                node_colors.append('green')
        
        for (node, (x, y)), color in zip(pos.items(), node_colors):
            ax.scatter(x, y, s=node_size, c=color, alpha=0.8,
                      edgecolors='white', linewidths=0.5, zorder=5)
            
            if show_node_labels:
                ax.annotate(node, xy=(x, y), fontsize=8,
                           ha='center', va='center',
                           color='white', weight='bold', zorder=6)
        
        # Legend
        legend_elements = [
            Patch(facecolor='red', label='Origin'),
            Patch(facecolor='blue', label='Destination'),
            Patch(facecolor='purple', label='Origin & Destination'),
            Patch(facecolor='green', label='Transfer Node')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        if title is None:
            title = f"Traffic Network: {self.env.loader.network_name}"
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig
    
    def _plot_without_coords(self, figsize, node_size, edge_width, title, save_path):
        """Fallback plot using spring layout."""
        fig, ax = plt.subplots(figsize=figsize)
        
        pos = nx.spring_layout(self.env.nx_graph, seed=42)
        
        node_colors = [self.env.nx_graph.nodes[node].get('color', 'gray')
                      for node in self.env.nx_graph.nodes()]
        
        nx.draw_networkx_nodes(self.env.nx_graph, pos,
                              node_color=node_colors,
                              node_size=node_size, ax=ax)
        
        nx.draw_networkx_edges(self.env.nx_graph, pos,
                              width=edge_width, alpha=0.6,
                              arrows=True, arrowsize=15, ax=ax)
        
        nx.draw_networkx_labels(self.env.nx_graph, pos,
                               font_size=10, font_color='white',
                               font_weight='bold', ax=ax)
        
        legend_elements = [
            Patch(facecolor='red', label='Origin'),
            Patch(facecolor='blue', label='Destination'),
            Patch(facecolor='purple', label='Origin & Destination'),
            Patch(facecolor='green', label='Transfer Node')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        if title is None:
            title = f"Traffic Network: {self.env.loader.network_name}"
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig


def visualize_flow_on_map(env,
                         link_flow: Optional[np.ndarray] = None,
                         figsize: Tuple[int, int] = (16, 12),
                         show_background: bool = True,
                         title: Optional[str] = None,
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Convenient function to visualize traffic flow on geographic map.
    
    Parameters
    ----------
    env : BaseTrafficEnvironment
        Traffic environment
    link_flow : np.ndarray, optional
        Link flows to visualize
    figsize : tuple
        Figure size
    show_background : bool
        Show map background
    title : str, optional
        Plot title
    save_path : str, optional
        Save path
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    visualizer = NetworkVisualizer(env)
    
    return visualizer.plot_network_on_map(
        figsize=figsize,
        show_background=show_background,
        flow_data=link_flow,
        title=title,
        save_path=save_path
    )


def animate_flow_evolution(env, algorithm, output_path='flow_animation.gif',
                          fps=5, interval_frames=5, figsize=(20, 10),
                          show_convergence_plot=True):
    """
    Create an animation showing how link flows (temperature/congestion) evolve during optimization.

    Shows side-by-side: network map with flows + convergence plot of objective function.

    Parameters
    ----------
    env : BaseTrafficEnvironment
        Traffic environment
    algorithm : NonAtomicAlgorithm
        Algorithm instance with history stored (must have store_full_arrays=True)
    output_path : str
        Path to save animation (supports .gif)
    fps : int
        Frames per second
    interval_frames : int
        Show every Nth iteration (e.g., 5 means show iters 0, 5, 10, ...)
    figsize : tuple
        Figure size (width, height)
    show_convergence_plot : bool
        If True, shows convergence plot alongside network map

    Returns
    -------
    str
        Path to saved animation file
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow required for GIF animations (pip install Pillow)")

    if 'link_flow' not in algorithm.history or len(algorithm.history['link_flow']) == 0:
        raise ValueError("Algorithm must be run with store_full_arrays=True to create animations")

    # Get flow history
    link_flow_history = algorithm.history['link_flow']
    iterations = algorithm.history['iteration']
    beckmann_history = algorithm.history['beckmann_potential']
    system_cost_history = algorithm.history['system_cost']
    avg_time_history = algorithm.history['avg_travel_time']

    # Sample frames (every interval_frames iterations)
    frame_indices = list(range(0, len(link_flow_history), interval_frames))
    if frame_indices[-1] != len(link_flow_history) - 1:
        frame_indices.append(len(link_flow_history) - 1)  # Always include final frame

    print(f"Creating animation with {len(frame_indices)} frames...")
    print("Generating frames...")

    # CRITICAL: Calculate global max flow ratio across ALL iterations
    # This ensures consistent color mapping across all frames
    capacities = env.net_data['capacity'].values
    global_max_ratio = 0.0
    for link_flow in link_flow_history:
        flow_ratio = link_flow / (capacities + 1e-10)
        global_max_ratio = max(global_max_ratio, flow_ratio.max())
    global_max_ratio = max(global_max_ratio, 1.4)  # At least 1.4 for color scale
    print(f"Global flow/capacity ratio range: [0.0, {global_max_ratio:.2f}]")

    # Generate individual frames
    frame_files = []
    import tempfile
    import os

    temp_dir = tempfile.mkdtemp()

    for idx, actual_iter in enumerate(frame_indices):
        link_flow = link_flow_history[actual_iter]
        iter_num = iterations[actual_iter] if actual_iter < len(iterations) else actual_iter
        beckmann = beckmann_history[actual_iter] if actual_iter < len(beckmann_history) else 0

        # Create frame with side-by-side layout
        frame_path = os.path.join(temp_dir, f'frame_{idx:04d}.png')

        if show_convergence_plot:
            # Create figure with 2 subplots side by side
            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.3)

            # Left: Network map
            ax_map = fig.add_subplot(gs[0])
            visualizer = NetworkVisualizer(env)

            # Compute flow-to-capacity ratio for better contrast
            flow_ratio = link_flow / (capacities + 1e-10)  # Avoid division by zero

            # Plot with enhanced colormap (use higher contrast)
            # Pass global_max_ratio to ensure consistent color mapping
            _plot_network_with_flows(
                visualizer, ax_map, link_flow, flow_ratio,
                title=f'Iteration {iter_num}',
                colormap='hot',  # High contrast: white -> red -> black
                fixed_max_ratio=global_max_ratio
            )

            # Right: Convergence plot
            ax_conv = fig.add_subplot(gs[1])

            # Get history up to current iteration
            iters_so_far = iterations[:actual_iter+1]
            beckmann_so_far = beckmann_history[:actual_iter+1]

            # Determine if log scale is needed
            use_log_scale = beckmann_history[0] / beckmann > 10

            # Plot Beckmann trajectory with professional color
            ax_conv.plot(iters_so_far, beckmann_so_far, color='#2E86AB',
                        linewidth=2.5, alpha=0.9)

            # Formatting
            ax_conv.set_xlabel('Iteration', fontsize=13, fontweight='bold')
            ax_conv.set_ylabel('Objective Value', fontsize=13, fontweight='bold')
            ax_conv.set_title(f'{algorithm.__class__.__name__}\nBeckmann Potential',
                            fontsize=14, fontweight='bold')
            ax_conv.grid(True, alpha=0.3, linestyle='--')
            # Dynamic X-axis: grows with iterations
            ax_conv.set_xlim(0, iter_num * 1.05)  # 5% padding on right

            # FIX Y-AXIS: Set fixed limits so marker visually drops from top to bottom
            # Use the full range from initial to final values
            if use_log_scale:
                ax_conv.set_yscale('log')
                ax_conv.set_ylabel('Objective Value (log scale)', fontsize=13, fontweight='bold')
                # Fixed y limits: from best final value to worst initial value
                ax_conv.set_ylim(min(beckmann_history) * 0.9, max(beckmann_history) * 1.1)
            else:
                # Fixed y limits: from best final value to worst initial value
                ax_conv.set_ylim(min(beckmann_history) * 0.95, max(beckmann_history) * 1.05)

            # Add current value text
            ax_conv.text(0.98, 0.98, f'Current:\n{beckmann:.2e}',
                       transform=ax_conv.transAxes,
                       fontsize=11, verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            plt.tight_layout()
            plt.savefig(frame_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
        else:
            # Single plot (just the map)
            visualizer = NetworkVisualizer(env)
            visualizer.plot_network_on_map(
                figsize=figsize,
                show_background=True,
                flow_data=link_flow,
                title=f'{algorithm.__class__.__name__} - Iteration {iter_num} | Beckmann: {beckmann:.2e}',
                save_path=frame_path,
                colormap='hot'  # High contrast
            )

        frame_files.append(frame_path)

        if (idx + 1) % 10 == 0:
            print(f"  Generated {idx + 1}/{len(frame_indices)} frames")

    print(f"All frames generated. Creating {output_path}...")

    # Create GIF from frames
    images = [Image.open(f) for f in frame_files]
    duration = int(1000 / fps)  # milliseconds per frame

    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )

    # Clean up temporary files
    for f in frame_files:
        os.remove(f)
    os.rmdir(temp_dir)

    print(f"Animation saved to {output_path}")
    return output_path


def _plot_network_with_flows(visualizer, ax, link_flow, flow_ratio, title, colormap='hot', fixed_max_ratio=None):
    """Helper function to plot network with enhanced flow visualization.

    Parameters
    ----------
    fixed_max_ratio : float, optional
        If provided, uses this as the maximum for color normalization.
        This ensures consistent color mapping across animation frames.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import matplotlib.cm as cm

    # Use flow ratio for coloring (0.0 = shallow white to deep dark red/black)
    # If fixed_max_ratio provided, use it for consistent coloring across frames
    if fixed_max_ratio is not None:
        max_ratio = fixed_max_ratio
    else:
        max_ratio = max(flow_ratio.max(), 1.4)  # At least 1.4 for color scale

    norm = plt.Normalize(vmin=0, vmax=max_ratio)
    cmap = cm.get_cmap(colormap + '_r')  # '_r' reverses colormap: low=white, high=black

    # Create edge collection with colors based on flow ratio
    edge_colors = []
    edge_widths = []
    segments = []

    for i, (u, v) in enumerate(visualizer.env.nx_graph.edges()):
        if i < len(flow_ratio):
            ratio = flow_ratio[i]
            edge_colors.append(cmap(norm(ratio)))
            # Thicker lines for higher flows
            edge_widths.append(2.0 + 3.0 * min(ratio, 1.0))

            # Get coordinates (convert string node IDs to int indices)
            u_idx = int(u) - 1  # Node IDs are 1-based strings, coords are 0-based int index
            v_idx = int(v) - 1
            x1, y1 = visualizer.env.node_coords.loc[u_idx, ['X', 'Y']]
            x2, y2 = visualizer.env.node_coords.loc[v_idx, ['X', 'Y']]
            segments.append([(x1, y1), (x2, y2)])

    # Plot edges
    lc = LineCollection(segments, colors=edge_colors, linewidths=edge_widths, alpha=0.8)
    ax.add_collection(lc)

    # Plot nodes
    node_x = visualizer.env.node_coords['X'].values
    node_y = visualizer.env.node_coords['Y'].values
    ax.scatter(node_x, node_y, c='black', s=30, zorder=3, alpha=0.6)

    # Set limits
    margin = 0.01
    ax.set_xlim(node_x.min() - margin, node_x.max() + margin)
    ax.set_ylim(node_y.min() - margin, node_y.max() + margin)
    ax.set_aspect('equal')

    # Add OSM basemap if available
    if HAS_GEO and visualizer.has_coords and visualizer.is_latlon:
        try:
            import contextily as ctx
            ctx.add_basemap(
                ax,
                crs='EPSG:4326',
                source=ctx.providers.OpenStreetMap.Mapnik,
                alpha=0.5,
                zorder=0
            )
        except Exception as e:
            print(f"Warning: Could not add basemap: {e}")

    ax.axis('off')

    # Add title
    ax.text(0.5, 0.98, title, transform=ax.transAxes,
           fontsize=16, fontweight='bold', ha='center', va='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # Add colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Flow / Capacity Ratio', fontsize=11, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
