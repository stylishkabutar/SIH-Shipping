import numpy as np
import geopandas as gpd
from shapely.geometry import Point, box, LineString, MultiLineString, Polygon
from dataclasses import dataclass
import requests
import io
from typing import Tuple, List
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes

@dataclass
class Coordinate:
    lat: float
    lon: float

class OceanMapper:
    def __init__(self, grid_size: int = 100):
        """
        Initialize the ocean mapper with real geographic data.
        
        Args:
            grid_size (int): Size of the square grid (k x k)
        """
        self.grid_size = grid_size
        self.coastline_data = self._load_coastline_data()
        
    def _load_coastline_data(self) -> gpd.GeoDataFrame:
        """
        Load coastline data from Natural Earth dataset.
        Returns a GeoDataFrame containing coastline geometries.
        """
        # Load land polygons instead of coastlines
        url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_land.geojson"
        
        try:
            response = requests.get(url)
            land = gpd.read_file(io.StringIO(response.content.decode('utf-8')))
            return land
        except Exception as e:
            print(f"Error loading land data: {e}")
            return gpd.GeoDataFrame()

    def _rasterize_polygon(self, polygon: Polygon, grid: np.ndarray, 
                          top_left: Coordinate, bottom_right: Coordinate):
        """
        Convert a polygon to raster format on the grid.
        """
        # Get bounds of the polygon
        minx, miny, maxx, maxy = polygon.bounds
        
        # Convert to grid coordinates
        min_y = self._normalize_coordinate(
            miny,
            min(top_left.lat, bottom_right.lat),
            max(top_left.lat, bottom_right.lat)
        )
        max_y = self._normalize_coordinate(
            maxy,
            min(top_left.lat, bottom_right.lat),
            max(top_left.lat, bottom_right.lat)
        )
        min_x = self._normalize_coordinate(
            minx,
            min(top_left.lon, bottom_right.lon),
            max(top_left.lon, bottom_right.lon)
        )
        max_x = self._normalize_coordinate(
            maxx,
            min(top_left.lon, bottom_right.lon),
            max(top_left.lon, bottom_right.lon)
        )
        
        # Sample points within the bounding box
        y_coords = np.linspace(min_y, max_y, max(2, int(max_y - min_y)))
        x_coords = np.linspace(min_x, max_x, max(2, int(max_x - min_x)))
        
        for y in y_coords:
            for x in x_coords:
                if 0 <= int(y) < self.grid_size and 0 <= int(x) < self.grid_size:
                    # Convert grid coordinates back to geographic coordinates
                    lon = self._denormalize_coordinate(
                        x,
                        min(top_left.lon, bottom_right.lon),
                        max(top_left.lon, bottom_right.lon)
                    )
                    lat = self._denormalize_coordinate(
                        y,
                        min(top_left.lat, bottom_right.lat),
                        max(top_left.lat, bottom_right.lat)
                    )
                    
                    point = Point(lon, lat)
                    if polygon.contains(point):
                        grid[int(y), int(x)] = 1

    def get_ocean_map(self, top_left: Coordinate, bottom_right: Coordinate) -> Tuple[np.ndarray, gpd.GeoDataFrame]:
        """
        Generate ocean map for the given coordinate bounds using real land data.
        
        Args:
            top_left: Coordinate of top-left corner
            bottom_right: Coordinate of bottom-right corner
            
        Returns:
            Tuple[np.ndarray, gpd.GeoDataFrame]: Grid where 1 represents land, 0 represents water,
            and the GeoDataFrame of the clipped land data
        """
        bounds = box(
            min(top_left.lon, bottom_right.lon),
            min(top_left.lat, bottom_right.lat),
            max(top_left.lon, bottom_right.lon),
            max(top_left.lat, bottom_right.lat)
        )
        
        ocean_grid = np.zeros((self.grid_size, self.grid_size))
        
        # Clip land data to our region of interest
        region_gdf = self.coastline_data.clip(bounds)
        
        # Process each land polygon
        for geometry in region_gdf.geometry:
            if geometry is not None:
                if isinstance(geometry, Polygon):
                    self._rasterize_polygon(geometry, ocean_grid, top_left, bottom_right)
                elif hasattr(geometry, 'geoms'):  # MultiPolygon
                    for poly in geometry.geoms:
                        if isinstance(poly, Polygon):
                            self._rasterize_polygon(poly, ocean_grid, top_left, bottom_right)
        
        # Fill any holes in the land masses
        ocean_grid = binary_fill_holes(ocean_grid).astype(int)
        ocean_grid=[['H' if a==1 else 'F' for a in b] for b in ocean_grid]
        ocean_grid[0][0]='S'
        ocean_grid[-1][-1]='G'
        return ocean_grid

    def _normalize_coordinate(self, coord: float, min_val: float, max_val: float) -> int:
        """Convert geographic coordinate to grid index."""
        normalized = ((coord - min_val) / (max_val - min_val)) * (self.grid_size - 1)
        return int(np.clip(normalized, 0, self.grid_size - 1))

    def _denormalize_coordinate(self, grid_coord: float, min_val: float, max_val: float) -> float:
        """Convert grid coordinate back to geographic coordinate."""
        return min_val + (grid_coord / (self.grid_size - 1)) * (max_val - min_val)

    def visualize_map(self, ocean_grid: np.ndarray, land_data: gpd.GeoDataFrame, 
                     top_left: Coordinate, bottom_right: Coordinate):
        """
        Visualize the ocean map with land areas.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot the grid representation
        im1 = ax1.imshow(ocean_grid, cmap='Blues_r')
        ax1.set_title('Grid Representation')
        plt.colorbar(im1, ax=ax1, label='Land (1) vs Water (0)')
        
        # Plot the actual land areas
        land_data.plot(ax=ax2, color='lightgreen', edgecolor='black')
        ax2.set_xlim([min(top_left.lon, bottom_right.lon), max(top_left.lon, bottom_right.lon)])
        ax2.set_ylim([min(top_left.lat, bottom_right.lat), max(top_left.lat, bottom_right.lat)])
        ax2.set_title('Actual Land Areas')
        
        plt.tight_layout()
        return fig

# Example usage
def main():
    # Initialize mapper
    mapper = OceanMapper(grid_size=100)
    
    # Define region of interest (e.g., Florida and Caribbean)
    top_left = Coordinate(lat=30.0, lon=-87.0)     # Northwest Florida
    bottom_right = Coordinate(lat=24.0, lon=-80.0)  # Southeast Florida
    
    # Get ocean map with real land data
    ocean_map, land_data = mapper.get_ocean_map(top_left, bottom_right)
    
    # Visualize the results
    fig = mapper.visualize_map(ocean_map, land_data, top_left, bottom_right)
    plt.show()
    
    # Print example section of the map
    print("\nOcean Map Section (0 = water, 1 = land):")
    print(ocean_map[0:10, 0:10])  # Print first 10x10 grid

if __name__ == "__main__":
    main()