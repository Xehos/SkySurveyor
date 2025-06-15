import cv2
import numpy as np
import os
import folium
from PIL import Image as PILImage
import math
import uuid

class PhotogrammetryHandler:
    def __init__(self):
        self.images = []
        self.alignment_cache = {}  # Cache for edge-based alignment factors
        self.color_correction_cache = {}  # Cache for color matching
        self.feature_points_cache = {}  # Cache for feature detection
        
    def image_analysis(self,project,processed_model):
        pass
    def get_ground_elevation(self, latitude, longitude):
        import requests
        url = f"https://api.open-elevation.com/api/v1/lookup?locations={latitude},{longitude}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                return data['results'][0]['elevation']
        except Exception as e:
            print(f"Failed to fetch ground elevation: {e}")
        return 0

    def resize_image_to_hd(self, img_path, hd_path):
        try:
            with PILImage.open(img_path) as im:
                # Reduce max width for smaller files
                max_width = 600  # Reduced from 800
                if im.width > max_width:
                    new_height = int(im.height * (max_width / im.width))
                    im = im.resize((max_width, new_height), PILImage.LANCZOS)
                if im.mode != 'RGBA':
                    im = im.convert('RGB')  # Use RGB instead of RGBA when possible
                # Save with optimization and compression
                im.save(hd_path, "PNG", optimize=True, compress_level=9)
            return hd_path
        except Exception as e:
            print(f"Failed to resize image {img_path}: {e}")
            return img_path

    def apply_color_correction(self, image_path, reference_stats=None):
        """Bypassed: No color correction applied, returns original image path"""
        # Disabled color manipulation
        return image_path
    def create_multiresolution_tiles(self, image_path, bounds, zoom_levels=[15, 16]):
        """Create multi-resolution tiles with smaller file sizes"""
        try:
            tiles = {}
            base_img = PILImage.open(image_path)
            
            # Reduce base size if too large
            max_base_width = 500
            if base_img.width > max_base_width:
                new_height = int(base_img.height * (max_base_width / base_img.width))
                base_img = base_img.resize((max_base_width, new_height), PILImage.LANCZOS)
            
            for zoom in zoom_levels:
                # Calculate appropriate resolution for zoom level (smaller scale factors)
                scale_factor = 1.2 ** (zoom - 15)  # Reduced from 2 ** (zoom - 15)
                new_width = int(base_img.width * scale_factor)
                new_height = int(base_img.height * scale_factor)
                
                # Cap maximum tile size
                max_tile_size = 800
                if new_width > max_tile_size or new_height > max_tile_size:
                    if new_width > new_height:
                        new_height = int(new_height * (max_tile_size / new_width))
                        new_width = max_tile_size
                    else:
                        new_width = int(new_width * (max_tile_size / new_height))
                        new_height = max_tile_size
                
                if new_width > 0 and new_height > 0:
                    resized = base_img.resize((new_width, new_height), PILImage.LANCZOS)
                    tile_path = image_path.replace('.png', f'_z{zoom}.png')
                    # Save with maximum compression
                    resized.save(tile_path, 'PNG', optimize=True, compress_level=9)
                    tiles[zoom] = tile_path
            
            return tiles
        except Exception as e:
            print(f"Tile creation failed: {e}")
            return {}

    def detect_edges_for_alignment(self, image_path):
        """Detect edges in image for alignment reference"""
        try:
            # Load image for edge detection
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Detect edges using Canny
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours to identify major features
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Calculate edge density in different regions
            h, w = edges.shape
            regions = {
                'top': np.sum(edges[:h//4, :]) / (h//4 * w),
                'bottom': np.sum(edges[3*h//4:, :]) / (h//4 * w),
                'left': np.sum(edges[:, :w//4]) / (h * w//4),
                'right': np.sum(edges[:, 3*w//4:]) / (h * w//4),
                'center': np.sum(edges[h//4:3*h//4, w//4:3*w//4]) / (h//2 * w//2)
            }
            
            return regions
        except Exception as e:
            print(f"Edge detection failed for {image_path}: {e}")
            return None

    def calculate_alignment_factor(self, image, edge_data, adjacent_images):
        """Calculate custom alignment factor based on edge comparison"""
        if not edge_data:
            return {'lat_factor': 1.0, 'lon_factor': 1.0, 'rotation_offset': 0.0}
        
        # Base alignment factors
        lat_factor = 1.0
        lon_factor = 1.0
        rotation_offset = 0.0
        
        # Adjust based on edge density patterns
        # If more edges on one side, the image might be shifted
        edge_imbalance_threshold = 0.1
        
        # Horizontal alignment adjustment
        if edge_data['left'] - edge_data['right'] > edge_imbalance_threshold:
            lon_factor *= 0.95  # Shift slightly west
        elif edge_data['right'] - edge_data['left'] > edge_imbalance_threshold:
            lon_factor *= 1.05  # Shift slightly east
        
        # Vertical alignment adjustment
        if edge_data['top'] - edge_data['bottom'] > edge_imbalance_threshold:
            lat_factor *= 0.95  # Shift slightly south
        elif edge_data['bottom'] - edge_data['top'] > edge_imbalance_threshold:
            lat_factor *= 1.05  # Shift slightly north
        
        # Rotation adjustment based on edge patterns
        # If edges are predominantly horizontal/vertical, adjust rotation
        if edge_data['center'] > 0.3:  # High edge density in center
            # Analyze edge orientation (simplified)
            if edge_data['top'] + edge_data['bottom'] > edge_data['left'] + edge_data['right']:
                rotation_offset = 2.0  # Slight rotation adjustment for horizontal features
            else:
                rotation_offset = -2.0  # Slight rotation adjustment for vertical features
        
        return {
            'lat_factor': lat_factor,
            'lon_factor': lon_factor,
            'rotation_offset': rotation_offset
        }

    def calculate_rotated_bounds(self, image, agl, yaw_angle, alignment_factor):
        """Calculate bounds that properly account for yaw rotation"""
        # Basic FOV calculations
        aspect_ratio = image.width / image.height
        diag_fov_rad = math.radians(image.diagonal_fov)
        
        # Calculate horizontal and vertical FOV
        h_fov_rad = 2 * math.atan(aspect_ratio * math.tan(diag_fov_rad / 2) / math.sqrt(1 + aspect_ratio**2))
        v_fov_rad = 2 * math.atan(math.tan(diag_fov_rad / 2) / math.sqrt(1 + aspect_ratio**2))
        
        # Ground coverage before rotation
        ground_width = 2 * agl * math.tan(h_fov_rad / 2)
        ground_height = 2 * agl * math.tan(v_fov_rad / 2)

        # Apply 15% size increase
        ground_width *= 1.25
        ground_height *= 1.25

        # Apply alignment factors
        ground_width *= alignment_factor['lon_factor']
        ground_height *= alignment_factor['lat_factor']
        
        # Convert yaw to radians and apply rotation offset
        yaw_rad = math.radians(yaw_angle + alignment_factor['rotation_offset'])
        
        # Calculate rotated corners of the image footprint
        half_width = ground_width / 2
        half_height = ground_height / 2
        
        # Original corners (before rotation)
        corners = [
            (-half_width, -half_height),  # Bottom-left
            (half_width, -half_height),   # Bottom-right
            (half_width, half_height),    # Top-right
            (-half_width, half_height)    # Top-left
        ]
        
        # Rotate corners around center
        rotated_corners = []
        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)
        
        for x, y in corners:
            rotated_x = x * cos_yaw - y * sin_yaw
            rotated_y = x * sin_yaw + y * cos_yaw
            rotated_corners.append((rotated_x, rotated_y))
        
        # Find bounding box of rotated corners
        min_x = min(corner[0] for corner in rotated_corners)
        max_x = max(corner[0] for corner in rotated_corners)
        min_y = min(corner[1] for corner in rotated_corners)
        max_y = max(corner[1] for corner in rotated_corners)
        
        # Convert to lat/lon deltas
        earth_radius = 6378137
        dlat = (max_y - min_y) / 2 / earth_radius * (180 / math.pi)
        dlon = (max_x - min_x) / 2 / (earth_radius * math.cos(math.radians(image.latitude))) * (180 / math.pi)
        
        return dlat, dlon

    def detect_and_match_features(self, image_path1, image_path2):
        """Advanced feature detection and matching using ORB and SIFT"""
        try:
            img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
            
            if img1 is None or img2 is None:
                return None, None, []
            
            # Use ORB for fast feature detection
            orb = cv2.ORB_create(nfeatures=1000)
            kp1, des1 = orb.detectAndCompute(img1, None)
            kp2, des2 = orb.detectAndCompute(img2, None)
            
            if des1 is None or des2 is None:
                return kp1, kp2, []
            
            # FLANN matcher for better matching
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH,
                              table_number=6,
                              key_size=12,
                              multi_probe_level=1)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            
            matches = flann.knnMatch(des1, des2, k=2)
            
            # Apply Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            
            return kp1, kp2, good_matches
        except Exception as e:
            print(f"Feature matching failed: {e}")
            return None, None, []
    
    def calculate_seamline(self, image1_path, image2_path, overlap_region):
        """Calculate optimal seamline for blending overlapping images"""
        try:
            img1 = cv2.imread(image1_path)
            img2 = cv2.imread(image2_path)
            
            if img1 is None or img2 is None:
                return None
            
            # Convert to LAB color space for better color matching
            lab1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
            lab2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)
            
            # Calculate gradient magnitude for both images
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
            grad1 = cv2.Sobel(gray1, cv2.CV_64F, 1, 1, ksize=3)
            grad2 = cv2.Sobel(gray2, cv2.CV_64F, 1, 1, ksize=3)
            
            # Find optimal seam using dynamic programming
            # This is a simplified version - in practice, you'd use graph cuts
            h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
            
            # Create cost matrix based on gradient differences
            cost_matrix = np.abs(grad1[:h, :w] - grad2[:h, :w])
            
            # Find minimum cost path (simplified seam)
            seam_path = []
            for y in range(h):
                min_cost_x = np.argmin(cost_matrix[y, :])
                seam_path.append((min_cost_x, y))
            
            return seam_path
        except Exception as e:
            print(f"Seamline calculation failed: {e}")
            return None
    
    def create_interactive_overlay_with_effects(self, m, image, hd_path, bounds, edge_data, alignment_factor):
        """Create enhanced interactive overlay with modern effects, but no color correction"""
        try:
            # Skip color correction
            # corrected_path = self.apply_color_correction(hd_path)
            corrected_path = hd_path
            
            # Create multi-resolution tiles
            tiles = self.create_multiresolution_tiles(corrected_path, bounds)
            
            # Use highest resolution tile or original
            display_path = tiles.get(19, corrected_path)
            
            # Calculate opacity based on image quality metrics
            base_opacity = 0.8
            if edge_data:
                # Higher edge density = higher quality = higher opacity
                edge_quality = (edge_data['center'] + edge_data['top'] + edge_data['bottom'] + 
                              edge_data['left'] + edge_data['right']) / 5
                opacity = min(0.9, base_opacity + edge_quality * 0.2)
            else:
                opacity = base_opacity
            
            # Create enhanced overlay with custom styling
            overlay = folium.raster_layers.ImageOverlay(
                image=display_path,
                bounds=bounds,
                opacity=opacity,
                interactive=True,
                cross_origin=False,
                zindex=1
            )
            
            # Add popup with image information (simpler approach)
            center_lat = (bounds[0][0] + bounds[1][0]) / 2
            center_lon = (bounds[0][1] + bounds[1][1]) / 2
            
            quality_score = (edge_data['center'] * 100) if edge_data else 0
            
            popup_html = f"""
            <div style="text-align: center; font-family: Arial, sans-serif;">
                <h4 style="margin: 5px 0; color: #333;">Image ID: {image.id}</h4>
                <p style="margin: 3px 0;"><strong>Quality Score:</strong> {quality_score:.1f}%</p>
                <p style="margin: 3px 0;"><strong>Coordinates:</strong> {image.latitude:.6f}, {image.longitude:.6f}</p>
                <p style="margin: 3px 0;"><strong>Altitude:</strong> {image.altitude}m</p>
            </div>
            """
            
            # Add a marker with popup at the center of the image
            
            #folium.Marker(
            #    location=[center_lat, center_lon],
            #    popup=folium.Popup(popup_html, max_width=250),
            #    icon=folium.Icon(color='blue', icon='camera', prefix='fa'),
            #    tooltip=f"Image {image.id} - Click for details"
            #).add_to(m)
            
            # Add the overlay to the map
            overlay.add_to(m)
            
            # Add custom CSS for enhanced styling using HTML element
            custom_css = f"""
            <style>
            .leaflet-container {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }}
            .leaflet-popup-content {{
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            }}
            .leaflet-popup-content-wrapper {{
                border-radius: 8px;
            }}
            .leaflet-popup-tip {{
                border-radius: 2px;
            }}
            </style>
            """
            
            # Add CSS to the map
            m.get_root().html.add_child(folium.Element(custom_css))
            
            return corrected_path
        except Exception as e:
            print(f"Enhanced overlay creation failed: {e}")
            # Fallback to basic overlay
            folium.raster_layers.ImageOverlay(
                image=hd_path,
                bounds=bounds,
                opacity=0.7,
                interactive=True
            ).add_to(m)
            return hd_path
    
    def process_images(self, project, processed_model):
        processed_model.processing_status = 'PROCESSING'
        processed_model.save()
        images = project.images.all()
        os.makedirs('media/maps', exist_ok=True)
        
        # Center map on the first image with valid coordinates
        center_lat, center_lon = None, None
        for image in images:
            if image.latitude is not None and image.longitude is not None:
                center_lat, center_lon = image.latitude, image.longitude
                break
        if center_lat is None or center_lon is None:
            print("No images with valid coordinates.")
            processed_model.processing_status = 'COMPLETED'
            processed_model.save()
            return
        
        # Create enhanced map with custom styling
        m = folium.Map(
            location=[center_lat, center_lon], 
            zoom_start=18,
            tiles=None  # We'll add custom tiles
        )
        
        # Add multiple tile layers for better visualization
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Satellite',
            overlay=False,
            control=True
        ).add_to(m)
        
        folium.TileLayer(
            tiles='OpenStreetMap',
            name='Street Map',
            overlay=False,
            control=True
        ).add_to(m)
        
        hd_temp_files = []
        image_overlay_params = []
        
        # Prepare image paths for processing
        for image in images:
            if (image.latitude is not None and image.longitude is not None and 
                image.altitude is not None and image.diagonal_fov is not None and 
                image.width and image.height and image.image):
                img_path = f"media/{image.image}"
                hd_path = f"media/maps/hd_{image.id}.png"
                image_overlay_params.append((image, img_path, hd_path))
        
        # Enhanced processing with feature detection and color correction
        from concurrent.futures import ThreadPoolExecutor, as_completed
        def enhanced_processing_task(args):
            image, img_path, hd_path = args
            resized_path = self.resize_image_to_hd(img_path, hd_path)
            edge_data = self.detect_edges_for_alignment(resized_path)
            return image, resized_path, edge_data
        
        with ThreadPoolExecutor() as executor:
            future_to_param = {executor.submit(enhanced_processing_task, param): param for param in image_overlay_params}
            processed_images = []
            
            for future in as_completed(future_to_param):
                image, resized_path, edge_data = future.result()
                processed_images.append((image, resized_path, edge_data))
                if resized_path != f"media/{image.image}":
                    hd_temp_files.append(resized_path)
        
        # Sort images by quality for better layering
        def image_quality_score(item):
            _, _, edge_data = item
            if edge_data:
                return (edge_data['center'] + edge_data['top'] + edge_data['bottom'] + 
                       edge_data['left'] + edge_data['right']) / 5
            return 0
        
        processed_images.sort(key=image_quality_score)
        
        # Add enhanced overlays with modern effects
        for image, hd_path, edge_data in processed_images:
            if (image.latitude is not None and image.longitude is not None and 
                image.altitude is not None and image.diagonal_fov is not None and 
                image.width and image.height):
                
                ground_elev = self.get_ground_elevation(image.latitude, image.longitude)
                agl = image.altitude - ground_elev
                
                # Get yaw angle
                yaw_angle = image.flight_yaw if hasattr(image, 'flight_yaw') and image.flight_yaw is not None else 0
                
                # Calculate alignment factor based on edge analysis
                alignment_factor = self.calculate_alignment_factor(image, edge_data, [])
                
                # Calculate bounds with proper yaw rotation
                dlat, dlon = self.calculate_rotated_bounds(image, agl, yaw_angle, alignment_factor)
                
                # Create bounds
                bounds = [
                    [image.latitude - dlat, image.longitude - dlon],
                    [image.latitude + dlat, image.longitude + dlon]
                ]
                
                # Handle image rotation for display
                overlay_img_path = hd_path
                if yaw_angle != 0:
                    try:
                        pil_img = PILImage.open(hd_path).convert("RGBA")
                        # Apply both yaw and alignment rotation offset
                        total_rotation = yaw_angle + alignment_factor['rotation_offset']
                        rotated_img = pil_img.rotate(-total_rotation, expand=True, resample=PILImage.BICUBIC)
                        
                        # Resize if needed
                        max_width = 800
                        if rotated_img.width > max_width:
                            new_height = int(rotated_img.height * (max_width / rotated_img.width))
                            rotated_img = rotated_img.resize((max_width, new_height), PILImage.LANCZOS)
                        
                        rotated_path = hd_path.replace('.png', f'_rot{int(total_rotation)}.png')
                        rotated_img.save(rotated_path, "PNG")
                        overlay_img_path = rotated_path
                        hd_temp_files.append(rotated_path)
                    except Exception as e:
                        print(f"Error rotating image {hd_path}: {e}")
                
                # Create enhanced interactive overlay
                enhanced_path = self.create_interactive_overlay_with_effects(
                    m, image, overlay_img_path, bounds, edge_data, alignment_factor
                )
                if enhanced_path != overlay_img_path:
                    hd_temp_files.append(enhanced_path)
        
        # Add layer control for better user experience
        folium.LayerControl().add_to(m)
        
        # Add custom CSS for enhanced styling
        custom_css = """
        <style>
        .leaflet-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .leaflet-popup-content {
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        .leaflet-control-layers {
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        </style>
        """
        
        m.get_root().html.add_child(folium.Element(custom_css))
        
        # Generate a unique filename for the map
        unique_id = uuid.uuid4().hex[:8]
        map_filename = f"image_map_{unique_id}.html"
        map_path = os.path.join('media/maps', map_filename)
        m.save(map_path)
        # Save the map path to processed_model.model_file
        processed_model.model_file = map_path
        processed_model.processing_status = 'COMPLETED'
        processed_model.save()
        return hd_path
    
        # Delete temporary files
        for temp_file in hd_temp_files:
            try:
                os.remove(temp_file)
            except Exception as e:
                print(f"Failed to delete temp file {temp_file}: {e}")
        
        print(f"Enhanced orthomosaic created with {len(processed_images)} images")
        processed_model.processing_status = 'COMPLETED'
        processed_model.save()