


          
# SkySurveyor

SkySurveyor is a UAV Photogrammetry App developed for the Lablabs Zero Limits Hackathon 2025. It enables users to process aerial images from drones, generate orthomosaics, and visualize georeferenced maps with advanced alignment and overlay techniques.

## Features
- **Photogrammetry Processing:** Automatically processes drone images to create orthomosaic maps.
- **Modern Image Alignment:** Uses edge detection, feature matching, and static point analysis for precise image placement.
- **Interactive Map Visualization:** Displays processed images as overlays on interactive maps using Folium.
- **Multi-Resolution Tiling:** Supports efficient map rendering at multiple zoom levels.
- **Performance Optimizations:** Utilizes parallel processing and optimized image handling for large datasets.
- **Customizable Overlays:** Enhanced overlays with popups, markers, and custom CSS for improved user experience.

## Project Structure
```
core/
  models.py           # Django models (DroneImage, etc.)
  photogrammetry/
    PhotogrammetryHandler.py  # Main photogrammetry logic
  templates/core/     # HTML templates
  views.py            # Django views
media/                # Uploaded and processed images
static/               # Static files (CSS, JS, images)
skysurveyor/          # Django project settings and URLs
```

## Installation
1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd SkySurveyor
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run migrations:**
   ```bash
   python manage.py migrate
   ```
4. **Start the server:**
   ```bash
   python manage.py runserver
   ```

## Usage
- Upload drone images via the web interface.
- Start photogrammetry processing from the dashboard.
- View the generated orthomosaic and overlays on the interactive map.

## Key Technologies
- **Python, Django** for backend and web framework
- **OpenCV, NumPy, Pillow** for image processing
- **Folium** for map visualization
- **Concurrent Futures** for parallel processing

## Advanced Techniques
- **Edge-based and Feature-based Alignment:** Ensures accurate image stitching using Canny edge detection and ORB/SIFT feature matching.
- **Static Point Analysis:** Fine-tunes image placement by analyzing overlapping features, similar to panorama stitching.
- **Multi-Resolution Tiles:** Generates smaller, optimized PNG overlays for efficient map loading.

## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## License
MIT License

---
For more details, see the code in `core/photogrammetry/PhotogrammetryHandler.py` and related modules.

        