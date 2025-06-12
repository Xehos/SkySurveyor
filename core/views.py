from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.conf import settings
from django.contrib.auth import logout

from .models import DroneProject, DroneImage, ProcessedModel
import os
import numpy as np
import cv2
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS


def index(request):
    """Home page view"""
    return render(request, 'core/index.html')


@login_required
def dashboard(request):
    """User dashboard showing their projects"""
    projects = DroneProject.objects.filter(user=request.user).order_by('-updated_at')
    return render(request, 'core/dashboard.html', {'projects': projects})


def logout_view(request):
    """Custom logout view"""
    logout(request)
    return redirect('index')


@login_required
def project_detail(request, project_id):
    """View for a specific project's details"""
    project = get_object_or_404(DroneProject, id=project_id, user=request.user)
    images = project.images.all()
    processed_models = project.processed_models.all()
    
    return render(request, 'core/project_detail.html', {
        'project': project,
        'images': images,
        'processed_models': processed_models
    })


@login_required
def create_project(request):
    """Create a new drone project"""
    if request.method == 'POST':
        name = request.POST.get('name')
        description = request.POST.get('description', '')
        
        if name:
            project = DroneProject.objects.create(
                name=name,
                description=description,
                user=request.user
            )
            messages.success(request, f'Project "{name}" created successfully!')
            return redirect('project_detail', project_id=project.id)
        else:
            messages.error(request, 'Project name is required')
    
    return render(request, 'core/create_project.html')


@login_required
def upload_images(request, project_id):
    """Upload drone images to a project"""
    project = get_object_or_404(DroneProject, id=project_id, user=request.user)
    
    if request.method == 'POST':
        images = request.FILES.getlist('images')
        
        if images:
            # Maximum file size (10MB in bytes)
            max_file_size = 10 * 1024 * 1024
            valid_images = []
            invalid_files = []
            
            for img in images:
                # Check file size
                if img.size > max_file_size:
                    invalid_files.append(img.name)
                    continue
                    
                valid_images.append(img)
            
            # Handle invalid files
            if invalid_files:
                messages.error(request, f'The following files exceed the 10MB size limit: {", ".join(invalid_files)}')
                if not valid_images:
                    return render(request, 'core/upload_images.html', {'project': project})
            
            # Process valid images
            for img in valid_images:
                altitude = None
                latitude = None
                longitude = None
                # Extract EXIF data
                try:
                    
                    with Image.open(img) as image:
                        exif = image._getexif()
                        if exif:
                            for tag_id in exif:
                                tag = TAGS.get(tag_id, tag_id)
                                data = exif.get(tag_id)
                                
                                if tag == 'GPSInfo':
                                    gps_data = {}
                                    for gps_tag in data:
                                        sub_tag = GPSTAGS.get(gps_tag, gps_tag)
                                        gps_data[sub_tag] = data[gps_tag]
                                    
                                    if 'GPSLatitude' in gps_data and 'GPSLongitude' in gps_data:
                                        lat = gps_data['GPSLatitude']
                                        lat_ref = gps_data.get('GPSLatitudeRef', 'N')
                                        lon = gps_data['GPSLongitude']
                                        lon_ref = gps_data.get('GPSLongitudeRef', 'E')
                                        
                                        # Convert to decimal degrees
                                        lat = float(lat[0] + lat[1]/60 + lat[2]/3600)
                                        if lat_ref == 'S':
                                            lat = -lat
                                            
                                        lon = float(lon[0] + lon[1]/60 + lon[2]/3600)
                                        if lon_ref == 'W':
                                            lon = -lon
                                            
                                        latitude = lat
                                        longitude = lon
                                        
                                    if 'GPSAltitude' in gps_data:
                                        try:
                                            alt = gps_data['GPSAltitude']
                                            if isinstance(alt, tuple):
                                                altitude = float(alt[0]) / float(alt[1])
                                            else:
                                                # Some cameras store altitude as a single rational number
                                                altitude = float(alt.numerator) / float(alt.denominator)
                                        except (AttributeError, IndexError, ZeroDivisionError):
                                            altitude = None
                                        
                            drone_image.save()
                except Exception as e:
                    # Log error but continue processing
                    print(f"Error extracting EXIF data: {str(e)}")
                # Here you could extract EXIF data for lat/long/altitude
                # This would require additional libraries like Pillow's ExifTags
                # Create drone image object
                drone_image = DroneImage.objects.create(
                    project=project,
                    image=img,
                    user=request.user,
                    altitude=altitude,
                    longitude=longitude,
                    latitude=latitude
                )
            if valid_images:
                messages.success(request, f'{len(valid_images)} images uploaded successfully!')
                return redirect('project_detail', project_id=project.id)
        else:
            messages.error(request, 'No images selected')
    
    return render(request, 'core/upload_images.html', {'project': project})


@login_required
#@require_POST
def process_images(request, project_id):
    """Start processing drone images for 3D reconstruction"""
    project = get_object_or_404(DroneProject, id=project_id, user=request.user)
    
    # Check if project has images
    if not project.images.exists():
        messages.error(request, 'Project has no images to process')
        return redirect('project_detail', project_id=project.id)
    
    # Create a new processing job
    model_name = f"{project.name} - {project.processed_models.count() + 1}"
    processed_model = ProcessedModel.objects.create(
        project=project,
        name=model_name,
        processing_status='PENDING'
    )
    
    # In a real application, you would start a background task here
    # For example using Celery to handle the processing asynchronously
    # For now, we'll just update the status
    processed_model.processing_status = 'PROCESSING'
    processed_model.save()
    
    messages.success(request, 'Image processing started. This may take some time.')
    return redirect('project_detail', project_id=project.id)


@login_required
def view_model(request, model_id):
    """View a processed 3D model"""
    model = get_object_or_404(ProcessedModel, id=model_id, project__user=request.user)
    
    return render(request, 'core/view_model.html', {'model': model})
