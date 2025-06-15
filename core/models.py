from turtle import width
from django.db import models
from django.contrib.auth.models import User
import os
from django.utils import timezone


def drone_image_path(instance, filename):
    # File will be uploaded to MEDIA_ROOT/drone_images/user_<id>/<timestamp>_<filename>
    return f'drone_images/user_{instance.user.id}/{timezone.now().strftime("%Y%m%d%H%M%S")}_{filename}'


class DroneProject(models.Model):
    """Model representing a drone photogrammetry project"""
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='drone_projects')
    
    def __str__(self):
        return self.name


class DroneImage(models.Model):
    """Model representing individual drone images"""
    project = models.ForeignKey(DroneProject, on_delete=models.CASCADE, related_name='images')
    image = models.ImageField(upload_to=drone_image_path)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)
    altitude = models.FloatField(null=True, blank=True)
    diagonal_fov = models.FloatField(null=True, blank=True)
    width = models.IntegerField(null=True, blank=True)
    height = models.IntegerField(null=True, blank=True)
    flight_yaw = models.FloatField(null=True,blank=True)
    
    def __str__(self):
        return os.path.basename(self.image.name)


class ProcessedModel(models.Model):
    """Model representing processed 3D models from drone images"""
    project = models.ForeignKey(DroneProject, on_delete=models.CASCADE, related_name='processed_models')
    name = models.CharField(max_length=200)
    created_at = models.DateTimeField(auto_now_add=True)
    model_file = models.FileField(upload_to='processed_models', null=True, blank=True)
    point_cloud = models.FileField(upload_to='point_clouds', null=True, blank=True)
    orthomosaic = models.ImageField(upload_to='orthomosaics', null=True, blank=True)
    processing_status = models.CharField(
        max_length=20,
        choices=[
            ('PENDING', 'Pending'),
            ('PROCESSING', 'Processing'),
            ('COMPLETED', 'Completed'),
            ('FAILED', 'Failed'),
        ],
        default='PENDING'
    )
    
    def __str__(self):
        return self.name
