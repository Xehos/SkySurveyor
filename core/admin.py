from django.contrib import admin
from .models import DroneProject, DroneImage, ProcessedModel


@admin.register(DroneProject)
class DroneProjectAdmin(admin.ModelAdmin):
    list_display = ('name', 'user', 'created_at', 'updated_at')
    search_fields = ('name', 'description')
    list_filter = ('created_at', 'updated_at')


@admin.register(DroneImage)
class DroneImageAdmin(admin.ModelAdmin):
    list_display = ('project', 'user', 'uploaded_at', 'latitude', 'longitude', 'altitude')
    list_filter = ('uploaded_at', 'project')
    search_fields = ('project__name',)


@admin.register(ProcessedModel)
class ProcessedModelAdmin(admin.ModelAdmin):
    list_display = ('name', 'project', 'created_at', 'processing_status')
    list_filter = ('created_at', 'processing_status')
    search_fields = ('name', 'project__name')
