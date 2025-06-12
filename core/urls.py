from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('projects/create/', views.create_project, name='create_project'),
    path('projects/<int:project_id>/', views.project_detail, name='project_detail'),
    path('projects/<int:project_id>/upload/', views.upload_images, name='upload_images'),
    path('projects/<int:project_id>/process/', views.process_images, name='process_images'),
    path('models/<int:model_id>/', views.view_model, name='view_model'),
    path('logout/', views.logout_view, name='logout'),
]