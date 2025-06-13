class PhotogrammetryHandler:
    def __init__(self):
        self.images = []

    def process_images(self, project, processed_model):
        # Process images using photogrammetry software
        # ...
        processed_model.processing_status = 'COMPLETED'
        processed_model.save()
        