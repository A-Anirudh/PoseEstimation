from django.db import models

class ImageModel(models.Model):
    image_entry = models.ImageField(upload_to='cricketImages')

class ResultImageModel(models.Model):
    result_image = models.ImageField(upload_to='results')