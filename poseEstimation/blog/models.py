from django.db import models

class Blog(models.Model):
    title = models.CharField(blank=False, max_length=500)
    content = models.TextField(blank=False)
    created = models.DateTimeField(auto_now_add=True)
    slug = models.SlugField(max_length=50, null=False, unique=True)
    image = models.ImageField(
        upload_to='images', blank=True)
    def save(self, *args, **kwargs):
        super().save()
        self.slug = self.slug or slugify(self.title + '-' + str(self.id))
        super().save(*args, **kwargs)
    class Meta:
        verbose_name = ("Blog")
        verbose_name_plural = ("Blogs")

    def __str__(self):
        return f"{self.title}"

