from django.shortcuts import render
from .models import Blog
from django.views.generic import ListView, DetailView

class BlogListView(ListView):
    model = Blog
    template_name = 'blog/home.html'
    context_object_name = 'item'
    ordering = ['created']

class BlogDetailView(DetailView):
    model = Blog
    slug_url_kwarg = 'question'
    slug_field = 'slug'
