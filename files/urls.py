from django.urls import path 

from .views import (
  FileUploadView,
  FileListView,
  FileToolsView,
)

urlpatterns = [
  path("upload/",FileUploadView.as_view()),
  path("",FileListView.as_view()),
  path(
        "tools/",
        FileToolsView.as_view()
    ),
]