from rest_framework import serializers
from .models import UploadedFile 

class UploadedFileSerializer(serializers.ModelSerializer):
  folder = serializers.SerializerMethodField()

  class Meta:
    model = UploadedFile
    fields = [
      "id",
      "file",
      "original_name",
      "folder",
      "uploaded_at"
    ]
    read_only_fields = [
      "id",
      "original_name",
      "folder",
      "uploaded_at"
    ]

  def get_folder(self, obj):
    if obj.folder:
      return obj.folder.name
    return None

  def create(self, validated_data):
    uploaded_file = validated_data["file"]
    user = None
    request = self.context.get("request")
    if request and hasattr(request, "user") and request.user.is_authenticated:
      user = request.user
    return UploadedFile.objects.create(
      file=uploaded_file,
      original_name=uploaded_file.name,
      user=user,
    )