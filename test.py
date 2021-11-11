from rt_utils import RTStructBuilder
image_builder = RTStructBuilder.create_from(
  dicom_series_path = "test 1",
  rt_struct_path = "test 2"
)
print(image_builder.dicom_series_path)