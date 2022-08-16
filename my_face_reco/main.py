import face_recognition
import numpy as np
from PIL import Image,ImageDraw
#
#
phuong = face_recognition.load_image_file('/home/Face_Recognize/avatar/SV-2019602461.jpeg')
phuong_encode = face_recognition.face_encodings(phuong)[0]
phuong_loca = face_recognition.face_locations(phuong)

phuong_test = face_recognition.load_image_file('/home/phanminhgiang/Downloads/c4affd10a252670c3e43.jpg')
phuong_test_loca = face_recognition.face_locations(phuong_test)
phuong_test_enc= face_recognition.face_encodings(phuong_test)

pil_image= Image.fromarray(phuong_test)
draw = ImageDraw.Draw(pil_image)

# Loop through each face found in the unknown image
for (top, right, bottom, left), face_encoding in zip(phuong_test_loca, [phuong_encode]):
    # See if the face is a match for the known face(s)
    matches = face_recognition.compare_faces(phuong_encode, phuong_test_enc)

    name = "Unknown"

    # If a match was found in known_face_encodings, just use the first one.
    # if True in matches:
    #     first_match_index = matches.index(True)
    #     name = known_face_names[first_match_index]

    # Or instead, use the known face with the smallest distance to the new face
    face_distances = face_recognition.face_distance([phuong_encode], face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = 'Phuong'

    # Draw a box around the face using the Pillow module
    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

    # Draw a label with a name below the face
    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
    draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))


# Remove the drawing library from memory as per the Pillow docs
del draw

# Display the resulting image
pil_image.show()
# import face_recognition
# import cv2
#
# # Lấy ảnh từ file
# minh_img = face_recognition.load_image_file(f='/home/Face_Recognize/avatar/SV-Phùng Tuấn Minh.jpg')
# giang_img = face_recognition.load_image_file(f='/home/Face_Recognize/avatar/SV-Phan Minh Giang.jpg')
# dung_img = face_recognition.load_image_file(f='/home/Face_Recognize/avatar/SV-Nguyễn Dung.jpg')
# kim_img = face_recognition.load_image_file(f='/home/Face_Recognize/avatar/SV-Hoàng Hữu Kim.jpg')
# phuong_img = face_recognition.load_image_file(f='/home/Face_Recognize/avatar/SV-Kim Phượng.jpg')
# ly_img = face_recognition.load_image_file(f='/home/Face_Recognize/avatar/SV-Lưu Khánh Ly.jpg')
# cuong_img = face_recognition.load_image_file(f='/home/Face_Recognize/avatar/SV-Phạm Mạnh Cường.jpg')
# nguyen_img = face_recognition.load_image_file(f='/home/Face_Recognize/avatar/SV-Nguyễn Khắc Nguyên.jpg')
# trang_img = face_recognition.load_image_file(f='/home/Face_Recognize/avatar/SV-Nguyễn Trang.jpg')
# huy_img = face_recognition.load_image_file(f='/home/Face_Recognize/avatar/SV_Nguyễn Văn Huy.jpg')
# gv_hop_img = face_recognition.load_image_file(f='/home/Face_Recognize/avatar/GV-Đặng Trọng Hợp.jpg')
# gv_thang_img = face_recognition.load_image_file(f='/home/Face_Recognize/avatar/GV-Nguyễn Chiến Thắng.jpg')
# gv_hoang_img = face_recognition.load_image_file(f='/home/Face_Recognize/avatar/GV-Nguyễn Xuân Hoàng.jpg')
# gv_ha_img = face_recognition.load_image_file(f='/home/Face_Recognize/avatar/GV-Trần Việt Hà.jpg')
# # Lấy 1 ảnh bất kỳ để test khuôn mặt
# unknown_face = face_recognition.load_image_file('/home/Face_Recognize/face/giang.jpg')
# # Đưa các ảnh đã lấy vào list sv( Chứa ảnh sinh viên), gv( Chứa ảnh giảng viên)
# sv = [minh_img, giang_img, dung_img, kim_img, phuong_img, ly_img, cuong_img, nguyen_img, trang_img, huy_img]
# gv = [gv_ha_img, gv_hoang_img, gv_hop_img, gv_thang_img]
# # Mã hóa ảnh theo vùng nhận diện khuôn mặt
# sv_encodings = []
# for s in sv:
#     locate = face_recognition.face_locations(s)
#     encode = face_recognition.face_encodings(s, locate)
#     sv_encodings.append(s)
#
# # results is an array of True/False telling if the unknown face matched anyone in the known_faces array
# results_sv = face_recognition.compare_faces(sv_encodings, unknown_face_encoding)
# results_gv = face_recognition.compare_faces(gv_encodings, unknown_face_encoding)
#
# for i in results_sv:
#     print(i)
#
# print()
#
# for g in results_gv:
#     print(g)
