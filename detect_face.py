from main import MTCNNMain

img_path = "test_images/cricket.jpg"
save_path = "test.jpg"
m = MTCNNMain(img_path)
boxes = m.detect_faces()
m.draw_square(boxes,save_path=save_path)